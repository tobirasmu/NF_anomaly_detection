import os
import sys

sys.path.append(os.getcwd())
# %%
import torch.nn as nn
import torch
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from scipy.interpolate import interp1d
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc

from tools.dataset import  GHL_new, NF_NEW
from tools.reader import load_GHL_data
from models.AE import auto_encoder
from models.NF import MAF
import config

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

cfg = config.Config()

parser = argparse.ArgumentParser()
parser.add_argument('-ws', '--window-size', default=2400, type=int)
parser.add_argument('-ed', '--emb-dim', default=30, type=int)
parser.add_argument('-ae', '--autoencoder-epochs', default=100, type=int)
parser.add_argument('-nfe', '--nf-flow-epochs', default=100, type=int)
parser.add_argument('-ni', '--num-iterations', default=100, type=int)
parser.add_argument('-aq', '--abnorm-quantile', default=0.01, type=float)
args = parser.parse_known_args()[0]
WINDOW_SIZE = args.window_size
EMB_DIM = args.emb_dim
AMB_QUANTILE = args.abnorm_quantile

if torch.cuda.is_available():
    print('Using CUDA')
    torch.cuda.manual_seed(43)
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')
print(f"Training autoencoder with window size {WINDOW_SIZE}")

# Loading the data
X, L, _ = load_GHL_data(cfg.dir.data.joinpath('GHL'), saved=True, relabeled=True)

# Resultsfile
resultsdir = cfg.dir.results
os.makedirs(resultsdir, exist_ok=True)
resultsfile = resultsdir.joinpath('results_multirep.csv')
if not os.path.exists(resultsfile):
    with open(resultsfile, 'w') as file:
        csv.writer(file).writerow(['rep', 'iteration', 'curve', 'MSE', 'Log-likelihood',
                                   'LL-MSEs', 'z-mahal', 'Mahalanobis', 'all'])
        file.close()

for rep in range(5):
    if rep == 1:
        savefolder = resultsdir.joinpath(f'running_results')
        os.makedirs(savefolder, exist_ok=True)
    torch.manual_seed(rep * 42)

    # Defining the autoencoder training set
    dataset = GHL_new(X, L, window_size=WINDOW_SIZE, stride_size=WINDOW_SIZE // 5,
                      obs_weights=None)  # First iteration and here we only iterate wrt. the NF
    # Assembling in one big dataset
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    # %% Define and train the autoencoder
    print(f"{f' REPETITION {rep} ':-^100s}")
    print(f"{' Training the autoencoder part ':-^100s}")

    autoencoder = auto_encoder(5, kernel_size=5, channels=256, emb_dim=EMB_DIM,
                               window_size=WINDOW_SIZE, variational=False).to(device)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9)
    criterion = torch.nn.MSELoss(reduction='none')

    for epoch in range(args.autoencoder_epochs):
        autoencoder.train()

        running_loss = None
        train_bar = tqdm(train_loader)
        for batch, _, obs_weight in train_bar:
            enc, dec = autoencoder(batch.to(device))

            loss = criterion(batch.to(device), dec).sum(axis=(1, 2)) * obs_weight.to(
                device)  # Mean squared error between the decoding and the input
            loss = loss.sum()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)  # Clipping gradient for smoother learning
            optimizer.step()

            if running_loss is None:
                running_loss = loss.cpu().detach().numpy()
            else:
                running_loss = 0.9 * running_loss + 0.1 * loss.cpu().detach().numpy()
            train_bar.set_description(
                f"Epoch {epoch: >2d}/{args.autoencoder_epochs - 1} (LR: {optimizer.param_groups[0]['lr']: >7.4f}). Loss: {running_loss: >11.3f}")
        scheduler.step()

    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    autoencoder.eval()

    eval_criterion = torch.nn.MSELoss(reduction='none')
    eval_bar = tqdm(eval_loader)
    encoded_data = []
    MSE = []
    Ls = []
    with torch.no_grad():
        for batch, l, _ in eval_bar:
            enc, dec = autoencoder(batch.to(device))
            mse = eval_criterion(batch.to(device), dec).sum(dim=(1, 2))
            encoded_data.append(enc.detach().cpu())
            MSE.append(mse.detach().cpu())
            Ls.append(l)

    encoded_data = torch.concatenate(encoded_data, dim=0)
    MSE = torch.concatenate(MSE, dim=0)
    Ls = torch.concatenate(Ls, dim=0)

    # %% Saving the autoencoder model and the encoded values
    if rep == 1:
        torch.save(autoencoder.state_dict(), savefolder.joinpath('AE_model.pt'))
        torch.save(encoded_data.detach().cpu(), savefolder.joinpath('embeddings.pt'))

    # Initializing the weights of the model
    observation_weights = dataset.obs_weights

    for iteration in range(args.num_iterations):
        """
        Running over the iterations where the weights will be updated and NF models trained again.
        """
        if rep == 1:
            iteration_savefolder = savefolder.joinpath(f'iteration_{iteration}')
            os.makedirs(iteration_savefolder, exist_ok=True)
        # Making NF train loaders
        nf_train_loader = torch.utils.data.DataLoader(NF_NEW(encoded_data, Ls, MSE, observation_weights), batch_size=128,
                                                      shuffle=True)
        nf_eval_loader = torch.utils.data.DataLoader(NF_NEW(encoded_data, Ls, MSE, observation_weights), batch_size=128,
                                                     shuffle=False)

        print(f"{'':-^100s}")
        print(f"{f'Iteration {iteration} - Training the normalizing flow part':-<100s}")
        print(f"{'':-^100s}")
        # For the normalizing flow, we want to save the results after each iteration, i.e. the log-likelihood
        normalizing_flow = MAF(n_blocks=32, input_size=EMB_DIM, hidden_size=256, n_hidden=1, cond_label_size=None,
                               batch_norm=False, activation='tanh').to(device)

        optimizer = torch.optim.Adam(normalizing_flow.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9)

        all_logLiks = torch.zeros((args.nf_flow_epochs, len(nf_train_loader.dataset)))
        for epoch in range(args.nf_flow_epochs):
            normalizing_flow.train()
            train_bar = tqdm(nf_train_loader)
            running_loss = None

            for batch, _, _, obs_weight in train_bar:
                loss = - (normalizing_flow.log_prob(batch.to(device)) * obs_weight.to(device)).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(normalizing_flow.parameters(), 1.0)  # Clipping gradient for smoother learning
                optimizer.step()

                if running_loss is None:
                    running_loss = loss.cpu().detach().numpy()
                else:
                    running_loss = 0.9 * running_loss + 0.1 * loss.cpu().detach().numpy()
                train_bar.set_description(
                    f"Epoch {epoch: >2d}/{args.nf_flow_epochs - 1} (LR: {optimizer.param_groups[0]['lr']: >7.4f}). Loss: {running_loss: >11.3f}")
            scheduler.step()

            # Saving the log-likelihood of all the samples
            normalizing_flow.eval()
            logLiks = []
            with torch.no_grad():
                eval_bar = tqdm(nf_eval_loader)
                eval_bar.set_description('Evaluating all samples for saving')
                for batch, _, _, _ in eval_bar:
                    logLiks.append(normalizing_flow.log_prob(batch.to(device)).detach().cpu())
            all_logLiks[epoch] = torch.concatenate(logLiks, dim=0)

        # Saving the log-likelihood of all the samples
        embeddings, z_embeddings = [], []
        with torch.no_grad():
            eval_bar = tqdm(nf_eval_loader)
            for batch, _, _, _ in eval_bar:
                enc, _ = normalizing_flow(batch.to(device))
                z_embeddings.append(enc.detach().cpu())

        z_embeddings = torch.concatenate(z_embeddings, dim=0)

        # %% Saving the nf_model and the log-likelihood matrix
        if rep == 1:
            torch.save(z_embeddings, iteration_savefolder.joinpath('z-vals.pt'))
            torch.save(all_logLiks, iteration_savefolder.joinpath('log-likelihoods.pkl'))
            if iteration == 80:
                torch.save(normalizing_flow.state_dict(), cfg.dir.results.joinpath('running_results/nf_model_iteration_80.pt'))

        # %% Model evaluation and subsequent updating of weights
        # Training the interpolator
        sampled_densities = []
        with torch.no_grad():
            for i in tqdm(range(100)):
                x_rand = torch.randn((1000, EMB_DIM)).to(device)
                _, jacobian = normalizing_flow.inverse(x_rand)
                sampled_densities.append((torch.distributions.Normal(0, 1).log_prob(x_rand) - jacobian).sum(axis=1).detach().cpu())

        estimated_density = torch.concatenate(
            [all_logLiks[-1].min()[None], torch.concatenate(sampled_densities), all_logLiks[-1].max()[None]])
        estimated_density = torch.sort(estimated_density).values
        estimated_probabilities = torch.linspace(1e-10, 1, 100002)
        interpolator = interp1d(estimated_density, estimated_probabilities)

        # Calculating the mahalanobis distances
        enc_mean = encoded_data.mean(dim=0)
        try:
            S_inv = torch.linalg.inv(torch.cov(encoded_data.T))
        except torch._C._LinAlgError:
            S_inv = torch.eye(encoded_data.shape[1])
        diff_vec = encoded_data - enc_mean[None, :]
        mahalanobis = torch.diag(torch.sqrt(torch.mm(torch.mm(diff_vec, S_inv), diff_vec.T))).numpy()

        # Calculating the z-space mahalanobis distances
        mu = z_embeddings.mean(dim=0)
        try:
            S_inv = torch.linalg.inv(torch.cov(z_embeddings.T))
        except torch._C._LinAlgError:
            S_inv = torch.eye(z_embeddings.shape[1])
        diff_vec = z_embeddings - mu[None, :]
        mahal_sq = torch.abs(torch.diag(torch.mm(torch.mm(diff_vec, S_inv), diff_vec.T)))

        # Gathering everything together
        ll = all_logLiks[-1]
        ll = 1.0 - torch.tensor(interpolator(ll))

        df = pd.DataFrame({'Log-likelihood': ll, 'MSE': MSE, 'Mahalanobis': torch.tensor(mahalanobis),
                           'LL-MSEs': torch.sqrt(ll * MSE),
                           'LL-Mahalanobis': torch.sqrt(ll * mahalanobis),
                           'Mahal-MSEs': torch.sqrt(torch.tensor(mahalanobis) * MSE),
                           'all': ll * MSE * torch.tensor(mahalanobis) ** (1 / 3),
                           'abnormal': Ls.sum(-1) > 0, 'z-mahal': torch.sqrt(mahal_sq)})

        # Updating the weights of the observations
        new_observation_weights = ll  # * MSE  # After relabelling, MSE is no longer a good indicator of outlyingness.
        new_observation_weights -= new_observation_weights.min()
        new_observation_weights /= new_observation_weights.max()
        new_observation_weights = 1.0 - new_observation_weights

        # Finding which observations to dedact a weight from
        positive_w = observation_weights > 0
        threshold = torch.quantile(new_observation_weights[positive_w], AMB_QUANTILE)
        observation_weights[new_observation_weights <= threshold] -= 0.1
        observation_weights = torch.maximum(observation_weights, torch.tensor(0.0))
        if rep == 1:
            torch.save(observation_weights, iteration_savefolder.joinpath('observation_weights.pt'))

        pr_score, roc_score = [], []
        variables = ['MSE', 'Log-likelihood', 'LL-MSEs', 'z-mahal', 'Mahalanobis', 'all']
        num_nan = []
        for vname in variables:
            variable = df[vname]
            num_nan.append(int(torch.sum(torch.isnan(torch.tensor(df[vname].values)))))
        print(num_nan)

        for vname in variables:
            variable = df[vname]
            prec, rec, th = precision_recall_curve(df.abnormal, variable)
            roc_score.append(roc_auc_score(df.abnormal, variable))
            pr_score.append(auc(rec, prec))

        with open(resultsfile, 'a') as file:
            csv.writer(file).writerow([rep,
                                       iteration,
                                       'roc',
                                       *roc_score])
            csv.writer(file).writerow([rep,
                                       iteration,
                                       'pr',
                                       *pr_score])
            file.close()
