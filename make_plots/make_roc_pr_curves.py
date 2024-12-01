import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt
import matplotlib.colors as cols
import matplotlib as mpl

import config

cfg = config.Config()

mpl.rc('font', family="Times New Roman")
plt.rcParams['font.size'] = 15
plt.rcParams['mathtext.fontset'] = 'cm'
colors = [cols.to_rgb(color_name) for color_name in ['C0', 'C1', 'C2', 'C3', 'C4']]
figwidth = 5.765


curvedir = cfg.dir.results.joinpath('curves')

curves = [pd.read_csv(curve) for curve in list(curvedir.glob('*'))]
num_curves = len(curves)
for i in range(num_curves):
    curves[i]['repetition'] = i
curves = pd.concat(curves)

name_dict = {'Log-likelihood': 'Estimated density',
             'MSE': 'Reconstruction error',
             'Mahalanobis': 'Mahalanobis distance (embedding space)'}
for variable in ['Log-likelihood', 'MSE', 'Mahalanobis']:
    fig, ax = plt.subplots(1, 2, figsize=(figwidth * 2, figwidth))
    for i in range(num_curves):
        this_curve = curves[curves.repetition == i]

        fpr, tpr, _ = roc_curve(this_curve.abnormal, this_curve[variable])
        ax[0].plot(fpr, tpr, color=f'C{i}', linewidth=3, alpha=0.8, label=f"AUC: {auc(fpr, tpr):.3f}")

        prec, rec, _ = precision_recall_curve(this_curve.abnormal, this_curve[variable])
        ax[1].plot(rec, prec, color=f'C{i}', linewidth=3, alpha=0.8, label=f"AUC: {auc(rec, prec):.3f}")
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('False Positive Rate')
    ax[1].set_xlabel('Recall')
    ax[0].set_ylabel('True Positive Rate')
    ax[1].set_ylabel('Precision')
    fig.suptitle(name_dict[variable], weight='bold')
    fig.savefig(cfg.dir.plots.joinpath(f'curves_{variable}.png'), dpi=200, bbox_inches='tight')
    fig.show()
