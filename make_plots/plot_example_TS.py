"""
Plotting the example time series from the GHL dataset.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

from tools.reader import load_GHL_data
import config

cfg = config.Config()

mpl.rc('font', family="Times New Roman")
plt.rcParams['font.size'] = 15
plt.rcParams['mathtext.fontset'] = 'cm'
figwidth = 5.765

X, L, T = load_GHL_data(data_dir=cfg.dir.data.joinpath('GHL'), saved=True, relabeled=True)
x = X[1]

ylabels = ['RT vol.', 'RT Temp.', 'HT Temp.', 'Inj. On', 'Heat On']
fig, axes = plt.subplots(5, 1, figsize=(figwidth*2, 3*2))
for i in range(5):
    axes[i].plot(x[125000:150000, i].numpy())
    axes[i].set_ylabel(ylabels[i])
    # Removing the axis apart of the bottom one
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    if i < 4:
        axes[i].spines['bottom'].set_visible(False)
        axes[i].xaxis.set_ticks([])
    else:
        axes[i].set_xlabel('Time')

fig.savefig('plots/GHL_example.png', dpi=400, bbox_inches='tight')
plt.show()
del X, L, T
