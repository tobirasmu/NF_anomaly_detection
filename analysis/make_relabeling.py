import matplotlib.pyplot as plt
import torch

from tools.reader import load_GHL_data
import config

cfg = config.Config()

"""
Here we are performing a bit of relabeling / preprocessing of the GHL dataset. One change is that we get rid of the
warm-up phase of approximately 6000 values (time it takes for the receiving tank to get up to average temperature for 
the first time)
The other thing we are doing, is to relabel some of the time-stamps to better fit with an actual change in the data and
distribution, by manually analyzing the behavior of the training time series (the one long that is without outliers).
"""

datadir = cfg.dir.data.joinpath('GHL')
X, L, T = load_GHL_data(datadir, saved=False, relabeled=False)

for i in range(len(X)):
    X[i] = X[i][6000:]
    L[i] = L[i][6000:]
    T[i] = T[i][6000:]

# %% Now walking through the instances one after one adding outliers
i = 0
x = X[i]
l = L[i]

_fr, _to = 0, x.shape[0]
fb, fe = 42747, 51380
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %% Now walking through the instances one after one adding outliers
i = 1
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 37332, 48565
l[fb:fe] = 1.0
fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
fig.show()
plt.close(fig)

# %%
i = 2
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 47103, 58330
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 3
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 51396, 62530
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 4
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 55632, 65525
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 5
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 60910, 71760
l[fb:fe] = 1.0

fb, fe = 100100, 101676
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 6
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 70237, 79285
l[fb:fe] = 1.0

fb, fe = 19600, 21259
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 7
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 70125, 79435
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 8
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 77478, 88650
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 9
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 76635, 86425
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 10
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 88066, 96335
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 11
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 92129, 101885
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 12
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 100326, 110960
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 13
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 105028, 116010
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 14
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 111337, 120040
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 15
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 111553, 122140
l[fb:fe] = 1.0

fb, fe = 178000, 179683
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 16
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 122710, 133565
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 17
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 119069, 130230
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 18
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 132628, 139990
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 19
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 128989, 140190
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
ax[1].plot([0, x.shape[0]], [310, 310])
fig.show()
plt.close(fig)

# %%
i = 20
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 143795, 153380
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 21
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 141496, 152245
l[fb:fe] = 1.0

fb, fe = 116750, 118422
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 22
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 155295, 166450
l[fb:fe] = 1.0

fb, fe = 167100, 168766
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 23
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 153349, 163380
l[fb:fe] = 1.0

fb, fe = 67800, 69208
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 24
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
# _fr, _to = 97500, 102500
fb, fe = _fr, _to

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
ax[1].plot([0, x.shape[0]], [310, 310])
fig.show()
plt.close(fig)

# %%
i = 25
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 49000, 50752
l[fb:fe] = 1.0

fb, fe = 78300, 79912
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 26
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 143000, 144641
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 27
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 4000, 5409
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 28
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 191800, 193357
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 29
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
# _fr, _to = 97500, 102500
fb, fe = _fr, _to

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 30
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
# _fr, _to = 97500, 102500
fb, fe = _fr, _to

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
ax[1].plot([0, x.shape[0]], [310, 310])
fig.show()
plt.close(fig)

# %%
i = 31
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
# _fr, _to = 97500, 102500
fb, fe = _fr, _to

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 32
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 32400, 33950
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 33
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
# _fr, _to = 97500, 102500
fb, fe = _fr, _to

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
ax[1].plot([0, x.shape[0]], [310, 310])
fig.show()
plt.close(fig)

# %%
i = 34
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
# _fr, _to = 97500, 102500
fb, fe = _fr, _to

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
ax[1].plot([0, x.shape[0]], [310, 310])
fig.show()
plt.close(fig)

# %%
i = 35
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
# _fr, _to = 97500, 102500
fb, fe = _fr, _to

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
ax[1].plot([0, x.shape[0]], [310, 310])
fig.show()
plt.close(fig)

# %%
i = 36
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 84000, 85567
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 37
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
# _fr, _to = 97500, 102500
fb, fe = _fr, _to

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
ax[1].plot([0, x.shape[0]], [310, 310])
fig.show()
plt.close(fig)

# %%
i = 38
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
# _fr, _to = 97500, 102500
fb, fe = _fr, _to

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
ax[1].plot([0, x.shape[0]], [310, 310])
fig.show()
plt.close(fig)

# %%
i = 39
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
# _fr, _to = 97500, 102500
fb, fe = _fr, _to

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
ax[1].plot([0, x.shape[0]], [310, 310])
fig.show()
plt.close(fig)

# %%
i = 40
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
# _fr, _to = 97500, 102500
fb, fe = _fr, _to

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
ax[1].plot([0, x.shape[0]], [310, 310])
fig.show()
plt.close(fig)

# %%
i = 41
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 169600, 171332
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 42
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
# _fr, _to = 97500, 102500
fb, fe = _fr, _to

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
ax[1].plot([0, x.shape[0]], [310, 310])
fig.show()
plt.close(fig)

# %%
i = 43
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 19700, 21325

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 44
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 37500, 39134
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %%
i = 45
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
# _fr, _to = 97500, 102500
fb, fe = _fr, _to

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
ax[1].plot([0, x.shape[0]], [310, 310])
fig.show()
plt.close(fig)

# %%
i = 46
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = _fr, _to

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
ax[1].plot([0, x.shape[0]], [310, 310])
fig.show()
plt.close(fig)


# %%
i = 47
x = X[i]
l = L[i]
_fr, _to = 0, x.shape[0]
fb, fe = 142300, 143927
l[fb:fe] = 1.0

fig, ax = plt.subplots(3, 2, figsize=(20, 10))
ax = ax.reshape(-1)
for i in range(5):
    ax[i].plot(torch.arange(_fr, _to).numpy(), x[_fr:_to, i].numpy())
    ax[i].plot(torch.arange(fb, fe).numpy(), x[fb:fe, i].numpy(), c='red')
ax[-1].plot(torch.arange(_fr, _to).numpy(), l[_fr:_to].numpy())
ax[-1].plot(torch.arange(fb, fe).numpy(), l[fb:fe].numpy(), c='red')
fig.show()
plt.close(fig)

# %% Saving the files
torch.save((X, L, T), datadir.joinpath('relabeled_loaded_data.pt'))
