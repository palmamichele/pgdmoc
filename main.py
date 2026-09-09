import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import time
from pathlib import Path

from pgd_moc import pgd_moc
sys.path.append(str(Path("fmca") / "build" / "py"))
import FMCA

import torch

np.random.seed(0)
torch.manual_seed(0)


n_points = 1000
nbins=10 #for DMOC grid construction
norm = "EUCLIDEAN" #refers to d_x
FUNCTION = "sin" #choose function to test
DOMAIN_MIN = -50.0
DOMAIN_MAX = 50.0
batch_size= n_points
nrestart=1
M=40


def f(x):

    if FUNCTION == "linear":
        return 2.0 * x + 3.0

    elif FUNCTION == "sin":
        return torch.sin(x)

    elif FUNCTION == "quadratic":
        return x ** 2

    elif FUNCTION == "tanh":
        return torch.tanh(x)

    else:
        raise ValueError(
            f"Not implemented"
        )

def analytical_moc(t):

    t = np.asarray(t)

    if FUNCTION == "linear":

        return 2.0 * t

    elif FUNCTION == "sin":

        return np.where(
        t <= np.pi,
        2.0 * np.sin(t / 2.0),
        2.0
    )



    elif FUNCTION == "quadratic":

        return np.where(
            t <= 50.0,
            t * (100.0 - t),
            2500.0
        )

    elif FUNCTION == "tanh":

        return 2.0 * np.tanh(t / 2.0)

    else:
        raise ValueError(
            f"Not implemented"
        )


def box_clipping(x):
    a=DOMAIN_MIN
    b=DOMAIN_MAX
    return torch.clamp(x, a,b)


def l2_distance(x,y):
    diff = x-y
    return diff.flatten(1).norm(p=2, dim=1)


X_np = np.random.uniform(
    DOMAIN_MIN,
    DOMAIN_MAX,
    n_points
)

#make X and Y as 2D arrays: (n_points, 1)
X_np = X_np.reshape(-1, 1)

X_torch = torch.tensor(
    X_np,
    dtype=torch.float64
)

with torch.no_grad():
    Y_torch = f(X_torch)


#convert to numpy for FMCA
Y_np = Y_torch.numpy()


#FMCA expects: (n_features, n_points) numpy float64
X = X_np.T.astype(np.float64)
Y = Y_np.T.astype(np.float64)


#compute dmax, and DMOC (c_moc)
dmoc = FMCA.DiscreteModulusOfContinuity()
start_time = time.time()  
dmoc.init(X,Y, None,None, nbins, norm, norm)
c_moc = dmoc.omegat()
data_t = time.time() - start_time 
t_values = dmoc.tgrid()
dmax = t_values[-1]


#gt analytical moc
gt_moc = analytical_moc(t_values)

#compute PGD-moc (p_moc)
start_time = time.time()

p_moc, _ = pgd_moc(
    f, #f_\theta
    X_torch,
    Y_torch, #either f_\theta(X) or original labels for X
    l2_distance, #d_Y as loss function (assuming it satisfies metric properties)
    box_clipping,
    "L2", #L2, L1, Linf
    t_values, #t_1,...,t_K
    None, 
    M,
    nrestart,
    nbins,
    batch_size
)

pgd_t = time.time() - start_time
print(f"DMOC time:    {data_t:.6f} s")
print(f"PGD-MOC time: {pgd_t:.6f} s")


#plots
plt.figure(figsize=(8, 5))
plt.plot(
    t_values,
    c_moc,
    label="DMOC",
    linewidth=2
)


plt.plot(
    t_values,
    p_moc,
    "o-",
    label="PGD-MOC",
    linewidth=2,
    markersize=4
)


plt.plot(
    t_values,
    gt_moc,
    "--",
    label=f"Analytical MOC {FUNCTION}",
    linewidth=2
)

plt.xlabel(r"$t$")
plt.ylabel(r"$\omega(t)$")

plt.title(
    f"DMOC vs PGD-MOC vs Analytical MOC\n n_points={n_points}, nbins={nbins}, norm={norm}, DMOC(s) ={data_t:.3f}, PGD-MOC(s)={pgd_t:.3f}"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(f'{FUNCTION}_{norm}_{n_points}_{nbins}.png')