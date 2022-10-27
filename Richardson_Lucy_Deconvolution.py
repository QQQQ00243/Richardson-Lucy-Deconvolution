import numpy as np


def norm2(x1, x2):
    delta_squared = (x1 - x2)**2
    return np.sqrt(np.sum(delta_squared))

def R_L_deconvolve(o, H, max_iter, eps=1e-3):
    x = o
    converge = False
    for _ in range(max_iter):
        E = np.dot(H, x)
        relative = o / E
        new = x * relative
        if norm2(new, x) < eps:
            converge = True
            break
        else:
            x = new
    if not converge:
        raise Warning(f"Exit without converging for threshold {eps} with\
             number of iterations {max_iter}. Try increasing the max_iter.")
    return x
