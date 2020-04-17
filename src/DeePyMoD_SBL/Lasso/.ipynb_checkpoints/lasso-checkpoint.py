import torch

def soft_threshold(x, alpha):
    return torch.max(torch.abs(x) - alpha, torch.zeros_like(x)) * torch.sign(x)

def lasso(X, y, l1, conv_tol=torch.tensor(1e-7), w=None):
    L = 2 * torch.symeig(X.T @ X).eigenvalues.max()
    
    if w == None:
        w = torch.zeros((X.shape[1], 1), dtype=torch.float32) # if not given, start with zeros, else do warm restart
        
    t = torch.tensor(1.0)
    z = w
    
    converged=False
    while not converged:
        w_old = w
        t_old = t
        z_old = z
        
        w = soft_threshold(z_old - 2 * X.T @ (X @ z_old - y) / L, l1/L)
        t = (1 + torch.sqrt(1 + 4 * t_old**2))/2
        z = w + (t_old - 1) / t * (w - w_old)
        converged = torch.max(torch.abs(w - w_old)) < conv_tol
    return w