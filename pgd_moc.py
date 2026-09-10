import torch
import numpy as np

# class PGD_MOC:
#     def __init__(self, model, dy_f, clipping_f, norm, t_values=None, step_size=None, numiter=1, numrestarts=1, nbins=100):
#         self.model = model
#         self.dy_f = dy_f
#         self.clipping_f = clipping_f
#         self.norm = norm
#         self.t_values = t_values
#         self.step_size = step_size
#         self.numiter = numiter
#         self.numrestarts = numrestarts
#         self.nbins = nbins


#     def compute_moc(self, X, Y):
#         return pgd_moc(
#             self.model,
#             X,
#             Y,
#             self.dy_f,
#             self.clipping_f,
#             self.norm,
#             self.t_values,
#             self.step_size,
#             self.numiter,
#             self.numrestarts,
#             self.nbins
#         )


def project_linf(x, a, epsilon):
    """
    Project point x onto the L_infty ball centered at a with radius epsilon
    """
    return torch.clamp(x, a - epsilon, a + epsilon)


def project_l2(x,a, epsilon):
    """
    Project each point in x onto the L2 ball centered at a with radius epsilon
    """

    diff = x-a
    norm = diff.flatten(1).norm(p=2, dim=1)
    #norm = torch.linalg.vector_norm(diff)

    scale = torch.clamp(epsilon / (norm + 1e-12), max=1.0)
    scale = scale.view(-1, *([1] * (x.ndim - 1)))

    return a + scale * diff

def initialize_perturbation(x,t, norm):

    if norm == "L2":
        delta = torch.randn_like(x)

        delta_norm = delta.flatten(1).norm(p=2, dim=1, keepdim=True)
        delta_norm = delta_norm.view(-1, *([1] * (x.ndim - 1)))

        delta = delta / (delta_norm + 1e-12)

        u = torch.rand(
            x.shape[0], 1,
            device=x.device,
            dtype=x.dtype
        )

        dimension = x[0].numel()
        radius = t * u.pow(1.0 / dimension)

        delta = delta * radius.view(
            -1, *([1] * (x.ndim - 1))
        )

        #return torch.zeros_like(x)
        return delta

        

    elif norm == "Linf":
        return torch.empty_like(x).uniform_(-t, t)

    else:
        raise Exception(f"Sorry, no implementation for the specified norm: {norm}")

# def box_clipping(x):
#     #for MNIST in [0,1]^n
#     a=0
#     b=1
#     return torch.clamp(x, a,b)


#check independence assumption of obs in the batch
def pgd_m_x(
    model,
    x,
    y,
    t,
    dy_f, #
    norm,
    clipping_f=None,
    M=40,
    step_size=None,
    n_restarts=1
):
    """
    x: a batch of datapoints assumed to be in set S.
    
    y: a batch of datapoints, can be either network output on x i.e. model(x), or original label value for x 
    
    dy_f: needs to return per-example values.

    M: int, the number of steps for PGD

    clipping: is a function for the projection of x+delta on a constrained feasible set S, needs to be implemented by the user as it is problem specific. e.g. inputs must live in [0,1]^n

    Notice if y is original value for x, and dy_f is the (criterion) loss function, this just reduces to usual PGD-Attack.
    More generally, this can be used in pgd_moc to compute a moc, under the assumption that dy_f is a metric.  

    
    """


    if norm=="L1":
        raise Exception(f"Sorry, no implementation for the specified norm: {norm}")
    elif norm=="L2":
        # dx_f = np.linalg.norm 
        # proj_f = project_l2 
        pass
    elif norm=="Linf":
        # dx_f = lambda x: np.linalg.norm(x, ord=np.inf)
        # proj_f = project_linf
        pass
    else:
        raise Exception(f"Sorry, no implementation for the specified norm: {norm}")


    if t == 0:
        B = x.shape[0]
        return (
            torch.zeros(B, device=x.device, dtype=x.dtype),
            torch.zeros_like(x),
        )

    if step_size is None:
        step_size = t / 10.0

    B = x.shape[0]

    #network pars are fixed throughout the optimization
    best_value = torch.full(
    (B,),
    -float("inf"),
    device=x.device,
    dtype=x.dtype
    )

    best_delta = torch.zeros_like(x)
    

    for _ in range(n_restarts):

        #initialize the adv perturbation, based on the norm type 
        delta = initialize_perturbation(x,t,norm)
        delta= delta.detach().requires_grad_(True)
        #zero_center = torch.zeros_like(delta)


        #PGD
        for _ in range(M):

            if clipping_f!= None:
                adv_x = clipping_f(x+delta)
            adv_output = model(adv_x)
         
            #adv_x.requires_grad_(True)
            objective = dy_f(adv_output,y) #should return shape (B,)

            if objective.ndim != 1 or objective.shape[0] != B:
                raise ValueError(
                    f"dy_f must return shape ({B},), "
                    f"got {tuple(objective.shape)}"
                )

            grad = torch.autograd.grad(
                objective.sum(),
                delta
            )[0]

            #gradient of each example's objective w.r.t. its own delta assuming examples are independent

            #gradient step update based on the specific norm, returns updated delta.
            if norm == "L2":
                grad_norm = grad.flatten(1).norm(p=2, dim=1, keepdim=True)
                grad_normalized = grad / (
                    grad_norm.view(B, *([1] * (grad.ndim - 1))) + 1e-12
                )
                
                delta = delta + step_size * grad_normalized

                adv_x = project_l2(x + delta, x, t) 

            elif norm == "Linf":
                delta = delta + step_size * grad.sign()
                adv_x = project_linf(x + delta, x, t)
                # adv_x = adv_x + step_size * grad.sign()
                # adv_x = project_linf(adv_x, x, t)
                # delta_new = adv_x - x

            if clipping_f!=None:
                adv_x =clipping_f(adv_x)
            delta = adv_x-x
            delta = delta.detach().requires_grad_(True)
            
        
        with torch.no_grad():
            if clipping_f!= None:
                adv_x = clipping_f(x + delta)
            delta = adv_x - x
            value = dy_f(model(adv_x), y)


   

        mask = value > best_value
        best_value[mask] = value[mask]
        best_delta[mask] = delta.detach()[mask]

    return best_value, best_delta


def pgd_moc(
    model, #f_\theta
    X,
    Y, #either f_\theta(X) or original labels for X
    dy_f, #d_Y as loss function (assuming it satisfies metric properties)
    clipping_f=None,
    norm="L2", #L2, L1, Linf
    t_values=None, #t_1,...,t_K
    step_size=None, 
    numiter=1,
    numrestarts=1,
    nbins=100,
    batch_size=1
):
    """
    Approximate

        omega(t) = max_x m_x(t)

    over the sampled points X. m_x(t) is computed via PGD.

    X: torch.Tensor of shape (n_points, n_features) shall be the entire reference set.
    """
    #put lower and upper bounds on the grid, and number of bins

    if norm=="L1":
        p=1
    elif norm=="L2":
        p=2
    elif norm=="Linf":
        p=np.inf
    else:
        raise Exception(f"Sorry, no implementation for the specified norm: {norm}")

    
    if t_values is None:
        #by default we use a logarithmically spaced grid
        delta = X.max(dim=0).values - X.min(dim=0).values
        TX_ = torch.linalg.vector_norm(delta, ord=p).item()
        qX_ = np.nextafter(0, 1)
    
        log_qX = np.log(qX_)
        log_TX = np.log(TX_)
        log_step = (log_TX - log_qX) / (nbins - 1)

        t_values = np.exp(log_qX + np.arange(nbins) * log_step)
        t_values[0] = qX_
        t_values[-1] = TX_
        
    
    p_moc = []

    for k, t in enumerate(t_values):

        max_mx = 0.0

        for i in range(0,X.shape[0], batch_size):

            x = X[i:i+batch_size]
            y = Y[i:i+batch_size]

            mx, delta = pgd_m_x(
                model,
                x,
                y,
                float(t),
                dy_f,
                norm,
                clipping_f,
                numiter,
                step_size,
                numrestarts
            )

            max_mx = max(max_mx, mx.max().item())

        if k>0:
            max_mx = max(max_mx, p_moc[k-1])

        p_moc.append(max_mx)

    return np.array(p_moc), np.array(t_values)