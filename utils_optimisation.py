import scipy.stats as dists
import numpy as np
from rsome import msk_solver as msk
from rsome import dro, norm, E
import torch
import cvxpy as cv

def out_of_sample_per_and_dis_CVAR(conditional,model_val,xhat,betahat,alpha,lamb,MCsize,
                              mu_f,
                              sig_f,
                              mux = None,
                              random_state=0,
                              generator=None,
                              ):

    """
    Helper function for generated data that monte carlo estimates the objective function
    conditional - context variables of problem (numpy)
    model_val - objective function value of solved problem
    xhat - optimal solutions of solved problem
    betahat - optimal solutions of solved problem
    alpha - quantile CVAR loss
    lamb - weighing parameter (higher - preference for returns over risk)
    MCsize - number of samples used for monte carlo estimation
    mu_f - data generating process function for conditional mean
    sig_f - data generating process function for conditional covariance
    mux - can use a fixed sample instead of generating it in function
    generator - data generating object (deprecated)
    """
    # Potentially needs changing in case of non-normal simulation
    if not mux is None:
        MCp = mux
    else:
        MCp = dists.multivariate_normal.rvs(mean=mu_f(conditional),cov = sig_f(conditional),size=MCsize,random_state=random_state)

    returns = np.array([xhat@p for p in MCp])

    cost = -returns
    q = np.quantile(cost,alpha)
    cvar = (cost)[(cost)>q].mean()
    cost_return = cost.mean()

    performance = cvar  + lamb * cost_return

    disappointment = performance - model_val


    return performance, disappointment

def out_of_sample_per_and_dis_CVAR_simple(conditional,model_val,xhat,alpha,lamb,mux):
    """
    Lightweight helper function for generated data that monte carlo estimates the objective function that uses predefined sample
    conditional - context variables of problem (numpy)
    model_val - objective function value of solved problem
    xhat - optimal solutions of solved problem
    alpha - quantile CVAR loss
    lamb - weighing parameter (higher - preference for returns over risk)
    mux - fixed sample
    """
    MCp = mux

    returns = np.array([xhat@p for p in MCp])

    cost = -returns
    q = np.quantile(cost,alpha)
    cvar = (cost)[(cost)>q].mean()
    cost_return = cost.mean()

    performance = cvar  + lamb * cost_return

    disappointment = performance - model_val


    return performance, disappointment


def solve_portfolio_wasserstein(samples,
                                alpha,
                                lamb,
                                theta = 0,
                                solver = msk,
                                nor = 2
                               ):

    """
    samples - np.array where rows are the sampled empirical distributions
    alpha - quantile CVAR loss
    lamb - weighing parameter (higher - preference for returns over risk)
    theta - radius of Wasserstein ball defining the ambiguity set
    solver - rsome object denoting the solver (default msk in our case)
    nor - sets the wasserstein metric (could be 1 for EMD, np.inf)
    """

    # Number or assets under consideration
    J = samples.shape[-1]
    # Number of samples
    S = samples.shape[-2]


    # Model setup
    model = dro.Model(S)

    p = model.rvar(J)
    v = model.rvar() #auxiliary random variable

    x = model.dvar(J)
    beta = model.dvar(1)
    m = model.dvar(1) #auxiliary variable

    fset = model.ambiguity()

    for s in range(S):
        fset[s].suppset(norm(p - samples[s],nor) <= v)

    fset.exptset(E(v)<= theta)
    pr = model.p
    fset.probset(pr == 1/S)


    model.minsup(E(beta + (1/(1-alpha))*m - lamb*p@x), fset)
    model.st(m >= 0)
    model.st(m >= -p@x - beta)
    model.st(x.sum() == 1, x >= 0)

    m.adapt(p)
    m.adapt(v)
    for s in range(S):
        m.adapt(s)

    model.solve(solver,display=False)
    return model, x, beta, m


def solve_uncertain_moments(alpha,lamb,mu0,sigma0,gamma1,gamma2,solver="MOSEK",verbose=False,kwargs={}):
    """
    alpha - quantile CVAR loss
    lamb - weighing parameter (higher - preference for returns over risk)
    mu0 - estimated conditional mean vector
    sigma0 - uncertainty quantification (estimated conditional covariance matrix)
    gamma1, gamma2 - parameters to define the ambiguity set around the conditional mean (ellipsoid) and conditional covariance (SDP cone)
    solver - string indicating solver to be used (we use mosek, should be capable of semi definite programming)
    """

    m = mu0.shape[0]

    mu0 = cv.Parameter((m,1),value = mu0.reshape(-1,1))

    sigma0 = cv.Parameter((m,m),value=sigma0)

    # variables
    x = cv.Variable((m,1))
    beta = cv.Variable()
    r = cv.Variable()
    q = cv.Variable((m,1))
    Q = cv.Variable((m,m),PSD = True)
    P = cv.Variable((m,m))
    p = cv.Variable((m,1))
    s = cv.Variable()
    PsM = cv.Variable((m+1,m+1))
    SemiI1 = cv.Variable((m+1,m+1)) #when a = 1
    SemiI2 = cv.Variable((m+1,m+1)) #when a = 0

    Qmu0 = cv.Variable((m,1))

    # objective function
    obj = cv.Minimize(
    gamma2*cv.sum(cv.multiply(sigma0,Q))
    - mu0.T@Qmu0
    + r
    + cv.sum(cv.multiply(sigma0,P))
    - 2*mu0.T@p
    + gamma1*s
      )

    # constraints
    cons = []

    # Inner problem constraints
    cons += [q + 2*Q@mu0 + 2*p ==0]
    cons += [SemiI1[:m,:m] == Q,
             SemiI1[[-1],:-1] == (q.T+x.T*(lamb+1/(1-alpha)))/2,
             SemiI1[:-1,[-1]] == (q+x*(lamb+1/(1-alpha)))/2,
             SemiI1[-1,-1] == r-beta*(1-1/(1-alpha))
            ] # A la 17c (S is R)
    cons += [SemiI2[:m,:m] == Q,
             SemiI2[[-1],:-1] == (q.T+x.T*lamb)/2,
             SemiI2[:-1,[-1]] == (q+x*lamb)/2,
             SemiI2[-1,-1] == r-beta
            ] # A la 17c (S is R)

    cons += [SemiI1 >> 0] #PSD constraint
    cons += [SemiI2 >> 0] #PSD constraint

    cons += [PsM[:m,:m] == P,
             PsM[[-1],:-1] == p.T,
             PsM[:-1,[-1]] == p,
             PsM[-1,-1] == s
            ] # constraint 3e constructed
    cons += [PsM >> 0]
    cons += [Q >> 0]
    # Symmetric matrices constraint
    cons += [Q == Q.T]
    cons += [P == P.T]
    #DPP constraint
    cons += [Qmu0 == Q@mu0]
    # Outer problem constraints
    cons += [cv.sum(x) == 1]
    cons += [x >= 0]

    prob = cv.Problem(obj,cons)
    prob.solve(solver,verbose = verbose,**kwargs)

    return prob, x, beta, mu0, sigma0


# def solve_deterministic_mean_var(eta,ro,mu0,sigma0,solver="MOSEK",verbose=False,kwargs={}):
#
#     m = mu0.shape[0]
#
#     mu0 = mu0.reshape(-1,1)
#
#     x = cv.Variable((m,1))
#
#     obj = cv.Maximize(eta*mu0.T@x - (ro/2)*cv.quad_form(x,sigma0))
#
#     cons = []
#     cons += [x>=0,
#              cv.sum(x)==1]
#
#     prob = cv.Problem(obj,cons)
#     prob.solve(solver,verbose = verbose,**kwargs)
#
#     return prob, x, mu0, sigma0

#
def solve_deterministic_CVAR(alpha,lamb,mus,solver="MOSEK",verbose=False,kwargs={}):
    """
    function name is a misnomer, this function solves the problem using SAA, instead of DRO
    alpha - quantile CVAR loss
    lamb - weighing parameter (higher - preference for returns over risk)
    mus - array of predicted samples (numpy)
    """
    n_sam, m = mus.shape

    x = cv.Variable((m,1))
    m = cv.Variable((n_sam,1))
    beta = cv.Variable()

    obj = cv.Minimize(beta + (1/(1-alpha))*(1/n_sam)*cv.sum(m)-lamb*(1/n_sam)*cv.sum(mus@x))
    cons = []
    cons += [m >= 0]
    cons += [m >= -mus@x - beta]
    cons += [x>=0,
             cv.sum(x)==1]

    prob = cv.Problem(obj,cons)
    prob.solve(solver,verbose = verbose,**kwargs)

    return prob, x, beta


def get_gamma1(mus,sigs,true_ho,quantile):
    """
    mus - array of predicted samples (tensor)
    sigs - estimate of covariance (tensor)
    true_ho - array of ground truth values in validation set (tensor)
    quantile - [0,1] what quantile of estimated robustness parameters on holdout should be picked
    """
    deltas = mus.unsqueeze(1) - true_ho.unsqueeze(1)
    distances = ((deltas)@torch.linalg.inv(sigs)@(deltas.transpose(-2,-1))).squeeze()
    gamma1 = distances.quantile(quantile)

    return gamma1


def get_gamma2(mus,sigs,true_ho,quantile):
    """
    mus - array of predicted samples (tensor)
    sigs - estimate of covariance (tensor)
    true_ho - array of ground truth values in validation set (tensor)
    quantile - [0,1] what quantile of estimated robustness parameters on holdout should be picked
    """
    d = (true_ho - mus).unsqueeze(-1)
    est_covs = d@d.transpose(-2,-1)

    est_covs = est_covs.mean(axis=0)

    gam2 = cv.Variable(true_ho.shape[0])
    obj = cv.Minimize(cv.sum(gam2))
    cons = [gam2 >= 0]

    for i in range(true_ho.shape[0]):

        cons += [gam2[i]*sigs[i] - est_covs>> 0]

    prob = cv.Problem(obj,cons)

    prob.solve()

    return np.quantile(gam2.value,quantile), gam2.value


def get_theta(data,samples,norm,quantile):
    """
    bit of a misnomer, in the paper we refer to this parameter as phi
    data - holdout set (index,parameters) matrix
    samples - sampled estimates (index,sample index, parameters) tensor
    """

    theta = torch.norm(samples - data.unsqueeze(-2),norm,dim=-1).mean(axis=1).quantile(quantile)
    return float(theta)
