import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
import copy
import gpytorch



# Single NN, for MCD (multiple output, only num outputs)
class MCDnet(nn.Module):
    def __init__(self,nin,nout,architecture =[],act_function=None,dropout = 0):
        """
        Simple fully connected feedforward NN to be used for monte carlo dropout, designed for multiple output tabular data
        nin - number of inputs
        nout - number of outputs
        architecture - list of integers (size of hidden layers)
        act_function - torch.nn class activation funciton
        dropout - bernouli prob of dropout for hidden layers
        """
        super(MCDnet, self).__init__()

        build = self.build_architecture(nin,nout,architecture,act_function,dropout)
        self.layers = nn.Sequential(build)


    def build_architecture(self,nin,nout,architecture =[],act_function=None,dropout = 0):
        build = OrderedDict()

        if len(architecture)==0:
            build["shallow"] = nn.Linear(nin,nout)
            if act_function:
                build["activation"] = act_function()
        else:
            build["input"] = nn.Linear(nin,architecture[0])
            if dropout:
                build[f"do 0"] = nn.Dropout(dropout)
            if act_function:
                build["act 0"] = act_function()

            for i in range(len(architecture)-1):
                build[f"dense {i+1}"] = nn.Linear(architecture[i],architecture[i+1])
                if dropout:
                    build[f"do {i+1}"] = nn.Dropout(dropout)
                if act_function:
                    build[f"act {i+1}"] = act_function()

            build["output"] = nn.Linear(architecture[-1],nout)

        return build

    def forward(self,x):
        y = self.layers(x)
        return y


    def MCD_sampling(self,x,num_samples,generator = False):
        """
        will return samples for an input in the form (instance,UQ samples,outputs)

        if generator is specified the samples will be scaled
        """
        self.train()
        with torch.no_grad():
            sampled = []
            for _ in range(num_samples):
                if not generator:
                    sampled.append(self.forward(x).unsqueeze(-1).transpose(-1,-2))
                else:
                    sampled.append(self.forward(x).unsqueeze(-1).transpose(-1,-2)*generator.std_y + generator.means_y)


        out = torch.cat(sampled,dim=1)
        return out

#         #Example use
#         from torch.optim import Adam
#         ex_member = MCDnet(3,6,[10,10,10],dropout = 0.5,act_function=torch.nn.ReLU)
#         opt = Adam(ex_member.parameters(),lr= 0.0001)

#         train_net(ex_member,tr,opt,simpleMSE,500,128,True)
#         ex_member.MCD_sampling(tr.X[[0]],20,generator=g).squeeze()

def simpleMSE(output, target):
    "MSE adopted for multiple output setting"
    loss = torch.mean((output - target)**2)
    return loss

# Homogeneous neural ensemble

class DEmember(nn.Module):
    def __init__(self,nin,nout,architecture =[],act_function=None,dropout = 0):
        """
        nin - number of inputs
        nout - number of outputs
        architecture - list of integers (size of hidden layers)
        act_function - torch.nn class activation funciton
        dropout - bernouli prob of dropout for hidden layers
        """
        super(DEmember, self).__init__()

        build = self.build_architecture(nin,nout,architecture,act_function,dropout)
        self.layers = nn.Sequential(build)

        #Final joint layer outputs
        fjlo = self.layers[-1].out_features

        #Multiple output layers
        self.mu_layer = nn.Linear(fjlo,nout)

        self.sigma_layer = nn.Linear(fjlo,nout)

        self.ro_layer = nn.Linear(fjlo,int(nout*(nout-1)/2))

    def build_architecture(self,nin,nout,architecture =[],act_function=None,dropout = 0):
        build = OrderedDict()

        if len(architecture)==0:
            build["shallow"] = nn.Linear(nin,nout)
            if act_function:
                build["activation"] = act_function()
        else:
            build["input"] = nn.Linear(nin,architecture[0])
            if dropout:
                build[f"do 0"] = nn.Dropout(dropout)
            if act_function:
                build["act 0"] = act_function()

            for i in range(len(architecture)-2):
                build[f"dense {i+1}"] = nn.Linear(architecture[i],architecture[i+1])
                if dropout:
                    build[f"do {i+1}"] = nn.Dropout(dropout)
                if act_function:
                    build[f"act {i+1}"] = act_function()

            build["final hidden"] = nn.Linear(architecture[-2],architecture[-1])

        return build

    def forward(self,x,alpha = 0.075, e = 1e-3):
        """
        Needs to be tied to a loss function that understands the structure.
        output - (nout means,nout variances,(nout-1)*nout/2 correlation coefficients)
        The correlation estimates are concatenated from the lower triangular top to bottom.
        The covariance estimates are constructed from the variance and correlation estimates.
        This is to encourage the formation of a PSD matrix

        The Russel paper suggests two stabilising measures during training
        alpha - constant multiplied with the input to the tanh layer to slow saturation (typically 0.05)
        e - (1 -constant) multiplied with rho, so the correlation coefficients don't get to 1 or -1
        """
        x = self.layers(x)

        # mean estimation
        mu = self.mu_layer(x)
        # variance estimation (need to figure out numerical stability for large values, maybe standardisation)
        s = torch.exp(self.sigma_layer(x))
#         s = torch.pow(self.sigma_layer(x),2)

        # correlation estimation
        rho = (1-e) * torch.tanh(alpha * self.ro_layer(x))

        # construct Covariance matrix
        Sigma = torch.diag_embed(s)
        tind = torch.tril_indices(s.shape[-1],s.shape[-1],offset = -1)

        Sigma[...,tind[0],tind[1]] = rho * torch.sqrt(s[...,tind[0]] *s[...,tind[1]])
        Sigma[...,tind[1],tind[0]] = rho * torch.sqrt(s[...,tind[0]] *s[...,tind[1]])

        return mu, Sigma

#         #Example use
#         from torch.optim import Adam
#         ex_member = DEmember(3,6,[100,100,100],dropout = 0.0,act_function=torch.nn.ReLU)
#         opt = Adam(ex_member.parameters(),lr= 0.0001)
#         train_net(ex_member,tr,opt,gaussian_multivariate_NLL,500,128,True)

def gaussian_multivariate_NLL(output,target,e = 1e-5):
    """
    output - (mean vector (nout), covariance matrix (nout x nout))
    """

    v = (target - output[0]).unsqueeze(-1)

    NLL1 = 0.5 * torch.matmul(torch.matmul(v.transpose(-2,-1),output[1].inverse()),v).squeeze(1)
    # clamp idea from Russel paper (prevents nans)
#     NLL2 = 0.5 * torch.log(torch.linalg.det(output[1]).clamp(min=e)).unsqueeze(-1)
    # log det A = sum log eigenvalues idea to increase stability, since inf determinant values were inducing nans
    # potentially clamp eigvals if necessary (note that clamping should be done at the level of the log, ie it is some small value if an odd number of eigenvalues are negative)
#     NLL2 = 0.5 * torch.sum(torch.log(torch.real(torch.linalg.eigvals(output[1]))),axis=1)
    NLL2 = 0.5 * torch.where(torch.remainder(torch.sum(torch.real(torch.linalg.eigvals(output[1])) <= 0),2) != 0,
                             torch.log(torch.tensor(e)),
                            torch.sum(torch.log(torch.abs(torch.real(torch.linalg.eigvals(output[1])))),axis=1)
                            )

    regulariser = 0
#     regulariser = 0.05 * torch.sum(torch.pow(-torch.real(torch.linalg.eigvals(output[1])).clamp(max = 0),2),axis = 1)
    return torch.mean(NLL1 + NLL2 + regulariser)

def evaluate(model,data,loss_f,bsize = 128):

    loader = DataLoader(data,batch_size=bsize,shuffle = False)
    model.eval()
    loss = 0
    with torch.no_grad():
        for X,y in loader:
            pred = model(X)
            loss += X.shape[0] * loss_f(pred,y)

    loss /= len(data)

    return loss

def train_net(member,train_dataset,optimiser,loss_f,epochs,bsize,verbose = False):

    loader = DataLoader(train_dataset,batch_size=bsize,shuffle = True)
    member.train()

    for i in range(epochs):

        for X,y in loader:
            pred = member(X)
            loss = loss_f(pred,y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        if verbose:
            if i in range(0,epochs,int(epochs/10)):
                print(f"Loss at epoch {i+1}={evaluate(member,train_dataset,loss_f).round(decimals = 3)}")



# object for ensemble predictions
class ensembler(nn.Module):

    def __init__(self,models,architecture =[]):
        """
        models - container of torch Modules which have been trained
        architecture - potential stacking architecture (skip for now)

        example:
        example_models = []
        for i in range(num_models):
            member = DEmember(3,6,[100,100,100],dropout = 0.0,act_function=torch.nn.ReLU)
            opt = Adam(member.parameters(),lr= 0.0001)
            train_net(emember,tr,opt,gaussian_multivariate_NLL,10,128,True)
            example_models.append(member)

        ens = ensembler(example_models)
        """
        super(ensembler, self).__init__()

        self.models = models

    def forward(self,x,full = False, generator = None):
        """
        Specify generator if you want to destandardise variables
        """
        if generator:
            mus = [predict_normalised(self.models[i](x),generator,return_tensor=True)[0] for i in range(len(self.models))]
            Sigmas = [predict_normalised(self.models[i](x),generator,return_tensor=True)[1] for i in range(len(self.models))]
        else:
            mus = [self.models[i](x)[0] for i in range(len(self.models))]
            Sigmas = [self.models[i](x)[1] for i in range(len(self.models))]

        mus = torch.stack(mus)
        Sigmas = torch.stack(Sigmas)

        mu_ens = torch.mean(mus,axis = 0)

        sigma_ens = (torch.sum(Sigmas, axis = 0) +
                    torch.sum((mus - mu_ens).unsqueeze(-1) * (mus - mu_ens).unsqueeze(-1).transpose(-2,-1),axis=0)) / len(self.models)

        if full:
            return mus, Sigmas
        else:
            return mu_ens, sigma_ens


def predict_normalised(output,generator,return_tensor = False):
    """
    output - iterable(mean_vector,covariance matrix), can be either tensor or numpy array
    generator - generator object as defined in this work (can only be used after the normalise method has been called)

    returns: scaled_mean, scaled_covariance
    """
    output = [torch.Tensor(o) for o in output]

    scaled_mu = output[0] *generator.std_y + generator.means_y
    scaled_sigma = np.matmul(generator.std_y.reshape(-1,1),generator.std_y.reshape(1,-1)) * output[1]

    if return_tensor:
        return torch.Tensor(scaled_mu), torch.Tensor(scaled_sigma)
    else:
        return scaled_mu, scaled_sigma



# Gaussian process model
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        """
        train_x - (n x d) tensor (d - number of inputs)
        train_y - (n x k) tensor (k - number of outputs)
        likelihood - gpytorch.likelihoods object
        """
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        num_tasks = train_y.shape[1]
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5)), num_tasks=num_tasks, rank=num_tasks
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

        # Example use
#         num_tasks = tr.y.shape[1]
#         likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks,rank = num_tasks)
#         model = MultitaskGPModel(tr.X, tr.y, likelihood)

#         model.train()
#         likelihood.train()
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
#         mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
#         training_iterations = 100

#         train_GP(model,tr,mll,optimizer,training_iterations,True)

def train_GP(model,data,loss_f,optimiser,n_iter, verbose = False):
    for i in range(n_iter):

        optimiser.zero_grad()
        output = model(data.X)
        loss = -loss_f(output, data.y)
        loss.backward()
        optimiser.step()

        if verbose:
            if i in range(0,n_iter,int(0.1*n_iter)+1):
                print(f'Iter {i + 1}- Loss: {loss.item()}')


def custom_conf_region(likelihood,num_std, generator=None):
    """
    likelihood - gpytorch likelihood object
    """
    std2 = likelihood.stddev.mul_(num_std).detach()
    mean = likelihood.mean.detach()

    if not generator:
        return mean.sub(std2), mean.add(std2)
    else:
        return mean.sub(std2)*generator.std_y + generator.means_y , mean.add(std2)*generator.std_y + generator.means_y
