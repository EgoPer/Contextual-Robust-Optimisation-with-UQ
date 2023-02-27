import numpy as np
import scipy.stats as dists
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torch import tensor
import torch
from multivariate_laplace import multivariate_laplace #Thank https://github.com/david-salac/multivariate-Laplace-extension-for-SciPy

class portfolio_dataset(Dataset):

    def __init__(self,X,y):
        self.original_X = tensor(X)
        self.original_y = tensor(y)
        self.X = self.original_X
        self.y = self.original_y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):
        inp = self.X[idx]
        out = self.y[idx]
        return inp, out

    def fit_standardise(self,do_y = False):

        mean_x = self.original_X.mean(axis=0)
        std_x = self.original_X.std(axis=0)
        self.X = (self.original_X - mean_x) / std_x
        self.means_X = mean_x
        self.std_X = std_x
        self.means_y = torch.zeros(self.original_y.shape[-1])
        self.std_y = torch.ones(self.original_y.shape[-1])

        if do_y:
            mean_y = self.original_y.mean(axis=0)
            std_y = self.original_y.std(axis=0)

            self.y = (self.original_y - mean_y) / std_y
            self.means_y = mean_y
            self.std_y = std_y

        self.stand_data = [self.means_X,self.std_X,self.means_y,self.std_y ]

    def standardise(self,mean_x,std_x,mean_y,std_y):

        self.X = (self.original_X - mean_x) / std_x
        self.y = (self.original_y - mean_y) / std_y

        self.means_X = mean_x
        self.std_X = std_x
        self.means_y = mean_y
        self.std_y = std_y
        self.stand_data = [mean_x,std_x,mean_y,std_y]


class generator:
    def __init__(self,n_inputs,n_outputs):
        self.nx = n_inputs
        self.ny = n_outputs

    def __repr__(self):
        return f"A generator for a dataset with {self.nx} inputs and {self.ny} outputs."

    def assign_input_distributions(self,distribution_vector):
        """
        distribution vector - list of tuples (scipy distribution methods, parameters in dct form)
        """
        assert len(distribution_vector) == self.nx, "the number of distributions does not match the number of inputs"
        assert all([type(m[1]) == dict for m in distribution_vector]), "distribution parameters must be in dictionary form so they can be inserted into the methods using **"

        self.x_dist = distribution_vector

    def generate_inputs(self,data_size,state = None):

        self.n = data_size
        X = []
        for d in self.x_dist:
            X.append(d[0].rvs(size = data_size,random_state = state,**d[1]))

        X = np.array(X).T #transpose so rows are instances
        return X

    def generate_outputs(self, X, joint_distribution, mean_function, covar_function, state = None, df = 5):
        """
        joint_distribution - scipy distribution method (governs residual assumption) as tuple (scipy,params)
        mean_function - data generating process for means, should output number of outputs
        covar_function - if we want to make process heteroskedastic (otherwise just a constant function)
        """

        cmu = np.apply_along_axis(mean_function,1,X)

        csig = np.apply_along_axis(covar_function,1,X)

        states = dists.randint.rvs(low = 1, high = int(1e7), size = X.shape[0], random_state = state)

        if "multivariate_normal" in str(joint_distribution):
            y = np.array([joint_distribution.rvs(mean=cmu[i],cov=csig[i],random_state = states[i]) for i in range(X.shape[0])])
        elif "multivariate_laplace" in str(joint_distribution):
            y = np.array([joint_distribution.rvs(mean=cmu[i],cov=csig[i],random_state = states[i]) for i in range(X.shape[0])])
        elif "multivariate_t" in str(joint_distribution):
            y = np.array([joint_distribution.rvs(loc=cmu[i],shape=csig[i],random_state = states[i],df = df) for i in range(X.shape[0])])
        else:
            y = np.array([joint_distribution.rvs(loc=cmu[i],shape=csig[i],random_state = states[i]) for i in range(X.shape[0])])

        return y.reshape((self.n,self.ny))

    def generate_dataset(self,data_size,joint_distribution, mean_function, covar_function, state = None):

        assert len(self.x_dist) != 0 , "please define distribution for inputs first"

        X = self.generate_inputs(data_size,state = state)

        y = self.generate_outputs(X,joint_distribution,mean_function,covar_function,state = state)

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.original_X = self.X
        self.original_y = self.y
        return self.X, self.y

    def normalise_dataset(self):

        self.means_X = self.original_X.mean(axis=0)
        self.means_y = self.original_y.mean(axis=0)

        self.std_X = self.original_X.std(axis=0)
        self.std_y = self.original_y.std(axis=0)#np.cov(self.original_y,rowvar=False)

        stan_x = (self.original_X - self.means_X) / self.std_X
        stan_y = (self.original_y - self.means_y) / self.std_y

        return stan_x, stan_y

    def get_torch_split_datasets(self,test_size,val_size = None,seed=None,normalise=False):

        if normalise:
            X, y = self.normalise_dataset()
        else:
            X, y = self.X, self.y

        train_indices, test_indices = train_test_split(range(X.shape[0]), test_size=test_size, random_state=seed)

        train = portfolio_dataset(X[train_indices],y[train_indices])
        test = portfolio_dataset(X[test_indices],y[test_indices])

        if val_size:
            train_indices, val_indices = train_test_split(train_indices, test_size=val_size/(1-test_size), random_state=seed)

            validation = portfolio_dataset(X[val_indices],y[val_indices])
            train = portfolio_dataset(X[train_indices],y[train_indices])

            return train, validation, test

        else:
            return train, test


    # Example use
    # g = generator(3,6)
    # dist_v = [(dists.laplace,{"loc":1,"scale":2}),(dists.laplace,{"loc":1,"scale":2}),(dists.laplace,{"loc":1,"scale":2})]
    # dist_v = [(dists.norm,{"loc":1000,"scale":50}),(dists.norm,{"loc":0.02,"scale":0.01}),(dists.lognorm,{"scale":np.exp(0),"s":1})]
    # g.assign_input_distributions(dist_v)

    # def mu_f(x):
    #     mu = np.array([86.8625, 71.6059, 75.3759, 97.6258, 52.7854, 84.89])

    #     v1 = np.array([1,1,1,1,1,1])
    #     v2 = np.array([1,1,1,1,1,1])
    #     v3 = np.array([1,1,1,1,1,1])

    #     y = mu + 0.1*(x[0] - 1000) * v1 + 1000 * x[1] * v2 + 10*np.log(x[2]+1) *v3
    #     return y

    # def sig_f(x):
    #     sig = np.array([[136.687, 0,0,0,0,0],
    #                    [8.79766, 142.279,0,0,0,0],
    #                    [16.1504, 15.0637, 122.61,0,0,0],
    #                    [18.4944, 15.6961, 26.344, 139.14,0,0],
    #                    [3.41394, 16.5922, 14.8795, 13.9914, 151.73,0],
    #                    [24.8156, 18.7292, 17.1574, 6.36536, 24.7703, 144.67]])
    #     sig = sig + sig.T - np.diag(np.diag(sig))
    #     return sig

    # dist_out = dists.multivariate_normal


    # inn, oop = g.generate_dataset(100,dist_out, mu_f, sig_f, state = None)

    # tr,v,te = g.get_torch_split_datasets(0.1,0.1)
