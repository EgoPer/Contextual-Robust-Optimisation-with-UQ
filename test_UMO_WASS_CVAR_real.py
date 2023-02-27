import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as dists
from tqdm import tqdm
from data import generator, portfolio_dataset
import scipy.stats as dists
import numpy as np
import pandas as pd
from predictive_models import MultitaskGPModel, train_GP
import gpytorch
import cvxpy as cv
import torch
from torch.optim import Adam
from predictive_models import MCDnet
from predictive_models import simpleMSE, train_net
from predictive_models import gaussian_multivariate_NLL, DEmember, ensembler
from predictive_models import predict_normalised
from utils_optimisation import *
# Real data
import glob

# Procedure to process dataset into a useful form (multi-output prediction)
csvs = []
for file in glob.glob("CNNpred/Processed*.csv"):
    csvs.append(file)

specific_tags = ["mom","ROC","EMA"]

dfX = pd.DataFrame()
dfy = []
dfy_cols = []

#Has to mirror the cleaning below
common_cols = set.intersection(*[set(pd.read_csv(csv).set_index("Date").iloc[200:].dropna(axis=1).drop(columns=["Close","Name"]).columns)
                                         for csv in csvs])

for csv in csvs:

    ticker = csv.split("_")[-1].split(".")[0]
    data = pd.read_csv(csv)
    data = data.set_index("Date")
    data = data.iloc[200:]
    data = data.dropna(axis=1)
    data = data.drop(columns=["Close","Name"])

    if dfX.empty:


        stock_specific_cols = set([col for col in common_cols if any([i in col for i in specific_tags])])
        shared_cols = common_cols - stock_specific_cols

        dfX = data.loc[:,shared_cols]


    for col in stock_specific_cols:
        if col != "mom":
            dfX.loc[:,col+f"_{ticker}"] = data.loc[:,col]
        else:
            dfy.append(data.loc[:,col])
            dfy_cols.append(f"ret_{ticker}")

dfy = pd.DataFrame(dfy,index=dfy_cols).T

# Splitting function for real dataset to numerically set size of sets
def split(data,n_test,n_val=0,standardise = False):


    if n_val:
        train = data.iloc[:-n_test,:]
        test = data.iloc[-n_test:,:]
        val = train.iloc[-n_val:,:]
        train = train.iloc[:-n_val,:]

        return train, val, test
    else:
        train = data.iloc[:-n_test,:]
        test = data.iloc[-n_test:,:]

        return train, test
# Transform datasets into a torch object
def get_torch_sets(dfX,dfy,n_test,n_val = 0):


    x_split = split(dfX,n_test,n_val)

    y_split = split(dfy,n_test,n_val)

    torch_ds = list(zip(x_split,y_split))

    torch_ds = [portfolio_dataset(x.to_numpy().astype(np.float32),y.to_numpy().astype(np.float32))
                for x,y in torch_ds]

    return torch_ds

# Sets size of validation and test set (we use a year's worth of data for validation and two years for testing)
size_test = 252*2
size_val = 252
tr,v, te = get_torch_sets(dfX,dfy,size_test,size_val)
tr.fit_standardise(do_y=True)
v.standardise(*tr.stand_data)
te.standardise(*tr.stand_data)

data_size = np.array([1,2,4,6,8])*252 # Data sizes tested for (should be an iterable)
num_datasets = 5 # Number of randomly generated datasets

model_tries = 5 # Number of models trained with random initialisation ceteris paribus
alpha = 0.9 # Governs the alpha-CVAR (1-epsilon = alpha in the paper, minimisation problem so we care about the right tail) in the objective function
lamb = 1 # the tradeoff parameter between returns and CVAR
num_sam = 25 # Number of samples generated for Wasserstein sets
m = te.y.shape[-1] #output size

# Whole dataset in one file
whole_ds = portfolio_dataset(dfX.to_numpy().astype(np.float32),dfy.to_numpy().astype(np.float32))

#size of training set (sliding)
n_lookback = 252*3
# solver = "CVXOPT"
# options = {"reltol": 1e-4,"abstol":1e-5}
solver = "MOSEK"
options ={}


# Determines what set to determine robustness parameters from in this case validation
dat = v

# #quantiles
quantile = 0.5 # for gamma 1
quantile2 = 0.9 # for gamma 2
quantile3 = 0.9 # for theta - phi in the paper (where it was changed to not cause confusion)
k = 0.005 # scaling parameter for theta/phi

# norm governing the wasserstein metric
p_wass = 2

results_file = f"results_UM_alphalambda_k_{k}_{(alpha,lamb)}_tries_{model_tries}_quantiles_{[quantile,quantile2,quantile3]}.pkl"
results_file = f"results_real_just_DE_sampling_{k}_{(alpha,lamb)}_tries_{model_tries}_quantiles_{[quantile,quantile2,quantile3]}.pkl"

res_cols = ["test_index","approach","experiment_no","return"]
results = []

# Set unconditional gamma parameters
allmu, allsig = dat.original_y.mean(axis=0).unsqueeze(0), torch.cat([dat.original_y.T.cov().unsqueeze(0)]*dat.original_y.shape[0],dim = 0)
gamma1_uncon = get_gamma1(allmu,
                      allsig,
                      dat.original_y,quantile)
gamma2_uncon, gm2un = get_gamma2(allmu,allsig,dat.original_y,quantile2)

# Unconditional Wasserstein (same weights for all)
sampleUncon = tr.original_y
theta_Uncon =k*get_theta(dat.original_y,tr.original_y,p_wass,quantile3)

prob_WASS_Uncon, x_WASS_Uncon, beta, m_wass = solve_portfolio_wasserstein(samples = sampleUncon.numpy().squeeze(),
                        alpha = alpha,
                        lamb = lamb,
                        theta = theta_Uncon,
                        solver = msk,
                        nor = p_wass
                           )

for en in tqdm(range(te.y.shape[0])):

    # Rolling window average
    spec_ds = whole_ds.original_y[-(size_test+n_lookback)+en:-(size_test)+en]
    muSP = spec_ds.mean(axis =0).numpy()
    SigmaSP = np.cov(spec_ds.numpy(),rowvar=False)

    try:
        prob, x_W ,beta, mu0, sigma0 = solve_uncertain_moments(alpha,lamb,muSP,SigmaSP,
                                                               gamma1 = gamma1_uncon,
                                                               gamma2 = gamma2_uncon,
                                                                verbose = False,
                                                                solver = solver,
                                                                kwargs=options)
        results.append([en,"Rolling_window_UM",0,te.original_y[en,:]@x_W.value])
    except:
        results.append([en,"Rolling_window_UM",0,np.nan])

    # Deterministic (average window)
    try:
        prob, x_DET_rolling, beta = solve_deterministic_CVAR(alpha,lamb,spec_ds)
        results.append([en,"Rolling_window_SAA",0,te.original_y[en,:]@x_DET_rolling.value])
    except:
        results.append([en,"Rolling_window_SAA",0,np.nan])
    # Random
    results.append([en,"Equal_weights",0,te.original_y[en,:]@np.array([1/m]*m).reshape(-1,1)])
    # Uncon Wass
    results.append([en,"Uncon_WASS",0,te.original_y[en,:]@x_WASS_Uncon.get()])


for exp in tqdm(range(model_tries)):
    # Train Deep Ensemble
    members = []
    n_members = 10
    for _ in range(n_members):
        ex_member = DEmember(tr.X.shape[-1],tr.y.shape[-1],[100,100,100],dropout = 0.0,act_function=torch.nn.ReLU)
        opt = Adam(ex_member.parameters(),lr= 0.0001)
        train_net(ex_member,tr,opt,gaussian_multivariate_NLL,200,128,False)
        members.append(ex_member)

    ens = ensembler(members)

    # Train GP
    num_tasks = tr.y.shape[1]
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks,rank = num_tasks)
    model = MultitaskGPModel(tr.X, tr.y, likelihood)


    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iterations = 100

    train_GP(model,tr,mll,optimizer,training_iterations,False)

    model.eval()
    likelihood.eval()

    # Get gamma values
    allmu_DE, allsig_DE = [i.detach() for i in ens(dat.X,generator= tr)]
    allmu, allsig = dat.original_y.mean(axis=0).unsqueeze(0), torch.cat([dat.original_y.T.cov().unsqueeze(0)]*dat.original_y.shape[0],dim = 0)

    model.eval()
    likelihood.eval()
    allmu_GP, allsig_GP = [], []
    for i in range(dat.X.shape[0]):
        cond_model = likelihood(model(dat.X[[i]]))
        mu, sig = predict_normalised((cond_model.mean.detach(),cond_model.covariance_matrix.detach()),generator=tr)
        allmu_GP.append(mu)
        allsig_GP.append(sig)

    allmu_GP = torch.cat(allmu_GP)
    allsig_GP = torch.stack(allsig_GP)

    gamma1_DE = get_gamma1(allmu_DE,allsig_DE,dat.original_y,quantile)
    gamma1_GP = get_gamma1(allmu_GP,allsig_GP,dat.original_y,quantile)

    gamma2_DE, gm2de = get_gamma2(allmu_DE,allsig_DE,dat.original_y,quantile2)
    gamma2_uncon, gm2un = get_gamma2(allmu,allsig,dat.original_y,quantile2)
    gamma2_GP, gm2gp = get_gamma2(allmu_GP,allsig_GP,dat.original_y,quantile2)

    # Get theta
    cond_modelA = likelihood(model(dat.X))
    allsam_GP = cond_modelA.rsample(torch.Size([num_sam])).transpose(0,1) * tr.std_y + tr.means_y
    theta_GP = k*get_theta(dat.original_y,allsam_GP,p_wass,quantile3)

    # Train monte carlo dropout network and get theta
    mcd = MCDnet(tr.X.shape[-1],tr.y.shape[-1],[100,100,100],dropout = 0.5,act_function=torch.nn.ReLU)
    opt = Adam(mcd.parameters(),lr= 0.0001)
    train_net(mcd,tr,opt,simpleMSE,100,128,False)

    allsam_MCD = mcd.MCD_sampling(dat.X,num_sam,generator=tr)
    theta_MCD = k*get_theta(dat.original_y,allsam_MCD,p_wass,quantile3)

    # Get theta_DE
    allsam_DE = []
    for i in range(dat.y.shape[0]):
        m, s = [i.detach().numpy() for i in ens(dat.X[i],generator= tr)]
        allsam_DE.append(dists.multivariate_normal.rvs(m,s,size=num_sam))
    allsam_DE = torch.tensor(np.stack(allsam_DE))
    theta_DE = k*get_theta(dat.original_y,allsam_DE,p_wass,quantile3)


    for en in tqdm(range(te.y.shape[0]),leave = True):
        conditional = te.original_X[en,:].numpy()
        inp = (torch.Tensor(conditional)-tr.means_X)/tr.std_X
        with torch.no_grad():
            muEDE, SigmaEDE = [i.detach().numpy() for i in ens(inp,generator= tr)]

        # Deep Ensemble (try except formed bc mosek occasionally fails, assume no decisions in that case)
        try:
            prob, x_DE, beta, mu0, sigma0 = solve_uncertain_moments(alpha,lamb,muEDE,SigmaEDE,
                                                                 gamma1 = gamma1_DE,
                                                                 gamma2 = gamma2_DE,
                                                                 verbose = False,
                                                                 solver = solver,
                                                                 kwargs=options)
            results.append([en,"DE",exp,te.original_y[en,:]@x_DE.value])

        except:
            results.append([en,"DE",exp,np.nan])

        # # DE-SAA (predicted ~ non robust equivalent for this type of problem)
        mus = dists.multivariate_normal.rvs(mean=muEDE,
                                            cov=SigmaEDE,
                                            size=num_sam,
                                            random_state=0)
        try:
            prob, x_DET_pred, beta = solve_deterministic_CVAR(alpha,lamb,mus)
            results.append([en,"Conditional_DE_SAA",exp,te.original_y[en,:]@x_DET_pred.value])

        except:
            results.append([en,"Conditional_DE_SAA",exp,0])


        try:
            prob, x_WASS_DE, beta, m_wass = solve_portfolio_wasserstein(samples = mus,
                                    alpha = alpha,
                                    lamb = lamb,
                                    theta = theta_DE,
                                    solver = msk,
                                    nor = p_wass,
                                       )
            results.append([en,"DE_WASS",exp,te.original_y[en,:]@x_WASS_DE.get()])

        except:
            results.append([en,"DE_WASS",exp,0])

        # GP

        cond_model = likelihood(model(te.X[[en]]))
        muGP, sigGP = predict_normalised((cond_model.mean.detach(),cond_model.covariance_matrix.detach()),generator=tr)
        try:
            prob, x_GP, beta, mu0, sigma0 = solve_uncertain_moments(alpha,lamb,
                                                                    muGP.numpy().squeeze(),
                                                                    sigGP.numpy(),
                                                                 gamma1 = gamma1_GP,
                                                                 gamma2 = gamma2_GP,
                                                                 verbose = False,
                                                                 solver = solver,
                                                                 kwargs=options)
            results.append([en,"GP",exp,te.original_y[en,:]@x_GP.value])

        except:
            results.append([en,"GP",exp,np.nan])



        sampleGP = cond_model.rsample(torch.Size([num_sam])) * tr.std_y + tr.means_y
        prob, x_WASS_GP, beta, m_wass = solve_portfolio_wasserstein(samples = sampleGP.detach().numpy().squeeze(),
                                alpha = alpha,
                                lamb = lamb,
                                theta = theta_GP,
                                solver = msk,
                                nor = p_wass
                                   )
        try:
            results.append([en,"GP_WASS",exp,te.original_y[en,:]@x_WASS_GP.get()])
        except:
            results.append([en,"GP_WASS",exp,np.nan])

        # SAA (predicted ~ non robust equivalent for this type of problem)

        try:
            prob, x_DET_GP, beta = solve_deterministic_CVAR(alpha,lamb,sampleGP.detach().numpy().squeeze())
            results.append([en,"Conditional_GP_SAA",exp,te.original_y[en,:]@x_DET_GP.value])
        except:
            results.append([en,"Conditional_GP_SAA",exp,np.nan])


        #MCD net
        sampleMCD = mcd.MCD_sampling(inp.unsqueeze(0),num_sam,generator=tr).numpy()

        prob, x_MCD, beta, m_wass = solve_portfolio_wasserstein(samples = sampleMCD.squeeze(),
                                alpha = alpha,
                                lamb = lamb,
                                theta = theta_MCD,
                                solver = msk,
                                nor = p_wass
                                   )
        try:
            results.append([en,"MCD_WASS",exp,te.original_y[en,:]@x_MCD.get()])
        except:
            results.append([en,"MCD_WASS",exp,np.nan])


        # SAA (predicted ~ det equivalent for this type of problem)

        try:
            prob, x_DET_MCD, beta = solve_deterministic_CVAR(alpha,lamb,sampleMCD.squeeze())
            results.append([en,"Conditional_MCD_SAA",exp,te.original_y[en,:]@x_DET_MCD.value])
        except:
            results.append([en,"Conditional_MCD_SAA",exp,np.nan])

# saves the dataframe in a pickle file
results = pd.DataFrame(results,columns = res_cols)
results.to_pickle(results_file)
