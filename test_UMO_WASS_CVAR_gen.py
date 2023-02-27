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
from multivariate_laplace import multivariate_laplace



# Conditional mean generating function
def mu_f(x):
    mu = np.array([86.8625, 71.6059, 75.3759, 97.6258, 52.7854, 84.89])

    y = mu
    y1 = np.tanh(x[1]*np.exp(x[0]/2-2))*30
    y2 = np.tanh((x[1])*np.sin(x[2])*3)*50
    y3 = np.log(np.abs(x[0]*x[1]*x[2])) * 10
    y4 = np.sin(x[1]) + x[0]**2 - (x[0]*x[2])
    y5 = (np.sin(x[0]) + np.sin(x[1]/(x[2]*10)))*20
    y6 = y1 - y2

    y = mu + np.array([y1,y2,y3,y4,y5,y6])
    return y

# Conditional covariance generating function
def sig_f(x):
    sig_root = np.array([[136.687, 0,0,0,0,0],
                   [8.79766, 142.279,0,0,0,0],
                   [16.1504, 15.0637, 122.61,0,0,0],
                   [18.4944, 15.6961, 26.344, 139.14,0,0],
                   [3.41394, 16.5922, 14.8795, 13.9914, 151.73,0],
                   [24.8156, 18.7292, 17.1574, 6.36536, 24.7703, 144.67]])



    sig = sig_root + sig_root.T - np.diag(np.diag(sig_root))
    # Heteroskedasticity
    sig *= np.tanh(x[0])/5 + 1
    sig = sig@sig
    return sig


model_tries = 5 # Number of models trained with random initialisation ceteris paribus
data_size = np.array([1,2,4,6,8])*252 # Data sizes tested for (should be an iterable)
num_datasets = 5 # Number of randomly generated datasets
num_sam = 25 # Number of samples generated for Wasserstein sets

# Generates seeds for datasets
np.random.seed(42)
data_generator_seeds = np.random.randint(0,1000,size=num_datasets)

# Seed for monte carlo evaluation
eval_seed = 12

alpha = 0.9 # Governs the alpha-CVAR (1-epsilon = alpha in the paper, minimisation problem so we care about the right tail) in the objective function
lamb = 1 # the tradeoff parameter between returns and CVAR

# #quantiles
quantile = 0.5 # for gamma 1
quantile2 = 0.9 # for gamma 2
quantile3 = 0.9 # for theta - phi in the paper (where it was changed to not cause confusion)
k = 0.1 # scaling parameter for theta/phi

# norm governing the wasserstein metric
p_wass = 2

solver = "MOSEK"
options ={}

results_file = f"results_UM_WASS_gen_ood_nonl_het_k_{k}_alphalambda_{(alpha,lamb)}_quantiles_{[quantile,quantile2,quantile3]}_sizes_{data_size}_numdatasets_{num_datasets}_tries_{model_tries}_numsam_{num_sam}_.pkl"
res_cols = ["test_index","data_size","seed","approach","experiment_no","return","performance","dissapointment"]
results = []

# Generate data
# Distributions of covariates
dist_v = [(dists.norm,{"loc":0,"scale":1}),(dists.norm,{"loc":0,"scale":1}),(dists.norm,{"loc":0,"scale":1})]
# Joint distribution of targets
dist_out = dists.multivariate_normal
#out of sample distributions for test
dist_v_ood = [(dists.norm,{"loc":2,"scale":1}),(dists.norm,{"loc":-2,"scale":1}),(dists.norm,{"loc":2,"scale":1})]

# constant test set
g = generator(3,6)
g.assign_input_distributions(dist_v_ood)
# dist_out_ood = multivariate_laplace
dist_out_ood = dists.multivariate_normal
inn, oop = g.generate_dataset(101,dist_out_ood, mu_f, sig_f, state = 1453)
tr,te = g.get_torch_split_datasets(0.99,0,normalise=False)

for size in tqdm(data_size):
    for seed in tqdm(data_generator_seeds,leave=False):
        # Generate new dataset
        g = generator(3,6)
        g.assign_input_distributions(dist_v)
        inn, oop = g.generate_dataset(size,dist_out, mu_f, sig_f, state = seed)
        # tr,v,te = g.get_torch_split_datasets(0.2,0.1,normalise=False)
        tr,v,_ = g.get_torch_split_datasets(1/size,0.2,normalise=False)

        tr.fit_standardise(do_y=True)
        v.standardise(*tr.stand_data)
        te.standardise(*tr.stand_data)


        # Determines what set to determine robustness parameters from in this case validation
        dat = v

        # Procedure to obtain robustness parameters for unconditional uncertain moments
        allmu, allsig = dat.original_y.mean(axis=0).unsqueeze(0), torch.cat([dat.original_y.T.cov().unsqueeze(0)]*dat.original_y.shape[0],dim = 0)
        gamma1_uncon = get_gamma1(allmu,
                              allsig,
                              dat.original_y,quantile)
        gamma2_uncon, gm2un = get_gamma2(allmu,allsig,dat.original_y,quantile2)

        # Unconditional Wasserstein
        sampleUncon = tr.original_y
        theta_Uncon =k*get_theta(dat.original_y,tr.original_y,p_wass,quantile3)

        prob_WASS_Uncon, x_WASS_Uncon, beta, m = solve_portfolio_wasserstein(samples = sampleUncon.numpy().squeeze(),
                                alpha = alpha,
                                lamb = lamb,
                                theta = theta_Uncon,
                                solver = msk,
                                nor = p_wass
                                   )

        # Solve deterministic equivalent
        prob, x_DET_rolling, beta = solve_deterministic_CVAR(alpha,lamb,tr.original_y)

        # Solve across test set
        for en in tqdm(range(te.y.shape[0]),leave = False):

            conditional = te.original_X[en,:].numpy()
            mux = dist_out_ood.rvs(mean=mu_f(conditional),
                                        cov = sig_f(conditional),
                                        size=10000,
                                        random_state=eval_seed)

            # Training set
            muSP = tr.original_y.mean(axis=0).numpy()
            SigmaSP = np.cov(tr.original_y,rowvar=False)
            try:
                prob, x_W ,beta, mu0, sigma0 = solve_uncertain_moments(alpha,lamb,muSP,SigmaSP,
                                                                       gamma1 = gamma1_uncon,
                                                                       gamma2 = gamma2_uncon,
                                                                        verbose = False,
                                                                        solver = solver,
                                                                        kwargs=options)

                p,d = out_of_sample_per_and_dis_CVAR_simple(conditional,prob.value,x_W.value.T,alpha,lamb,mux)
                results.append([en,size,seed,"Unconditional_UM",0,te.original_y[en,:]@x_W.value,p,d])
            except:
                results.append([en,size,seed,"Unconditional_UM",0,0,0,np.nan])

            # Perfect knowledge
            try:
                prob, x_P, beta = solve_deterministic_CVAR(alpha,lamb,mux)
                p,d = out_of_sample_per_and_dis_CVAR_simple(conditional,prob.value,x_P.value.T,alpha,lamb,mux)
                results.append([en,size,seed,"Perfect_knowledge",0,te.original_y[en,:]@x_P.value,p,d])
            except:
                results.append([en,size,seed,"Perfect_knowledge",0,0,0,np.nan])



            # Deterministic (training)
            try:
                p,d = out_of_sample_per_and_dis_CVAR_simple(conditional,prob.value,x_DET_rolling.value.T,alpha,lamb,mux)
                results.append([en,size,seed,"Unconditional_SAA",0,te.original_y[en,:]@x_DET_rolling.value,p,d])
            except:
                results.append([en,size,seed,"Unconditional_SAA",0,0,0,np.nan])


            # Unconditional Wass (same decisions)
            p,d = out_of_sample_per_and_dis_CVAR_simple(conditional,prob_WASS_Uncon.get(),x_WASS_Uncon.get(),alpha,lamb,mux)
            results.append([en,size,seed,"Uncon_WASS",0,te.original_y[en,:]@x_WASS_Uncon.get(),p,d])


        for exp in tqdm(range(model_tries),leave= False):
            # Train Deep Ensemble
            members = []
            n_members = 10
            for _ in range(n_members):
                ex_member = DEmember(tr.X.shape[-1],tr.y.shape[-1],[100,100,100],dropout = 0.0,act_function=torch.nn.ReLU)
                opt = Adam(ex_member.parameters(),lr= 0.0001)
                train_net(ex_member,tr,opt,gaussian_multivariate_NLL,100,128,False)
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




            # Get gamma values
            model.eval()
            likelihood.eval()
            allmu_GP, allsig_GP = [], []
            for i in range(dat.X.shape[0]):
                cond_model = likelihood(model(dat.X[[i]]))
                mu, sig = predict_normalised((cond_model.mean.detach(),cond_model.covariance_matrix.detach()),generator=tr)
                allmu_GP.append(mu)
                allsig_GP.append(sig)



            allmu_DE, allsig_DE = [i.detach() for i in ens(dat.X,generator= tr)]
            allmu_GP = torch.cat(allmu_GP)
            allsig_GP = torch.stack(allsig_GP)

            gamma1_DE = get_gamma1(allmu_DE,allsig_DE,dat.original_y,quantile)
            gamma1_GP = get_gamma1(allmu_GP,allsig_GP,dat.original_y,quantile)

            gamma2_DE, gm2de = get_gamma2(allmu_DE,allsig_DE,dat.original_y,quantile2)
            gamma2_GP, gm2gp = get_gamma2(allmu_GP,allsig_GP,dat.original_y,quantile2)

            # Get theta
            cond_modelA = likelihood(model(dat.X))
            allsam_GP = cond_modelA.rsample(torch.Size([num_sam])).transpose(0,1) * tr.std_y + tr.means_y
            theta_GP = k*get_theta(dat.original_y,allsam_GP,p_wass,quantile3)

            # Train monte carlo dropout network and get theta
            mcd = MCDnet(tr.X.shape[-1],tr.y.shape[-1],[100,100,100],dropout = 0.5,act_function=torch.nn.ReLU)
            opt = Adam(mcd.parameters(),lr= 0.0001)
            train_net(mcd,tr,opt,simpleMSE,200,128,False)

            allsam_MCD = mcd.MCD_sampling(dat.X,num_sam,generator=tr)
            theta_MCD = k*get_theta(dat.original_y,allsam_MCD,p_wass,quantile3)

            allsam_DE = []
            for i in range(dat.y.shape[0]):
                m, s = [i.detach().numpy() for i in ens(dat.X[i],generator= tr)]
                allsam_DE.append(dists.multivariate_normal.rvs(m,s,size=num_sam))
            allsam_DE = torch.tensor(np.stack(allsam_DE))
            theta_DE = k*get_theta(dat.original_y,allsam_DE,p_wass,quantile3)

            for en in tqdm(range(te.y.shape[0]),leave = False):

                conditional = te.original_X[en,:].numpy()
                # standardises the context
                inp = (torch.Tensor(conditional)-tr.means_X)/tr.std_X

                # monte carlo sample for empirical evaluation of devised solution
                mux = dist_out_ood.rvs(mean=mu_f(conditional),
                                            cov = sig_f(conditional),
                                            size=10000,
                                            random_state=eval_seed)


                with torch.no_grad():
                    muEDE, SigmaEDE = [i.detach().numpy() for i in ens(inp,generator= tr)]

                # Deep Ensemble (try except formed bc mosek occasionally fails - though problem almost entirely resolved by robustness parameter algorithm, assume no decisions in that case)
                # DE-UM
                try:
                    prob, x_DE, beta, mu0, sigma0 = solve_uncertain_moments(alpha,lamb,muEDE,SigmaEDE,
                                                                         gamma1 = gamma1_DE,
                                                                         gamma2 = gamma2_DE,
                                                                         verbose = False,
                                                                         solver = solver,
                                                                         kwargs=options)

                    p,d = out_of_sample_per_and_dis_CVAR_simple(conditional,prob.value,x_DE.value.T,alpha,lamb,mux)
                    results.append([en,size,seed,"DE",exp,te.original_y[en,:]@x_DE.value,p,d])

                except:
                    results.append([en,size,seed,"DE",exp,0,0,np.nan])

                # SAA (predicted ~ det equivalent for this type of problem)
                mus = dists.multivariate_normal.rvs(mean=muEDE,
                                                    cov=SigmaEDE,
                                                    size=num_sam,
                                                    random_state=0)
                try:
                    prob, x_DET_pred, beta = solve_deterministic_CVAR(alpha,lamb,mus)
                    p,d = out_of_sample_per_and_dis_CVAR_simple(conditional,prob.value,x_DET_pred.value.T,alpha,lamb,mux)
                    results.append([en,size,seed,"Conditional_DE_SAA",exp,te.original_y[en,:]@x_DET_pred.value,p,d])

                except:
                    results.append([en,size,seed,"Conditional_DE_SAA",exp,0,0,np.nan])


                prob, x_WASS_DE, beta, m_wass = solve_portfolio_wasserstein(samples = mus,
                                        alpha = alpha,
                                        lamb = lamb,
                                        theta = theta_DE,
                                        solver = msk,
                                        nor = p_wass,
                                           )
                try:
                    p,d = out_of_sample_per_and_dis_CVAR_simple(conditional,prob.get(),x_WASS_DE.get(),alpha,lamb,mux)
                    results.append([en,size,seed,"DE_WASS",exp,te.original_y[en,:]@x_WASS_DE.get(),p,d])

                except:
                    results.append([en,size,seed,"DE_WASS",exp,0,0,np.nan])



                # GP-UM, using inferred conditional distribution for uncertain moment ambiguity sets

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
                    p,d = out_of_sample_per_and_dis_CVAR_simple(conditional,prob.value,x_GP.value.T,alpha,lamb,mux)
                    results.append([en,size,seed,"GP",exp,te.original_y[en,:]@x_GP.value,p,d])

                except:
                    results.append([en,size,seed,"GP",exp,0,0,np.nan])

                # GP informed Wasserstein appraoc
                sampleGP = cond_model.rsample(torch.Size([num_sam])) * tr.std_y + tr.means_y
                prob, x_WASS_GP, beta, m = solve_portfolio_wasserstein(samples = sampleGP.detach().numpy().squeeze(),
                                        alpha = alpha,
                                        lamb = lamb,
                                        theta = theta_GP,
                                        solver = msk,
                                        nor = p_wass
                                           )
                try:
                    p,d = out_of_sample_per_and_dis_CVAR_simple(conditional,prob.get(),x_WASS_GP.get(),alpha,lamb,mux)
                    results.append([en,size,seed,"GP_WASS",exp,te.original_y[en,:]@x_WASS_GP.get(),p,d])
                except:
                    results.append([en,size,seed,"GP_WASS",exp,0,0,np.nan])


                #Monte carlo dropout net informed Wasserstein appraoc
                sampleMCD = mcd.MCD_sampling(inp.unsqueeze(0),num_sam,generator=tr).numpy()

                prob, x_MCD, beta, m = solve_portfolio_wasserstein(samples = sampleMCD.squeeze(),
                                        alpha = alpha,
                                        lamb = lamb,
                                        theta = theta_MCD,
                                        solver = msk,
                                        nor = p_wass
                                           )
                try:
                    p,d = out_of_sample_per_and_dis_CVAR_simple(conditional,prob.get(),x_MCD.get(),alpha,lamb,mux)
                    results.append([en,size,seed,"MCD_WASS",exp,te.original_y[en,:]@x_MCD.get(),p,d])
                except:
                    results.append([en,size,seed,"MCD_WASS",exp,0,0,np.nan])

# saves the dataframe in a pickle file
results = pd.DataFrame(results,columns = res_cols)
results.to_pickle(results_file)
