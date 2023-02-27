# Code for the experiments in the paper "Contextual Robust Optimisation with Uncertainty Quantification"

This paper was accepted to the short paper track at CPAIOR 2023. The core ambition was to develop pipelines that exploit covariate information to construct more appropriate ambiguity/uncertainty sets. On the flip side, this enables us to achieve a degree of decision robustness with respect to the prediction model, reducing the risk of overfitting in the prediction model leading to severely suboptimal decisions.

## Abstract

> We propose two pipelines for convex optimisation problems with uncertain parameters that aim to improve decision robustness by addressing the sensitivity of optimisation to parameter estimation. This is achieved by integrating uncertainty quantification (UQ) methods for supervised learning into the ambiguity sets for distributionally robust optimisation (DRO). The pipelines leverage learning to produce contextual/conditional ambiguity sets from side-information. The two pipelines correspond to different UQ approaches: i) explicitly predicting the conditional covariance matrix using deep ensembles (DEs) and Gaussian processes (GPs), and ii) sampling using Monte Carlo dropout, DEs, and GPs. We use i) to construct an ambiguity set by defining an uncertainty around the estimated moments to achieve robustness with respect to the prediction model. UQ ii) is used as an empirical reference distribution of a Wasserstein ball to enhance out of sample performance. DRO problems constrained with either ambiguity set are tractable for a range of convex optimisation problems. We propose data-driven ways of setting DRO robustness parameters motivated by either coverage or out of sample performance. These parameters provide a useful yardstick in comparing the quality of UQ between prediction models. The pipelines are computationally evaluated and compared with deterministic and unconditional approaches on simulated and real-world portfolio optimisation problems.

## Repository comments

I apologise if the code is confusing or not particularly well organised.

There are two testing files, one associated with generated data: *test_UMO_WASS_CVAR_gen.py* which generates its own data, and *test_UMO_WASS_CVAR_real.py* which takes data from [here](https://archive.ics.uci.edu/ml/datasets/CNNpred%3A+CNN-based+stock+market+prediction+using+a+diverse+set+of+variables) (the unzipped folder should be in the top folder). These test files contain the implemented prediction to optimisation pipelines, the predictive models are in *predictive_models.py*, the functions for the optimisation models and the algorithms to find robustness parameters are in *utils_optimisation.py*, and the data generation and objects are in *data.py*. The *multivariate_laplace.py* file is an [implementation](https://github.com/david-salac/multivariate-Laplace-extension-for-SciPy) of the namesake distribution, but was not used the experiments that made it into the paper
