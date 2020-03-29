#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script does all computation for experiment 2 (Diffusion and sorption)

# case "GP"
run_spacefilling('iso')

# case "GP+sampling"
run_50_trials('iso','alt', 0)

# case "GP+sampling+allocation"
run_50_trials('iso','refine', 20)

# case "Monte Carlo"
import numpy as np
from helpers import compute_lbme_from_ll, load_problem
import bbi

experiment = 'iso'
n_MC_repeats = 400

n_sample_sizes = np.logspace(1,np.log10(1.5e5),100,dtype=int)
n_MC_iters = n_sample_sizes.size
mc_filename = 'output/iso_mc.npz'

np.random.seed(0)

# save mode
selection_problem, problems, fields, names, n_sample, n_subsample = load_problem(experiment)
n_models = len(problems)

# compute reference solution
ll = np.array([problems[m].compute_loglikelihood() for m in range(n_models)])
reference_lbmes = np.array([compute_lbme_from_ll(ll[m]) for m in range(n_models)])

# pre-allocate array for mc_errors
MC_errors = np.zeros((n_MC_repeats, n_MC_iters))
max_sample_size = n_sample_sizes.max()
for i_MC_repeat in range(n_MC_repeats):
    indices = np.random.choice(n_sample, max_sample_size, replace=True)
    ll_sub = ll[:,indices]
    
    for i_iter, n in enumerate(n_sample_sizes):
        # distribute sampling over all models, so each model has an integer 
        # sample size, and sizes add up to n.
        sample_size_per_model = np.arange(n,n+n_models)//n_models
        MC_lbmes = np.array([compute_lbme_from_ll(ll_sub[i_model, :(n-1)]) for i_model in range(n_models)])
        MC_errors[i_MC_repeat,i_iter] = bbi.kldiv(reference_lbmes, MC_lbmes)    

np.savez(mc_filename, MC_errors=MC_errors, n_sample_sizes = n_sample_sizes)    


