#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:10:37 2019

@author: micha
"""
import bbi
import numpy as np
from scipy.special import lambertw
import scipy.io as io

def scale_input(array):
    return (array - array.mean(axis=0)[np.newaxis,:]) / array.std(axis=0)[np.newaxis,:]


def compute_lbme_from_ll(ll):
    ll_shifted = ll - ll.max()
    lbme = np.log(np.mean(np.exp(ll_shifted))) + ll.max()
    return lbme

def load_problem(experiment):
    # Code and models adapted from Marvin Hoege, 7/2018
    if (experiment == 'kin'):
        time_end = 30
        n_output = time_end * 2
        time_range = np.linspace(0, time_end, n_output)
        
        c_0 = 1             # initial concentration
        r_max_true = 0.1    # maximum reaction rate
        K_true = 1          # half velocity concentration
        
        # Zeroth order model: p[0] = r_max
        def model_0th(p):
            return np.array([max(c_0 - p[0] * t, 0) for t in time_range])
        
        # First order model: p[0] = r_max / K
        def model_1st(p):
            return c_0 * np.exp(-p[0] * time_range)
        
        # Michaelis-Menten kinetics model: p[0] = r_max, p[1] = K
        def model_MMK(p):
            return (p[1] * lambertw(c_0 / p[1] * np.exp(-p[0] / p[1] * time_range + c_0 / p[1]))).real
        
        obs_sigma = 0.1
        y = model_MMK([r_max_true, K_true]) + np.random.normal(size=time_range.size) * obs_sigma
        data = bbi.Data(y.real, obs_sigma)
        
        r_max_low = 0.01
        r_max_high = 0.15
        K_mean = np.log(K_true)
        K_sigma = 0.01
        
        n_sample = 8000
        r_max_samples = np.random.uniform(r_max_low, r_max_high, size=n_sample)
        K_samples = np.random.lognormal(K_mean, K_sigma, size=n_sample)
        ratio_samples = r_max_samples / K_samples
        
        grid_0th = r_max_samples[:, np.newaxis]
        grid_1st = ratio_samples[:, np.newaxis]
        grid_mmk = np.array([r_max_samples, K_samples]).T
        
        problem_mmk = bbi.Problem(grid_mmk, model_MMK, data)
        problem_0th = bbi.Problem(grid_0th, model_0th, data)
        problem_1st = bbi.Problem(grid_1st, model_1st, data)
        problems = [problem_mmk, problem_0th, problem_1st]
        
        n_subsample = 500
        grids = [grid_mmk, grid_0th, grid_1st]
        models = [model_MMK, model_0th, model_1st]
        names = ['MMK', '0th order', '1st order']
        ani = [2,1,1]
        
    # Models adapted from Anneli Guthke
    if (experiment == 'synth'):
        n_output = 15
        meas_loc = np.linspace(0.25, 4.75, n_output)
        
        # Linear model y = a*x+b
        def L2_model(p):
            return p[0] * meas_loc + p[1]
        
        # Nonlinear model y = a*cos(b*x+c)+d
        def NL4_model(p):
            return p[0] * np.cos(p[1] * meas_loc + p[2]) + p[3]
        
        sigma = 0.6
        content = io.loadmat('data/synth_data.mat')
        data = bbi.Data(content['d'], sigma**2)
        
        L2_prior_mean = np.array([1, 0])
        L2_prior_cov = np.array([[0.04, -0.007],[-0.007, 0.04]])
        NL4_prior_mean = np.array([2.6, 0.5, -2.8, 2.3])
        NL4_prior_cov = np.array([[0.46, -0.07, 0.24, -0.14],
                                  [-0.07, 0.04, -0.05, 0.02],
                                  [0.24, -0.05, 0.30, -0.16],
                                  [-0.14, 0.02, -0.16, 0.30]])
        
        n_sample = 10000
        L2_grid = np.random.multivariate_normal(L2_prior_mean, L2_prior_cov, size=n_sample)
        NL4_grid = np.random.multivariate_normal(NL4_prior_mean, NL4_prior_cov, size=n_sample)
        
        L2_problem = bbi.Problem(L2_grid, L2_model, data)
        NL4_problem = bbi.Problem(NL4_grid, NL4_model, data)
        problems = [L2_problem, NL4_problem]
        
        n_subsample = 500
        grids = [L2_grid, NL4_grid]
        models = [L2_model, NL4_model]
        names = ['linear', 'cosine']    
        ani = [2,4]
        
    # Data and models adapted from Anneli Guthke, 11/2016
    if (experiment == 'iso'):
        content = io.loadmat('data/iso_output.mat')
        params = io.loadmat('data/iso_input.mat')
        n_days = 20
        
        std_rel = 0.05
        std_abs = 2e-7 * 1e7
        d = np.load('data/iso_data.npy')
        data = bbi.Data(0, (d*std_rel + std_abs)**2)
        n_output = data.value.size
        
        param_lin = scale_input(params['MC_paras_M1'].T)
        param_fre = scale_input(params['MC_paras_M2'].T)
        param_lan = scale_input(params['MC_paras_M3'].T)
        
        mc_data = content['MC_dataw'][:n_days] * 1e7
        mc_data -= d    
        
        n_realizations = 51000
        reali_lin = mc_data[:,0:n_realizations].T
        reali_fre = mc_data[:,n_realizations:(2*n_realizations)].T
        reali_lan = mc_data[:,(2*n_realizations):].T
    
        problem_lin = bbi.Problem(param_lin, reali_lin, data)
        problem_fre = bbi.Problem(param_fre, reali_fre, data)
        problem_lan = bbi.Problem(param_lan, reali_lan, data)
        problems = [problem_lin, problem_fre, problem_lan]
        
        n_sample = n_realizations
        n_subsample = 500
        grids = [param_lin, param_fre, param_lan]
        models = [reali_lin, reali_fre, reali_lan]
        names = ['linear', 'Freundlich', 'Langmuir']    
        ani = [6,6,6]
        
    n_models = len(problems)
    selection_problem = bbi.SelectionProblem(grids, models, data)
    fields = [bbi.MixSquaredExponential([0.1, 10], [0.01, 1e4], n_output, anisotropy = ani[m]) for m in range(n_models)]
    return selection_problem, problems, fields, names, n_sample, n_subsample


def run_50_trials(experiment, crit, starting_phase):
    print('Starting 50 trials with experiment {} and crit = {}'.format(experiment, crit))

    if experiment == 'synth':
        n_iters = 100
    if experiment == 'iso':
        n_iters = 150
        
    # note: experiment "synth" re-generates the underlying grid 
    # for each "load_problem"-call. That means, the true solution changes every
    # time. Since we compute errors case-wise, we can still average errors.
    
    for i in range(51):
        np.random.seed(i)
        filename = 'output/{}_{}_run{}.npz'.format(experiment, crit, i)
        
        selection_problem, problems, fields, names, n_sample, n_subsample = load_problem(experiment)
        
        lbmes, model_idx, n_eval, criteria = bbi.select_model(
                    selection_problem, fields, n_iters, 
                    n_subsample = n_subsample, 
                    starting_phase = starting_phase, 
                    crit = crit)
        
        lbme_true = selection_problem.compute_lbme()
        errors = np.array([bbi.kldiv(lbme_true, lbme) for lbme in lbmes.T])
        
        np.savez(filename,
                 lbmes = lbmes,
                 criteria = criteria,
                 n_eval = n_eval,
                 model_idx = model_idx,
                 errors = errors
                 )
        
def run_spacefilling(experiment):
    print('Starting space-filling with experiment {}'.format(experiment))

    if experiment == 'synth':
        n_iters = 100
    if experiment == 'iso':
        n_iters = 150
    
    for i in range(51):
        np.random.seed(i)
        filename = 'output/{}_spacefilling_run{}.npz'.format(experiment,i)
        
        selection_problem, problems, fields, names, n_sample, n_subsample = load_problem(experiment)

        lbmes, model_idx, n_eval = bbi.select_model_spacefilling(
                selection_problem, fields, n_iters,
                n_subsample = n_subsample)
        
        lbme_true = selection_problem.compute_lbme()
        
        errors = np.array([bbi.kldiv(lbme_true, lbme) for lbme in lbmes.T])
        
        np.savez(filename,
                 lbmes = lbmes,
                 model_idx = model_idx,
                 n_eval = n_eval,
                 errors = errors
                 )

def run_failed_example():
    print('Starting failed example without starting phase')
    
    experiment = 'synth'
    n_iters = 100
    np.random.seed(0)
    filename = 'output/failure_example.npz'
    
    selection_problem, problems, fields, names, n_sample, n_subsample = load_problem(experiment)

    lbmes, model_idx, n_eval, criteria = bbi.select_model(
                selection_problem, fields, n_iters, 
                n_subsample = n_subsample, 
                starting_phase = 0, 
                crit = 'refine')
        
    lbme_true = selection_problem.compute_lbme()
    errors = np.array([bbi.kldiv(lbme_true, lbme) for lbme in lbmes.T])
        
    np.savez(filename,
             lbmes = lbmes,
             criteria = criteria,
             n_eval = n_eval,
             model_idx = model_idx,
             errors = errors
             )

