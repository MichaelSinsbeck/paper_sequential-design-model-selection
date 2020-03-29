#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from helpers import load_problem

experiments = ['synth', 'iso']
cases = ['spacefilling', 'alt', 'refine']

#experiments = ['synth']
#experiments = ['iso']
#cases = ['spacefilling', 'refine']

run_idx = 0



for experiment in experiments:
    # load experiment data
    np.random.seed(run_idx)
    problem, _, _, names, _, _ = load_problem(experiment)
    lbme_true = problem.compute_lbme()
    bme_true = np.exp(lbme_true)

    # determine n_eval
    filename = 'output/{}_refine_run{}.npz'.format(experiment,run_idx)
    content = np.load(filename)
    n_eval = content['n_eval']

# kombinierter plot:
#  1) allocation punkte plot
#  2) BME estimates über iterationen (mit allocation punkten)
#  3) Kriterium über iterationen

# data files:
# 1) n_eval bme_0 bme_1 ... bme_true_0 bme_true_1 ...
    lbmes = content['lbmes']
    n_models = lbmes.shape[0]
    bmes = np.exp(lbmes)
    
    plotdata = np.zeros((n_eval.size, 1+2*n_models))
    plotdata[:,0] = n_eval
    plotdata[:,1:1+n_models] = bmes.T
    plotdata[:,1+n_models:1+2*n_models] = bme_true
    
    filename = 'plotdata/{}_single_bmes.data'.format(experiment)
    np.savetxt(filename, plotdata, fmt='%.6e')
    
# 2) n_eval crit_0 crit_1 ...
    criteria = content['criteria']
    plotdata = np.zeros((n_eval.size-1, 1+n_models))
    plotdata[:,0] = n_eval[1:]
    plotdata[:,1:] = criteria.T
    
    filename = 'plotdata/{}_single_criteria.data'.format(experiment)
    np.savetxt(filename, plotdata, fmt='%.6e')

    # for each model:    
    # 3) (allocation punkte plot, model 0) n_eval, "0", bme_0, crit_0
    model_idx = content['model_idx']
    for m in range(n_models):
        this_index = (model_idx == m)
        
        count = np.sum(this_index)
        plotdata = np.zeros((count,3))
        plotdata[:,0] = n_eval[1:][this_index]
        plotdata[:,1] = m
        plotdata[:,2] = bmes[m,1:][this_index]
        
        print('Evaluation counts, {}, {}: {}'.format(experiment, names[m], count))
        
        filename = 'plotdata/{}_single_allocation_{}.data'.format(experiment, names[m])
        np.savetxt(filename, plotdata, fmt='%.6e')
        # hier eventuell model order wieder einführen


    # error plot:
    # file 1:  spacefilling vs even alternation vs full-sequential
    n_repeats = 51
    e_raw = np.zeros((n_eval.size,n_repeats,len(cases)))

    for i in range(n_repeats):
        for j, case in enumerate(cases):
            filename = 'output/{}_{}_run{}.npz'.format(experiment, case, i)
            content = np.load(filename)
            e_raw[:,i,j] = content['errors']
    plotdata = np.zeros((n_eval.size, len(cases)+1))
    plotdata[:,0] = n_eval
    plotdata[:,1:1+len(cases)] = np.median(e_raw, axis = 1)
    filename = 'plotdata/{}_errors.data'.format(experiment)
    np.savetxt(filename, plotdata, fmt='%.6e')
    
    # file 2: MC-error
    filename = 'output/{}_mc.npz'.format(experiment)
    content = np.load(filename)
    n_sample_sizes = content['n_sample_sizes']
    errors = np.median(content['MC_errors'], axis = 0)
    
    plotdata = np.zeros((n_sample_sizes.size,2))
    plotdata[:,0] = n_sample_sizes
    plotdata[:,1] = errors
    
    filename = 'plotdata/{}_mc.data'.format(experiment)
    np.savetxt(filename, plotdata, fmt='%i %.6e')
    
    mask = (n_sample_sizes < n_eval.size+2)
    filename = 'plotdata/{}_mc_short.data'.format(experiment)
    np.savetxt(filename, plotdata[mask,:], fmt='%i %.6e')
    
#%% Creating data for failed-example
    
content = np.load('output/failure_example.npz')


n_eval = content['n_eval']
criteria = content['criteria']
plotdata = np.zeros((n_eval.size-1, 3))
plotdata[:,0] = n_eval[1:]
plotdata[:,1:] = criteria.T

filename = 'plotdata/synth_failure.data'
np.savetxt(filename, plotdata, fmt='%.6e')
