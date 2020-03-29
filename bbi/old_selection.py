#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module modelSelection
"""
import numpy as np
from bbi.field import GpeSquaredExponential, AbstractMix
from bbi.design import find_optimal_node_linear
from bbi.mini_classes import Nodes, Problem
from bbi.mini_functions import kldiv


def select_model_old(subproblems, problems, fields, n_iters, nodes=None, params=None, crit='sigma', n_realizations=1000):
    n_models = len(subproblems)
    n_eval = np.arange(n_iters+1)
    n_subsample = subproblems[0].grid.shape[0]
    
    if nodes is None:
        nodes = np.array([Nodes() for m in range(n_models)])
    else:
        for nodes_idx in range(n_models):
            if nodes[nodes_idx] is None:
                nodes[nodes_idx] = Nodes()
                
    conditioned_fields = []
    lbmes = np.zeros((n_models, n_iters+1))
    if (isinstance(fields[0], AbstractMix)): 
        print('select_model_' + crit + '_map   ', end='')
        for m in range(n_models):
            map_field, map_params = get_field_and_params(fields[m], nodes[m])
            conditioned_fields.append(map_field.condition_to(nodes[m]))
            lbmes[m,0] = estimate_lbme(subproblems[m], problems[m], map_params[0], map_params[1], nodes[m])
    else:
        print('select_model_' + crit + '   ', end='')
        for m in range(n_models):
            conditioned_fields.append(fields[m].condition_to(nodes[m]))
            lbmes[m,0] = estimate_lbme(subproblems[m], problems[m], params[m][0], params[m][1], nodes[m])
            
    realizations = np.array([conditioned_fields[m].draw_many_realizations(n_realizations) for m in range(n_models)])
    
    initial_ll_estimates = np.zeros((n_models, n_subsample))
    design_lbmes = np.zeros((n_models, n_iters+1))
    lbme_realizations = np.zeros((n_models, n_realizations))
    for m in range(n_models):        
        initial_ll_estimates[m,:] = conditioned_fields[m].estimate_loglikelihood(subproblems[m].data)
        #initial_l_estimates = conditioned_fields[m].estimate_likelihood(subproblems[m].data)
        design_lbmes[m,0] = compute_lbme_from_ll(initial_ll_estimates[m])
        lbme_realizations[m,:] = compute_lbme_over_realizations(subproblems[m], realizations[m])
    #design_bmes[:,0] = np.mean(initial_l_estimates, axis=1)
    
    i_max = 0
    model_idx = np.zeros(n_iters+1)
    criteria = np.zeros((n_models, n_iters))
    for i in range(n_iters):
        print('.', end='')
        if (i > 0):
            realizations[i_max] = conditioned_fields[i_max].draw_many_realizations(n_realizations)
            lbme_realizations[i_max] = compute_lbme_over_realizations(subproblems[i_max], realizations[i_max])
            
        if (crit == 'kldiv'):
            criteria[:,i] = compute_kl_distances(lbme_realizations, design_lbmes[:,i])
            i_max = criteria[:,i].argmax()
        elif (crit == 'sigma'):
            criteria[:,i] = compute_sigmas(lbme_realizations)
            i_max = criteria[:,i].argmax()
        elif (crit == 'alt'):
            i_max = i % n_models
            
        model_idx[i+1] = i_max + 1
        
        if (isinstance(fields[0], AbstractMix)):
            ll_estimate, this_params = advance_one_model_map(conditioned_fields, i_max, subproblems, fields, nodes)
        else:
            ll_estimate = advance_one_model(conditioned_fields, i_max, subproblems, fields, nodes)
            this_params = params[i_max]
            
        design_lbme_estimate = compute_lbme_from_ll(ll_estimate)
        design_lbmes = update_lbmes(design_lbmes, design_lbme_estimate, i_max, i) 
        lbme_estimate = estimate_lbme(subproblems[i_max], problems[i_max], this_params[0], this_params[1], nodes[i_max])   
        lbmes = update_lbmes(lbmes, lbme_estimate, i_max, i)  
    
    print('')
    return lbmes, model_idx, n_eval


def compute_true_error(problems, lbmes, n_iters):
    n_models = len(problems)
    true_lbmes = np.array([compute_lbme(problems[m]) for m in range(n_models)])
    errors = np.array([kldiv(true_lbmes, lbmes[:,i]) for i in range(n_iters)])
    return errors


def compute_lbme(problem):
    ll = problem.compute_loglikelihood()
    #likelihood = problem_discrete.compute_likelihood()
    lbme = compute_lbme_from_ll(ll)
    #bme = np.mean(likelihood)
    return lbme


def compute_lbme_from_ll(ll):
    ll_shifted = ll - ll.max()
    lbme = np.log(np.mean(np.exp(ll_shifted))) + ll.max()
    return lbme


def estimate_lbme(subproblem, problem, l, var, nodes):
    this_grid = np.vstack((subproblem.grid, problem.grid))
    this_field = GpeSquaredExponential(l, var, this_grid)
    ll_estimate = this_field.estimate_conditional_likelihood(nodes, subproblem.data)
    # Remove all initial grid observations
    ll_estimate_new = ll_estimate[subproblem.grid.shape[0]:]
    lbme = compute_lbme_from_ll(ll_estimate_new)
    #bme = np.mean(l_estimate_new)
    return lbme


def compute_lbme_over_realizations(realization, grid, data):
    n_realizations = len(realization)
    lbme_realizations = np.zeros(n_realizations)
    for r in range(n_realizations):
        this_problem = Problem(grid, realization[r], data)
        lbme_realizations[r] = compute_lbme(this_problem)
        #bme_realizations[r] = np.mean(this_problem).compute_likelihood())
    return lbme_realizations


def compute_lbme_over_realizations_old(subproblem, realization):
    n_realizations = len(realization)
    lbme_realizations = np.zeros(n_realizations)
    for r in range(n_realizations):
        data = subproblem.data
        this_problem = Problem(subproblem.grid, realization[r], data)
        lbme_realizations[r] = compute_lbme(this_problem)
        #bme_realizations[r] = np.mean(this_problem).compute_likelihood())
    return lbme_realizations


def compute_sigmas(lbme_realizations):
    n_models = lbme_realizations.shape[0]
    sigmas = np.zeros(n_models)
    for m in range(n_models):
        # "Normalize" lbme
        this_lbme = lbme_realizations[m]
        this_lbme = this_lbme - this_lbme.max()
        #this_bme = this_bme / this_bme.max()
        sigmas[m] = np.std(np.exp(this_lbme)) / np.mean(np.exp(this_lbme))
        #sigmas[m] = np.std(this_bme) / np.mean(this_bme)
    return sigmas


def compute_kl_distances(lbme_realizations, this_lbme):
    n_models, n_realizations = lbme_realizations.shape
    kl_distances = np.zeros((n_models, n_realizations))
    for m in range(n_models):
        sub_realization = this_lbme.copy()
        for r in range(n_realizations):
            sub_realization[m] = lbme_realizations[m,r]
            kl_distances[m,r] = kldiv(this_lbme, sub_realization)
            #kl_distances[m,r] = kldiv(np.log(bme), np.log(sub_realization))
    kl_distances_mean = kl_distances.mean(axis=1)
    return kl_distances_mean


def advance_one_model(conditioned_fields, i_max, subproblems, fields, nodes):
    data = subproblems[i_max].data
    new_index = find_optimal_node_linear(nodes[i_max], conditioned_fields[i_max], data)
    y = subproblems[i_max].evaluate_model(new_index)
    nodes[i_max].append(new_index, y)
    
    conditioned_fields[i_max] = fields[i_max].condition_to(nodes[i_max])
    ll_est = conditioned_fields[i_max].estimate_loglikelihood(data)
    #ll_est = conditioned_fields[i_max].estimate_likelihood(data)
    return ll_est


def advance_one_model_map(conditioned_fields, i_max, subproblems, mixes, nodes):
    data = subproblems[i_max].data
    new_index = find_optimal_node_linear(nodes[i_max], conditioned_fields[i_max], data)
    y = subproblems[i_max].evaluate_model(new_index)
    nodes[i_max].append(new_index, y)
    
    field, map_params = get_field_and_params(mixes[i_max], nodes[i_max])
    conditioned_fields[i_max] = field.condition_to(nodes[i_max])
    ll_est = conditioned_fields[i_max].estimate_loglikelihood(data)
    #ll_est = conditioned_fields[i_max].estimate_likelihood(data)
    return ll_est, map_params


def get_field_and_params(mix, nodes):
    map_xi = mix.get_map_xi(nodes)
    map_params = mix.xi_to_parameters(map_xi)
    field = mix.get_map_field(nodes)
    return field, map_params


def update_lbmes(lbmes, lbme_estimate, i_max, i):
    lbmes[:,i+1] = lbmes[:,i]
    lbmes[i_max,i+1] = lbme_estimate
    return lbmes
