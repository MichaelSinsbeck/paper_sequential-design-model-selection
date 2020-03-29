#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module modelSelection
"""
import numpy as np
from bbi.field import AbstractMix
from bbi.design import compute_criterion_linear, make_subgrid
from bbi.mini_classes import Nodes, Problem
from bbi.mini_functions import kldiv
import copy


def select_model(problem, fields, n_iters, nodes = None, crit = 'sigma', n_subsample = None, n_realizations = 1000, starting_phase = 0):
    print('select_model  ', end='')
    n_models = problem.n_models
    n_eval = np.arange(n_iters + 1) # todo: add nodes.idx.size    
    grids = problem.grids
    
    # create empty nodes if necessary
    if nodes is None:
        nodes = [Nodes() for m in range(n_models)]
    else:
        for i_model in range(n_models):
            if nodes[i_model] is None:
                nodes[i_model] = Nodes()
            else:
                nodes[i_model] = copy.deepcopy(nodes[i_model])
    
    # condition prior fields to given nodes and compute iteration-zero-bme            
    conditioned_fields = []
    lbmes = np.zeros((n_models, n_iters+1))
    for i in range(n_models):
        if (isinstance(fields[i], AbstractMix)):
            this_prior_field = fields[i].get_map_field(nodes[i], grids[i])
            this_field = this_prior_field.condition_to(nodes[i], grids[i])
        else:
            this_field = fields[i].condition_to(nodes[i], grids[i])

        conditioned_fields.append(this_field)            
        lbmes[i,0] = this_field.estimate_lbme(grids[i], problem.data)
    
    model_idx = []
    criteria = np.zeros((n_models, n_iters))
    lbme_realizations = np.zeros((n_models, n_realizations))
    for i in range(n_iters):
        # compute criterion (for finding next node)
        next_node_idx = []
        crit_max = []
        for i_model, this_field in enumerate(conditioned_fields):
            subgrid, subindex = make_subgrid(grids[i_model], n_subsample, nodes[i_model])
            discrete_field = this_field.discretize(subgrid)

            realizations = discrete_field.draw_many_realizations(n_realizations)
            lbme_realizations[i_model,:] = compute_lbme_over_realizations(realizations, grids[i_model], problem.data)
            
            this_criterion = compute_criterion_linear(discrete_field, problem.data)
            new_index = subindex[np.argmax(this_criterion)]
            next_node_idx.append(new_index)
            crit_max.append(np.max(this_criterion))
        
        # compute criterion (for allocation)
        if (crit == 'kldiv'):
            criteria[:,i] = compute_kl_distances(lbme_realizations, lbmes[:,i])
        elif (crit == 'sigma'):
            criteria[:,i] = compute_sigmas(lbme_realizations)
        elif (crit == 'alt'):
            criteria[i % n_models,i] = 1
        elif (crit == 'refine'): # use refinement criterion of seq. des.
            criteria[:,i] = np.array(crit_max)
        else:
            raise NotImplementedError(
            'Unknown criterion given. please use sigma, alt or refine')

        # apply critaria, but only after starting phase
        # in starting phase, use even alternation
        if i >= n_models * starting_phase:
            i_max = criteria[:,i].argmax()
        else:
            i_max =i % n_models
            print('_',end='')
            
        print(i_max, end='')
        new_index = next_node_idx[i_max]
        model_idx.append(i_max)
        
        # advance respective model
        y = problem.evaluate_model(i_max, new_index)
        nodes[i_max].append(new_index, y)
        
        # map-estimate, if necessary
        if (isinstance(fields[i_max], AbstractMix)):
            this_prior_field = fields[i_max].get_map_field(nodes[i_max], grids[i_max])
            this_field = this_prior_field.condition_to(nodes[i_max], grids[i_max])
        else:
            this_field = fields[i_max].condition_to(nodes[i_max], grids[i_max])
        conditioned_fields[i_max] = this_field
        
        lbmes[:,i+1] = lbmes[:,i]
        lbmes[i_max, i+1] = this_field.estimate_lbme(grids[i_max], problem.data)

    print('')
    
    return lbmes, np.array(model_idx), n_eval, criteria


def select_model_spacefilling(problem, fields, n_iters, nodes = None, n_subsample = None):
    print('space filling ', end='')
    n_models = problem.n_models
    n_eval = np.arange(n_iters + 1) # todo: add nodes.idx.size    
    grids = problem.grids
    
        # create empty nodes if necessary
    if nodes is None:
        nodes = [Nodes() for m in range(n_models)]
    else:
        for i_model in range(n_models):
            if nodes[i_model] is None:
                nodes[i_model] = Nodes()
            else:
                nodes[i_model] = copy.deepcopy(nodes[i_model])
    
    # condition prior fields to given nodes and compute iteration-zero-bme            
    conditioned_fields = []
    lbmes = np.zeros((n_models, n_iters+1))
    for i in range(n_models):
        if (isinstance(fields[i], AbstractMix)):
            this_prior_field = fields[i].get_map_field(nodes[i], grids[i])
            this_field = this_prior_field.condition_to(nodes[i], grids[i])
        else:
            this_field = fields[i].condition_to(nodes[i], grids[i])

        conditioned_fields.append(this_field)            
        lbmes[i,0] = this_field.estimate_lbme(grids[i], problem.data)

    model_idx = []
    for i in range(n_iters):
        i_max = i%n_models
        model_idx.append(i_max)
        print(i_max, end='');
        
        this_field = conditioned_fields[i_max]
        # choose variance-minimizing point
        subgrid, subindex = make_subgrid(grids[i_max], n_subsample, nodes[i_max])
        discrete_field = this_field.discretize(subgrid)
        
        diag_c = np.diag(discrete_field.c)
        criterion = np.zeros_like(diag_c)
        mask = (diag_c > 0)
        criterion[mask] = np.sum(discrete_field.c**2, axis = 0)[mask] / np.diag(discrete_field.c)[mask]
        new_index = subindex[np.argmax(criterion)]
              
        y = problem.evaluate_model(i_max, new_index);
        nodes[i_max].append(new_index, y)

        # map-estimate, if necessary
        if (isinstance(fields[i_max], AbstractMix)):
            this_prior_field = fields[i_max].get_map_field(nodes[i_max], grids[i_max])
            this_field = this_prior_field.condition_to(nodes[i_max], grids[i_max])
        else:
            this_field = fields[i_max].condition_to(nodes[i_max], grids[i_max])
        conditioned_fields[i_max] = this_field
        
        lbmes[:,i+1] = lbmes[:,i]
        lbmes[i_max, i+1] = this_field.estimate_lbme(grids[i_max], problem.data)        

    print('')

    return lbmes, np.array(model_idx), n_eval

def compute_lbme(problem):
    ll = problem.compute_loglikelihood()
    ll_shifted = ll - ll.max()
    lbme = np.log(np.mean(np.exp(ll_shifted))) + ll.max()
    return lbme


def compute_lbme_over_realizations(realization, grid, data):
    n_realizations = len(realization)
    lbme_realizations = np.zeros(n_realizations)
    for r in range(n_realizations):
        this_problem = Problem(grid, realization[r], data)
        lbme_realizations[r] = compute_lbme(this_problem)
    return lbme_realizations


def compute_sigmas(lbme_realizations):
    n_models = lbme_realizations.shape[0]
    sigmas = np.zeros(n_models)
    for m in range(n_models):
        # "Normalize" lbme
        this_lbme = lbme_realizations[m]
        #this_lbme = this_lbme - this_lbme.max()
        #sigmas[m] = np.std(np.exp(this_lbme)) / np.mean(np.exp(this_lbme))
        sigmas[m] = np.std(np.exp(this_lbme))
    return sigmas


def compute_kl_distances(lbme_realizations, this_lbme):
    n_models, n_realizations = lbme_realizations.shape
    kl_distances = np.zeros((n_models, n_realizations))
    for m in range(n_models):
        sub_realization = this_lbme.copy()
        for r in range(n_realizations):
            sub_realization[m] = lbme_realizations[m,r]
            kl_distances[m,r] = kldiv(this_lbme, sub_realization)
    kl_distances_mean = kl_distances.mean(axis=1)
    return kl_distances_mean
