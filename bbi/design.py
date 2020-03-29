#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module sequentialDesign
Contains the following design methods (including some helper functions):
    1) design_linearized - the normal sequential design as described in my paper
        If used with a FieldColleciton, becomes the linearized version
    2) design_map - In each iteration picks the map (maximum a posteriori) field.
        Becomes an ml (maximum likelihood) method, if prior weights are uniform
    3) design_average - Finds new point by maximizing the average criterion
        over all fields (averaged using the field weights)
    4) design_sampled - computes the criterion by sampling from the random field.
"""
import numpy as np
from bbi.mini_classes import Nodes
import copy
import warnings


def design_linearized(problem, field, n_iterations, nodes=None, n_subsample = None):
    print('design_linearized   ', end='')
    if nodes is None:
        nodes = Nodes()
    else:
        nodes = copy.deepcopy(nodes)
        
    n_eval = np.arange(n_iterations+1) + nodes.idx.size

    grid = problem.grid
    n_sample = grid.shape[0]
    
    this_field = field.condition_to(nodes, grid)

    loglikelihoods = np.zeros((n_sample, n_iterations+1))
    loglikelihoods[:,0] = this_field.estimate_loglikelihood(grid, problem.data)

    for i_iteration in range(n_iterations):
        print('.', end='')
        
        subgrid, subindex = make_subgrid(grid, n_subsample, nodes)
        discrete_field = this_field.discretize(subgrid)
        
        new_sub_index = find_optimal_node_linear(discrete_field, problem.data)
        # index here is global index (of the global grid)
        new_index = subindex[new_sub_index]
        
        y = problem.evaluate_model(new_index)
        nodes.append(new_index, y)
        this_field = field.condition_to(nodes, grid)
        #print(nodes.idx)
        this_ll = this_field.estimate_loglikelihood(grid, problem.data)
        loglikelihoods[:, i_iteration + 1] = this_ll
    print('')
    return loglikelihoods, nodes, n_eval

def design_old(problem, field, n_iterations, nodes=None):
    print('design_linearized   ', end='')
    if nodes is None:
        nodes = Nodes()
    else:
        nodes = copy.deepcopy(nodes) # make a copy in order to not overwrite the node
    n_eval = np.arange(n_iterations+1) + nodes.idx.size

    this_field = field.condition_to(nodes)
    loglikelihoods = np.zeros((field.n_sample, n_iterations+1))
    loglikelihoods[:, 0] = this_field.estimate_loglikelihood(problem.data)

    for i_iteration in range(n_iterations):
        print('.', end='')
        new_index = find_optimal_node_linear_old(nodes, this_field, problem.data)
        y = problem.evaluate_model(new_index)
        nodes.append(new_index, y)
        this_field = field.condition_to(nodes)

        this_ll = this_field.estimate_loglikelihood(problem.data)
        loglikelihoods[:, i_iteration + 1] = this_ll
    print('')
    return loglikelihoods, nodes, n_eval

def make_subgrid(grid, n_subsample, nodes):
    n_sample = grid.shape[0]
    subindex_candidates = all_indices_except_visited_ones(n_sample, nodes)
    if n_subsample is None:
        subindex = subindex_candidates
    else:
        if n_subsample <= subindex_candidates.size:
            subindex = np.random.choice(subindex_candidates, n_subsample, replace = False)
        else:
            subindex = subindex_candidates
    return grid[subindex, :] , subindex

def all_indices_except_visited_ones(n_sample, nodes):
    mask = np.ones(n_sample, dtype=bool)
    mask[nodes.idx] = False
    return np.arange(n_sample)[mask]

def find_optimal_node_linear_old(nodes, field, data):
    index_list = all_indices_except_visited_ones(field.n_sample, nodes)
    criterion = compute_criterion_linear_old(index_list, field, data)
    new_index = np.argmax(criterion)
    return new_index

def compute_criterion_linear_old(index_list, field, data):
    n_sample = field.n_sample
    n_output = field.m.shape[1]
    c_is_2d = (field.c.ndim == 2)
    current_likelihood = field.estimate_likelihood_linearized(data)
    if c_is_2d:
        var_field_prior = np.diag(field.c)
    else:
        var_field_prior = np.full((n_sample, n_output), np.nan)
        for i_output in range(n_output):
            var_field_prior[:, i_output] = np.diag(field.c[:, :, i_output])

    criterion = np.full(n_sample, -np.inf)
    for this_idx in index_list:
        if c_is_2d:
            Q = field.c[this_idx, this_idx]
            q = field.c[:, this_idx]
            var_field = var_field_prior - q*q/Q
            var_field = var_field[:, np.newaxis] * np.ones((1, n_output))
        else:
            var_field = np.full((n_sample, n_output), np.nan)
            for i_output in range(n_output):
                Q = field.c[this_idx, this_idx, i_output]
                q = field.c[:, this_idx, i_output]
                var_field[:, i_output] = var_field_prior[:, i_output] - q*q/Q

        # sort out uncorrelated points
        if c_is_2d:
            idx_normal = index_list[field.c[index_list, this_idx] != 0]
        else:
            idx_normal = index_list

        # define properties of f(y_0)**2
        if c_is_2d:
            invc = (field.c[this_idx, this_idx] /
                    field.c[idx_normal, this_idx])[:, np.newaxis]
        else:
            invc = (field.c[this_idx, this_idx] /
                    field.c[idx_normal, this_idx])
        c_f = np.abs(invc)
        m_ff = (data.value - field.m[idx_normal, :]) * invc
        v_ff = 0.5*(var_field[idx_normal, :]+data.var)*(invc**2)
        c_ff = 0.5*(c_f**2)/np.sqrt(2*np.pi*v_ff)

        # define properties of p(y_0)
        n_normal = idx_normal.size
        c_p = np.ones((n_normal, n_output))
        m_p = np.zeros((n_normal, n_output))
        v_p = np.ones((n_normal, 1)) * field.c[this_idx, this_idx]

        # multiply gaussian bells
        c_total = integral_of_multiplied_normals(
            c_ff, m_ff, v_ff, c_p, m_p, v_p)
        c_var = np.prod(c_total, axis=1) - current_likelihood[idx_normal]**2
        c_var = c_var.clip(min=0)
        criterion[this_idx] = c_var.sum()/n_sample
    return criterion

def find_optimal_node_linear(field, data):
    #index_list = all_indices_except_visited_ones(field.n_sample, nodes)
    #criterion = compute_criterion_linear(index_list, field, data)
    criterion = compute_criterion_linear(field, data)
    new_index = np.argmax(criterion)
    #print(np.max(criterion))
    return new_index


def compute_criterion_linear(field, data):
    n_sample = field.n_sample
    n_output = field.m.shape[1]
    c_is_2d = (field.c.ndim == 2)
    current_likelihood = field.estimate_likelihood_linearized(data)
    if c_is_2d:
        var_field_prior = np.diag(field.c)
    else:
        var_field_prior = np.full((n_sample, n_output), np.nan)
        for i_output in range(n_output):
            var_field_prior[:, i_output] = np.diag(field.c[:, :, i_output])

    criterion = np.full(n_sample, -np.inf)
    all_indices = np.arange(n_sample)
    index_list = all_indices[np.diag(field.c) !=0] # remove zero-variance points

    for this_idx in index_list:
        if c_is_2d:
            Q = field.c[this_idx, this_idx]
            q = field.c[:, this_idx]
            var_field = var_field_prior - q*q/Q
            var_field = var_field[:, np.newaxis] * np.ones((1, n_output))
            
        else:
            var_field = np.full((n_sample, n_output), np.nan)
            for i_output in range(n_output):
                Q = field.c[this_idx, this_idx, i_output]
                q = field.c[:, this_idx, i_output]
                var_field[:, i_output] = var_field_prior[:, i_output] - q*q/Q

        # sort out uncorrelated points
        if c_is_2d:
            idx_normal = index_list[field.c[index_list, this_idx] != 0]
        else:
            idx_normal = index_list

        # define properties of f(y_0)**2
        if c_is_2d:
            invc = (field.c[this_idx, this_idx] /
                    field.c[idx_normal, this_idx])[:, np.newaxis]
        else:
            invc = (field.c[this_idx, this_idx] /
                    field.c[idx_normal, this_idx])
        c_f = np.abs(invc)
        m_ff = (data.value - field.m[idx_normal, :]) * invc
        v_ff = 0.5*(var_field[idx_normal, :]+data.var)*(invc**2)
        c_ff = 0.5*(c_f**2)/np.sqrt(2*np.pi*v_ff)

        # define properties of p(y_0)
        n_normal = idx_normal.size
        c_p = np.ones((n_normal, n_output))
        m_p = np.zeros((n_normal, n_output))
        v_p = np.ones((n_normal, 1)) * field.c[this_idx, this_idx]

        # multiply gaussian bells
        c_total = integral_of_multiplied_normals(
            c_ff, m_ff, v_ff, c_p, m_p, v_p)
        c_var = np.prod(c_total, axis=1) - current_likelihood[idx_normal]**2
        c_var = c_var.clip(min=0)
        criterion[this_idx] = c_var.sum()/n_sample
    return criterion


def integral_of_multiplied_normals(c1, m1, v1, c2, m2, v2):
    e = np.exp(-(m1-m2)**2 / (2*(v1+v2)))
    c = c1*c2/np.sqrt(2*np.pi*(v1+v2))*e
    # not computing m and v, because they are not needed
    return c


def design_map(problem, fields, n_iterations, nodes=None, n_subsample=None):
    print('design_map          ', end='')
    if nodes is None:
        nodes = Nodes()
    else:
        nodes = copy.deepcopy(nodes)
    n_eval = np.arange(n_iterations+1) + nodes.idx.size

    grid = problem.grid
    n_sample = grid.shape[0]
    
    this_prior_field = fields.get_map_field(nodes, grid)
    
    this_field = this_prior_field.condition_to(nodes, grid)
    
    loglikelihoods = np.zeros((n_sample, n_iterations+1))
    loglikelihoods[:, 0] = this_field.estimate_loglikelihood(grid, problem.data)

    for i_iteration in range(n_iterations):
        print('.', end='')
        
        subgrid, subindex = make_subgrid(grid, n_subsample, nodes)
        discrete_field = this_field.discretize(subgrid)

        with warnings.catch_warnings():
            warnings.filterwarnings('error')                
            try:
                new_sub_index = find_optimal_node_linear(discrete_field, problem.data)
                new_index = subindex[new_sub_index]
            except Warning:
                print(subindex)
                print(nodes.idx)
            
        y = problem.evaluate_model(new_index)
        nodes.append(new_index, y)

        this_prior_field = fields.get_map_field(nodes, grid)
        this_field = this_prior_field.condition_to(nodes, grid)

        this_ll = this_field.estimate_loglikelihood(grid, problem.data)
        loglikelihoods[:, i_iteration + 1] = this_ll
    print('')
    return loglikelihoods, nodes, n_eval


def design_average(problem, fields, n_iterations, nodes=None):
    print('design_average      ', end='')
    if nodes is None:
        nodes = Nodes()
    n_eval = np.arange(n_iterations+1) + nodes.idx.size

    this_fields = fields.condition_to(nodes)
    loglikelihoods = np.zeros((fields.n_sample, n_iterations+1))
    loglikelihoods[:, 0] = this_fields.estimate_loglikelihood(problem.data)

    for i_iteration in range(n_iterations):
        print('.', end='')
        new_index = find_optimal_node_average(nodes, this_fields, problem.data)
        y = problem.evaluate_model(new_index)
        nodes.append(new_index, y)
        this_fields = fields.condition_to(nodes)

        this_ll = this_fields.estimate_loglikelihood(problem.data)
        loglikelihoods[:, i_iteration + 1] = this_ll
    print('')
    return loglikelihoods, nodes, n_eval


def find_optimal_node_average(nodes, fields, data):
    index_list = all_indices_except_visited_ones(fields.n_sample, nodes)
    criterion = compute_average_criterion(index_list, fields, data)
    new_index = np.argmax(criterion)
    return new_index


def compute_average_criterion(index_list, fields, data):
    criterion = np.full((fields.n_sample, fields.n_fields), np.nan)
    for i, subfield in enumerate(fields.subfields):
        this_criterion = compute_criterion_linear(index_list, subfield, data)
        criterion[:, i] = this_criterion
    return criterion @ fields.weights


def design_sampled(problem, field, n_iterations, nodes=None, use_dimension_trick=False):
    print('design_sampled      ', end='')
    if nodes is None:
        nodes = Nodes()
    n_eval = np.arange(n_iterations+1) + nodes.idx.size

    this_field = field.condition_to(nodes)
    loglikelihoods = np.zeros((field.n_sample, n_iterations+1))
    loglikelihoods[:, 0] = this_field.estimate_loglikelihood(problem.data)

    for i_iteration in range(n_iterations):
        print('.', end='')
        new_index = find_optimal_node_sampled(nodes, this_field, problem.data)
        y = problem.evaluate_model(new_index)
        nodes.append(new_index, y)
        this_field = field.condition_to(nodes)

        this_ll = this_field.estimate_loglikelihood(problem.data)
        loglikelihoods[:, i_iteration + 1] = this_ll
    print('')
    return loglikelihoods, nodes, n_eval


def find_optimal_node_sampled(nodes, field, data, use_dimension_trick=False):
    index_list = all_indices_except_visited_ones(field.n_sample, nodes)
    if use_dimension_trick:
        criterion = approximate_criterion_sampled_trick(
            index_list, field, data)
    else:
        criterion = approximate_criterion_sampled(index_list, field, data)
    new_index = np.argmax(criterion)
    return new_index


def approximate_criterion_sampled_trick(index_list, field, data, n_subdivision=51):
    # "Dimension trick" means that I can get the same accuracy as a n-dimensional
    # grid by computing n 1-dimensional integrals.
    current_likelihood = field.estimate_likelihood(data)
    criterion = np.full(field.n_sample, -np.inf)

    for this_idx in index_list:
        y_list = field.y_list(this_idx, n_subdivision)
        likelihoods = []
        for this_y in y_list:
            cond_field = field.condition_to(Nodes(this_idx, this_y))

            this_ls = cond_field.estimate_componentwise_likelihood(data)
            likelihoods.append(this_ls)
        likelihoods = np.array(likelihoods)
        c_total = (likelihoods ** 2).mean(axis=0)
        c_var = np.prod(c_total, axis=1) - current_likelihood ** 2
        c_var = c_var.clip(min=0)
        criterion[this_idx] = np.sum(c_var)/field.n_sample
    return criterion

# This version does not use any tricks in case of (n_output > 1), therefore
# it will probably work on FieldCollections as well.
# If possible, I want to replace this by a version that does some sort of
# Dimensiontrick (see approximate_criterion_sampled), but I need to figure
# out, if this is possible


def approximate_criterion_sampled(index_list, field, data, n_subdivision=51):
    criterion = np.full(field.n_sample, -np.inf)
    for this_idx in index_list:
        #y_list = field.draw_realization_at_index(this_idx, n_subdivision)
        y_list, weights = field.quadrature_at_index(this_idx, n_subdivision)
        #weights = np.ones(y_list.shape[0])
        likelihoods = []
        for this_y in y_list:
            this_node = Nodes(this_idx, this_y)
            cond_field = field.condition_to(this_node)
            this_l = cond_field.estimate_likelihood(data)
            likelihoods.append(this_l)
        likelihoods = np.array(likelihoods)

        mean_likelihood = np.average(likelihoods, axis=0, weights=weights)[
            np.newaxis, :]
        var_likelihood = np.average(
            (likelihoods - mean_likelihood) ** 2, axis=0, weights=weights)
        criterion[this_idx] = var_likelihood.mean()
        # these three lines to the same as the following one, but with weights:
        #criterion[this_idx] = likelihoods.var(axis=0).mean()
    return criterion


def design_hybrid(problem, field, n_iterations, nodes=None):
    print('design_hybrid       ', end='')
    # same as design sampled, but if 95% of weights are concentrated on one
    # subfield, then discard all other subfields and use linearized criterion
    if nodes is None:
        nodes = Nodes()
    n_eval = np.arange(n_iterations+1) + nodes.idx.size

    this_field = field.condition_to(nodes)
    loglikelihoods = np.zeros((field.n_sample, n_iterations+1))
    loglikelihoods[:, 0] = this_field.estimate_loglikelihood(problem.data)

    for i_iteration in range(n_iterations):
        print('.', end='')

        if this_field.is_almost_gpe():
            map_field = this_field.get_map_field()
            new_index = find_optimal_node_linear(
                nodes, map_field, problem.data)
        else:
            new_index = find_optimal_node_sampled(
                nodes, this_field, problem.data)
        y = problem.evaluate_model(new_index)
        nodes.append(new_index, y)
        this_field = field.condition_to(nodes)

        this_ll = this_field.estimate_loglikelihood(problem.data)
        loglikelihoods[:, i_iteration + 1] = this_ll
    print('')
    return loglikelihoods, nodes, n_eval


def design_random(problem, field, n_iterations, nodes=None):
    print('design_random       ', end='')
    if nodes is None:
        nodes = Nodes()
    n_eval = np.arange(n_iterations+1) + nodes.idx.size

    grid = problem.grid
    n_sample = grid.shape[0]
    
    this_field = field.condition_to(nodes, grid)
    loglikelihoods = np.zeros((n_sample, n_iterations+1))
    loglikelihoods[:, 0] = this_field.estimate_loglikelihood(grid, problem.data)

    for i_iteration in range(n_iterations):
        print('.', end='')
        # choose random point
        new_index = pick_random_node_without_duplicates(n_sample, nodes)

        y = problem.evaluate_model(new_index)
        nodes.append(new_index, y)

        this_field = field.condition_to(nodes, grid)
        this_ll = this_field.estimate_loglikelihood(grid, problem.data)
        loglikelihoods[:, i_iteration + 1] = this_ll
    print('')
    return loglikelihoods, nodes, n_eval


def pick_random_node_without_duplicates(n_sample, nodes):
    index_list = all_indices_except_visited_ones(n_sample, nodes)
    return np.random.choice(index_list)


def design_min_variance(problem, field, n_iterations, nodes=None):
    print('design_min_variance ', end='')
    if nodes is None:
        nodes = Nodes()
    n_eval = np.arange(n_iterations+1) + nodes.idx.size

    this_field = field.condition_to(nodes)
    loglikelihoods = np.zeros((field.n_sample, n_iterations+1))
    loglikelihoods[:, 0] = this_field.estimate_loglikelihood(problem.data)

    for i_iteration in range(n_iterations):
        print('.', end='')
        # choose random point
        new_index = find_optimal_node_min_variance(
            nodes, this_field, problem.data)

        y = problem.evaluate_model(new_index)
        nodes.append(new_index, y)

        this_field = field.condition_to(nodes)
        this_ll = this_field.estimate_loglikelihood(problem.data)
        loglikelihoods[:, i_iteration + 1] = this_ll
    print('')
    return loglikelihoods, nodes, n_eval


def find_optimal_node_min_variance(nodes, field, data):
    index_list = all_indices_except_visited_ones(field.n_sample, nodes)
    criterion = compute_criterion_min_variance(index_list, field, data)
    new_index = np.argmax(criterion)
    return new_index


def compute_criterion_min_variance(index_list, field, data):
    criterion = np.full(field.n_sample, -np.inf)
    criterion[index_list] = 0
    c_is_2d = (field.c.ndim == 2)
    if c_is_2d:
        c = field.c
        criterion[index_list] = [c[i, i] * c[i, :]@c[i, :] for i in index_list]
        pass
    else:
        for i_output in range(field.n_output):
            c = field.c[:, :, i_output]
            new_term = [c[i, i] * c[i, :]@c[i, :] for i in index_list]
            criterion[index_list] += new_term
    #    for idx in index_list:
#        q = field.c[idx,:]
#        Q = field.c[idx,idx]
#        criterion[idx] = Q* q@q
    #criterion_at_index_list = [field.c[idx,idx]* (field.c[idx,:]@field.c[idx,:]) for idx in index_list]
    #criterion[index_list] = criterion_at_index_list
    return criterion
