#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module mini_functions

Contains some helpful utility-functions
    1) compute_errors

"""
import numpy as np


def kldiv(ll_true, ll_approx): 
    ll_true = ll_true - np.max(ll_true)
    ll_approx = ll_approx - np.max(ll_approx)
    
    l_true = np.exp(ll_true)
    l_approx = np.exp(ll_approx)
    ll_true = ll_true - np.log(sum(l_true))
    ll_approx = ll_approx - np.log(sum(l_approx))
    
    kldiv = np.dot(np.exp(ll_true), (ll_true-ll_approx))
    return kldiv


def compute_errors(ll_true, ll_approx):
    n_iterations = ll_approx.shape[1]
    errors = np.zeros((n_iterations))
    for i in range(n_iterations):
        errors[i] = kldiv(ll_true, ll_approx[:, i])
    return errors
