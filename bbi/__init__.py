"""
Module bbi
for Bayesian Bayesian Inverse Problems solving
"""
from .field import GpeMatern, \
                   GpeSquaredExponential
from .field import MixMatern, \
                   MixSquaredExponential
from .field import FieldCollection
from .field import covariance_squared_exponential, \
                   covariance_matern
from .field import DiscreteGpe, DiscreteGpeMatern, DiscreteGpeSquaredExponential
from .design import design_linearized, \
                    design_map, \
                    design_average, \
                    design_sampled, \
                    design_hybrid, \
                    design_random, \
                    design_min_variance, design_old
from .selection import  select_model, select_model_spacefilling
from .mini_classes import Nodes, Data, Problem, SelectionProblem
from .mini_functions import compute_errors, kldiv

del field
#del design
#del selection
del mini_functions
del mini_classes
