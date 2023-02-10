import numpy as np
from platypus import *

class bestFitness(Indicator):
    """find best fitness in population"""
    def __init__(self):
        super(bestFitness, self).__init__()
    def calculate(self, set):
        feasible = [s for s in set if s.constraint_violation == 0.0]
        if len(feasible) == 0:
            return 0.0
        elif feasible[0].problem.nobjs != 1:
            raise ValueError("bestFitness indicator can only be used for single-objective problems")
        best = None
        optimum = np.min if feasible[0].problem.directions[0] == Problem.MINIMIZE else np.max
        best = optimum([x.objectives[0] for x in feasible])
        return best