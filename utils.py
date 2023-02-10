import optproblems.cec2005
import numpy as np
from platypus import *

class interceptedFunction(object):
    """ Normalize returned evaluation types in CEC 2005 functions"""
    def __init__(self, initial_function):
        self.__initFunc = initial_function
    def __call__(self,variables):
        objs = self.__initFunc(variables)
        if isinstance(objs, np.floating):
            objs=[objs]
        return objs