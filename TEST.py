from platypus import *
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import optproblems.cec2005
import seaborn as sns
from platypus import ExperimentJob

from ObserverTasks import *
from PerfIndicators import *
from utils import *

algorithms= [(SMPSO, dict(), "SMPSO"), (GDE3, dict(), "DE"), (NSGAII, dict(variator= GAOperator(SBX(), CompoundMutation())), "NSGAII")]
for cec_function in optproblems.cec2005.CEC2005(2):
            problem = Problem(2, cec_function.num_objectives, function=interceptedFunction(cec_function))
            problem.CECProblem = cec_function
            problem.types[:] = Real(-50,50) if cec_function.min_bounds is None else Real(cec_function.min_bounds[0], cec_function.max_bounds[0])
            problem.directions = [Problem.MAXIMIZE if cec_function.do_maximize else Problem.MINIMIZE]
            # a couple (problem_instance,problem_name) Mandatory because all functions are instance of Problem class
            name = type(cec_function).__name__ + '_' + str(2) + 'D'
            label= ""
            if problem.directions[0] == Problem.MAXIMIZE:
                label = "Maximiser"
            else:
                label = "Minimiser"
            result1= experiment(algorithms=algorithms, problems=[(problem, name)], nfe=100, seeds=1, display_stats=True)
            result2= experiment(algorithms=algorithms, problems=[(problem, name)], nfe=100, seeds=1, display_stats=True)
            print("FINI")
