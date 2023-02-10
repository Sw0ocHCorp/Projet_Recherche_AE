from platypus import *
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import optproblems.cec2005

from ObserverTasks import *
from PerfIndicators import *
from utils import *

def ackley(individual): 
    a = 20 
    b = 0.2 
    c = 2*pi 
    fitness = -a * exp(-b * sqrt(1.0/len(individual)) * sum([xi**2 for xi in individual])) - \
                   exp (1.0/len(individual) * sum([cos(c*xi) for xi in individual])) + a + e 
    return fitness

def sixhump(individual):
    return ((4 - 2.1*individual[0]**2 + individual[0]**4 / 3.) * individual[0]**2 + individual[0] * individual[1]
            + (-4 + 4*individual[1]**2) * individual[1] **2)

def kim(individual):
     return (sin(individual[0]) + cos(individual[1]) + 0.016*(individual[0]-5)**2 + 0.008*(individual[1] - 5)**2)

if __name__ == "__main__":
    """    
    #==> Convergence des Algorithmes sur une Fonction de Benchmarks <==#
    # --> Tâches Observers
    #csv_writer= CSVWriter()
    plot_stat_task= PlotStatistics(isClassic= False)
    plot_search_task= PlotSearchSpace(benchmark_function= kim, isClassic= False)
    plot_stat_task2= PlotStatistics(isClassic= True)
    plot_search_task2= PlotSearchSpace(benchmark_function= kim, isClassic= True)
    
    #observers= np.array([csv_writer, plot_search_task, plot_stat_task], dtype= object)
    observers= np.array([plot_stat_task], dtype= object)
    pattern_observer= PatternObservers(tasks= observers)
    benchmark_problem= Problem(2, 1, function= kim)
    benchmark_problem.types[:]= Real(-5, 5)
    benchmark_problem.directions= [Problem.MINIMIZE]
    # ALGORITHMES à TESTER
    #   --> Particle Swarm Optimization 
    #csv_writer.reset_csv(filename= "pso.csv")
    plot_search_task.set_title(title= "Particle Swarm Optimization")
    smpso_algorithm= SMPSO(problem= benchmark_problem, swarm_size= 100, leader= 5)
    smpso_algorithm.run(2000)
    #observers= np.array([csv_writer, plot_search_task2, plot_stat_task2], dtype= object)
    observers= np.array([plot_stat_task2], dtype= object)
    pattern_observer.tasks= observers
    print("--> Résultat de l'algorithme SMPSO: Particle Swarm Optimization")
    for s in smpso_algorithm.result:
        print(s.objectives)
    print("--------------------------------------------")
    #   --> Differential Evolution
    #csv_writer.reset_csv(filename= "de.csv")
    plot_search_task2.set_title(title= "Differential Evolution")
    de_algorithm= GDE3(problem= benchmark_problem, population_size= 100)
    de_algorithm.run(2000)
    print("--> Résultat de l'algorithme GDE3: Différential Evolution")
    for s in de_algorithm.result:
        print(s.objectives)
    print("--------------------------------------------")
    #   --> Genetic Algorithm
    #csv_writer.reset_csv(filename= "ga.csv")
    plot_search_task2.set_title(title= "Genetic Algorithm")
    ga_algorithm= GeneticAlgorithm(problem= benchmark_problem, population_size= 100)
    ga_algorithm.run(2000)
    print("--> Résultat de l'algorithme GeneticAlgorithm Classique")
    for s in ga_algorithm.result:
        print(s.objectives)
    print("--------------------------------------------")
    #csv_writer.reset_csv(filename= "nsgaii.csv")
    plot_search_task2.set_title(title= "NSGAII Algorithm")
    nsga_algorithm= NSGAII(problem= benchmark_problem, population_size= 100)
    nsga_algorithm.run(2000)
    print("--> Résultat de l'algorithme NSGAII: Genetic Algorithm")
    for s in ga_algorithm.result:
        print(s.objectives)
    print("--------------------------------------------")
    # --> Plot Statistics
    algorithm_list= [ga_algorithm, nsga_algorithm, smpso_algorithm, de_algorithm]
    plot_stat_task.display_plot_stat(algorithm_list= algorithm_list)
    """
"""
    #==> Evolution de la fitness value en fonction du nombre d'itérations <==#
    benchmark_problem= Problem(2, 1, function= kim)
    benchmark_problem.types[:]= Real(-5, 5)
    benchmark_problem.directions= [Problem.MINIMIZE]
    resultNfe, resultMin=[], [] # empty results
    algo_names= ["SMPSO", "GDE3", "GA", "NSGAII"]
    smpso_algorithm= SMPSO(problem= benchmark_problem, swarm_size= 100, leader= 5)
    de_algorithm= GDE3(problem= benchmark_problem, population_size= 100)
    ga_algorithm= GeneticAlgorithm(problem= benchmark_problem, population_size= 100)
    nsga_algorithm= NSGAII(problem= benchmark_problem, population_size= 100)
    algorithm_list= [smpso_algorithm, de_algorithm, ga_algorithm, nsga_algorithm]
    plot_stat1= PlotStatistics(isClassic= False)    #SMPSO
    plot_stat2= PlotStatistics(isClassic= True)     #Le Reste(DE, GA, NSGAII)
    algo_list= np.empty((0, 1), dtype= object)
    for algorithm in algorithm_list:
        algo= np.array([algorithm])
        algo_list= np.vstack([algo_list, algo]) # add algorithm to list (for legend of plot at the end of
        for seed in range(10): # execute same algorithm several times
            random.seed(seed) # modify current seed
            # TO BE COMPLETED
            if isinstance(algorithm, SMPSO):
                algorithm.run(200, callback= plot_stat1.do_task)
            else:
                algorithm.run(200, callback= plot_stat2.do_task) 
            # create a pandas serie for Nfe & Min fitness
            resultNfe.append(pd.Series(algorithm.statistics['nfe']))
            resultMin.append(pd.Series(algorithm.statistics['min']))
            # execution may be long, so print where we are
            # mean of all executions for same algorithm using pandas
            X = pd.concat(resultNfe,axis=1).mean(axis=1).tolist()
            Y = pd.concat(resultMin,axis=1).mean(axis=1).tolist()
            # TO BE COMPLETED
    i= 0
    for algo in algorithm_list:
        plt.plot(algo.statistics['nfe'], algo.statistics['avg'], label= algo_names[i])
        i+= 1
    plt.title("Comparaison Genetic Algorithm, NSGAII, SMPSO, Differential Evolution")
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Average min fitness')
    plt.legend()
    plt.show()
"""    

#==> Etude Statistique sur Fonctions CEC2005<== #
plot_stat= PlotStatistics()
nexec= 20
nfe= 1000
#dims= [2,10,30,50]
dims= [10]
problem= None
Xovers=SBX()
Mutations=PM()
problems= []
types_problems= dict()
results = OrderedDict()
for dim in dims:
    for cec_function in optproblems.cec2005.CEC2005(dim):
        problem = Problem(dim, cec_function.num_objectives, function=interceptedFunction(cec_function))
        problem.CECProblem = cec_function
        problem.types[:] = Real(-50,50) if cec_function.min_bounds is None else Real(cec_function.min_bounds[0], cec_function.max_bounds[0])
        problem.directions = [Problem.MAXIMIZE if cec_function.do_maximize else Problem.MINIMIZE]
        # a couple (problem_instance,problem_name) Mandatory because all functions are instance of Problem class
        name = type(cec_function).__name__ + '_' + str(dim) + 'D'
        label= ""
        if problem.directions[0] == Problem.MAXIMIZE:
            label = "Maximiser"
        else:
            label = "Minimiser"
        types_problems[name]= label
        problems.append((problem, name))
    a1= SMPSO(problem= problem, swarm_size= 100, leader= 5)
    a2= GDE3(problem= problem, population_size= 100)
    a3= GeneticAlgorithm(problem= problem, population_size= 100)
    a4= NSGAII(problem= problem, population_size= 100)
    algorithms= [(GeneticAlgorithm, dict(), "GA"), (SMPSO, dict(), "SMPSO"), (GDE3, dict(), "DE"), (NSGAII, dict(), "NSGAII")]
    results = results | experiment(algorithms=algorithms, problems=problems, nfe=nfe, seeds=nexec,
                                    display_stats=True)
    indicators=[bestFitness()]
    plot_stat.plot_CEC2005_stat(results, indicators, bench_function= 8, types_problems= types_problems)

        

    
