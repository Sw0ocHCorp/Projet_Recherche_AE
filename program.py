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

def plot_search_spc(algorithm):     #Pour Particle Swarm
        solX= np.array([])
        solY= np.array([])
        solZ= np.array([])
        fig= plt.figure(figsize=(6,6))
        ax= fig.add_subplot(111, projection= "3d")
        x = np.arange(-5, 5, 0.1) # set of float values between
        y = np.arange(-5, 5, 0.1) # -0.5 and 0.5 step 0.1
        X, Y = np.meshgrid(x, y) # dot product between x & y
        Z = [kim([a,b]) for a,b in zip(np.ravel(X),np.ravel(Y))]
        Z = np.array(Z).reshape(X.shape)
        surf = ax.plot_surface(X, Y, Z, alpha=0.3)
        solX = [x.variables[0] for x in algorithm.particles] # extract x values from population
        solY = [x.variables[1] for x in algorithm.particles] # extract y values from population
        solZ = [x.objectives[0] for x in algorithm.particles] # extract z values from population
        surf = ax.scatter(solX,solY,solZ, color='red') # plot population
        plt.show() # show plot"""

#==> Optimisation des Algorithmes GA et NSGAII <==#
def opti_ga_nsgaii(nexec= 30, dim= 2):
    plot_stat= PlotStatistics()
    plot_stat.set_isClassic(True)
    algorithm_list= np.empty((0, 3), dtype= object)
    Xovers=[SBX(), UNDX(), PCX(), SPX()] # all possible Xovers
    Mutations=[PM(), UniformMutation(probability= 0.05, perturbation= 0.05), UM(), CompoundMutation()] #all possible Mutations
    fig=plt.figure() # a new figure
    # for all combinations of Xover and Mutation
    #cpt= 0
    sns.reset_orig()  # get default matplotlib styles back
    i= 0
    for Xover,Mutation in [(x,m) for x in Xovers for m in Mutations]:
        #cpt+=1
        myProblem = Problem (dim, 1, function= kim)
        # each variable of the myProblem is a float value in [-5,5]
        myProblem.types[:] = Real(-5,5)
        myProblem.directions[:]=[Problem.MINIMIZE]
        resultNfe, resultMin=[], [] # empty results
        XoverName, MutName=type(Xover).__name__, type(Mutation).__name__
        #algorithm = GeneticAlgorithm(myProblem, variator= GAOperator(Xover, Mutation))
        algorithm = NSGAII(myProblem, variator= GAOperator(Xover, Mutation))
        row= np.array([algorithm, XoverName, MutName])
        algorithm_list= np.vstack([algorithm_list, row]) # add algorithm to list (for legend of plot at the end of
        for seed in range(nexec): # execute same algorithm several times
            random.seed(seed) # modify current seed
            # TO BE COMPLETED
            algorithm.run(10, callback= plot_stat.do_task)
            # create a pandas serie for Nfe & Min fitness
            resultNfe.append(pd.Series(algorithm.statistics['nfe']))
            resultMin.append(pd.Series(algorithm.statistics['min']))
            # execution may be long, so print where we are
            print ('run {0} with {1} {2}'.format(seed,XoverName,MutName))
            # mean of all executions for same algorithm using pandas
            X = pd.concat(resultNfe,axis=1).mean(axis=1).tolist()
            Y = pd.concat(resultMin,axis=1).mean(axis=1).tolist()
            # TO BE COMPLETED
        for row in algorithm_list:
            r = np.round(np.random.rand(),1)
            g = np.round(np.random.rand(),1)
            b = np.round(np.random.rand(),1)
            plt.plot(row[0].statistics['nfe'], row[0].statistics['avg'], label= row[1] + " " + row[2], color= [r,g,b])
        plt.title("NSGAII XOver & Mutation comparisons on Kim (" + str(myProblem.nvars) + "D) on "+str(nexec)+" executions")
        plt.xlabel('Number of Function Evaluations')
        plt.ylabel('Average min fitness')
        plt.legend(loc= "upper right")
        plt.show()

def test_convergence():
    # --> Tâches Observers
    #csv_writer= CSVWriter()
    plot_stat_task= PlotStatistics()
    plot_search_task= PlotSearchSpace(benchmark_function= kim, isClassic= False)
    plot_search_task2= PlotSearchSpace(benchmark_function= kim, isClassic= True)
    
    #observers= np.array([csv_writer, plot_search_task, plot_stat_task], dtype= object)
    observers= np.array([plot_stat_task, plot_search_task], dtype= object)
    pattern_observer= PatternObservers()
    pattern_observer.attach_tasks(tasks= observers)
    benchmark_problem= Problem(10, 1, function= kim)
    benchmark_problem.types[:]= Real(-5, 5)
    benchmark_problem.directions= [Problem.MINIMIZE]
    # ALGORITHMES à TESTER
    #   --> Particle Swarm Optimization 
    #csv_writer.reset_csv(filename= "pso.csv")
    #plot_search_task.set_title(title= "Particle Swarm Optimization")
    smpso_algorithm= SMPSO(problem= benchmark_problem, swarm_size= 100, leader= 5)
    plot_stat_task.set_isClassic(isClassic= False)
    smpso_algorithm.run(2000, callback= pattern_observer.do_task)
    #observers= np.array([csv_writer, plot_search_task2, plot_stat_task2], dtype= object)
    print("--> Résultat de l'algorithme SMPSO: Particle Swarm Optimization")
    for s in smpso_algorithm.result:
        print(s.objectives)
    print("--------------------------------------------")
    #   --> Differential Evolution
    #csv_writer.reset_csv(filename= "de.csv")
    #plot_search_task.set_title(title= "Differential Evolution")
    de_algorithm= GDE3(problem= benchmark_problem, population_size= 100)
    plot_stat_task.set_isClassic(isClassic= True)
    de_algorithm.run(2000, callback= plot_stat_task.do_task)
    print("--> Résultat de l'algorithme GDE3: Différential Evolution")
    for s in de_algorithm.result:
        print(s.objectives)
    print("--------------------------------------------")
    #   --> Genetic Algorithm
    #csv_writer.reset_csv(filename= "ga.csv")
    #plot_search_task2.set_title(title= "Genetic Algorithm")
    ga_algorithm= GeneticAlgorithm(problem= benchmark_problem, variator= GAOperator(SBX(), CompoundMutation()), population_size= 100)
    ga_algorithm.run(2000, callback= plot_stat_task.do_task)
    print("--> Résultat de l'algorithme GeneticAlgorithm Classique")
    for s in ga_algorithm.result:
        print(s.objectives)
    print("--------------------------------------------")
    #csv_writer.reset_csv(filename= "nsgaii.csv")
    plot_search_task2.set_title(title= "NSGAII Algorithm")
    nsga_algorithm= NSGAII(problem= benchmark_problem, population_size= 100)
    nsga_algorithm.run(2000, callback= plot_stat_task.do_task)
    print("--> Résultat de l'algorithme NSGAII: Genetic Algorithm")
    for s in ga_algorithm.result:
        print(s.objectives)
    print("--------------------------------------------")
    # --> Plot Statistics
    algorithm_list= [smpso_algorithm, de_algorithm, ga_algorithm, nsga_algorithm]
    plot_stat_task.display_plot_stat(algorithm_list= algorithm_list)

#==> Evolution de la fitness value en fonction du nombre d'itérations <==#
def test_fitness_value_algorithms():
    benchmark_problem= Problem(50, 1, function= kim)
    benchmark_problem.types[:]= Real(-5, 5)
    benchmark_problem.directions= [Problem.MINIMIZE]
    resultNfe, resultMin=[], [] # empty results
    #algo_names= ["SMPSO", "GDE3", "GA", "NSGAII"]
    #algo_names= ["GA", "NSGAII"]
    algo_names= ["SMPSO", "GDE3", "NSGAII"]
    smpso_algorithm= SMPSO(problem= benchmark_problem, swarm_size= 100, leader= 5)
    de_algorithm= GDE3(problem= benchmark_problem, population_size= 100)
    #ga_algorithm= GeneticAlgorithm(problem= benchmark_problem, variator= GAOperator(SBX(),UniformMutation(probability= 0.05, perturbation= 0.05)), population_size= 100)
    nsga_algorithm= NSGAII(problem= benchmark_problem, variator= GAOperator(SBX(), CompoundMutation()), population_size= 100)
    #algorithm_list= [smpso_algorithm, de_algorithm, ga_algorithm, nsga_algorithm]
    #algorithm_list= [ga_algorithm, nsga_algorithm]
    algorithm_list= [smpso_algorithm, de_algorithm, nsga_algorithm]
    plot_stat= PlotStatistics()                    #isClassic= False -->   SMPSO
                                                    #isClassic= True  -->   Le Reste(DE, GA, NSGAII)
    algo_list= np.empty((0, 1), dtype= object)
    for algorithm in algorithm_list:
        algo= np.array([algorithm])
        algo_list= np.vstack([algo_list, algo]) # add algorithm to list (for legend of plot at the end of
        for seed in range(10): # execute same algorithm several times
            random.seed(seed) # modify current seed
            # TO BE COMPLETED
            if isinstance(algorithm, SMPSO):
                plot_stat.set_isClassic(isClassic= False)
                algorithm.run(200, callback= plot_stat.do_task)
            else:
                plot_stat.set_isClassic(isClassic= True)
                algorithm.run(200, callback= plot_stat.do_task) 
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
    plt.title("NSGAII, SMPSO, Differential Evolution("+ str(benchmark_problem.nvars) + "D)")
    #plt.title("Comparaison Genetic Algorithm, NSGAII ("+ str(benchmark_problem.nvars) + "D)")
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Average min fitness')
    plt.legend()
    plt.show()

#==> Etude Statistique sur Fonctions CEC2005<== #
def etude_stat_algorithms(nexec= 10, nfe= 1000, dims= [2]):
    plot_stat= PlotStatistics()
    problem= None
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
        algorithms= [(SMPSO, dict(), "SMPSO"), (GDE3, dict(), "DE"), (NSGAII, dict(variator= GAOperator(SBX(), CompoundMutation())), "NSGAII")]
        #Enlever les Argument plot_stat pour les fonction experiment | evaluate_job_generator | ExperimentJob
        """results = results | experiment(algorithms=algorithms, problems=problems, nfe=nfe, seeds=nexec,
                                        display_stats=True)"""
        indicators=[bestFitness()]
        plot_stat.plot_CEC2005_stat(results, indicators, dim= dims[0], types_problems= types_problems)

def etude_stat_convergence(dims= [2], nfe= 1000, pop_size= 100, swarm_size= 100, leader= 5):
    plot_stat= PlotStatistics()
    results = OrderedDict()
    statistics= {"NSGAII":[], "SMPSO":[], "DE":[]}
    cpt=0
    for dim in dims:
        epoch= ceil(nfe / pop_size)
        for cec_function in optproblems.cec2005.CEC2005(dim):
            if cpt >= 4:
                break
            isInit= False
            problem = Problem(dim, cec_function.num_objectives, function=interceptedFunction(cec_function))
            problem.CECProblem = cec_function
            problem.types[:] = Real(-50,50) if cec_function.min_bounds is None else Real(cec_function.min_bounds[0], cec_function.max_bounds[0])
            problem.directions = [Problem.MAXIMIZE if cec_function.do_maximize else Problem.MINIMIZE]
            # a couple (problem_instance,problem_name) Mandatory because all functions are instance of Problem class
            name = type(cec_function).__name__ + '_' + str(dim) + 'D'
            algorithms= [(SMPSO, dict(), "SMPSO"), (GDE3, dict(), "DE"), (NSGAII, dict(variator= GAOperator(SBX(), CompoundMutation())), "NSGAII")]
            
            plot_stat.set_fnct_name(name)
            #plot_stat.set_problem(problem)
            result= experiment(algorithms=algorithms, problems=[(problem, name)], nfe=pop_size, seeds=epoch, display_stats=True, plot_stat= plot_stat)
            """for e in range(4):
                plot_stat.set_isClassic(isClassic= True)
                if isInit == False:
                    nsga_algorithm.run(1, callback= plot_stat.save_conv_stat)
                    de_algorithm.run(1, callback= plot_stat.save_conv_stat)
                    results= experiment(algorithms=algorithms, problems=[(problem, name)], nfe=nfe, seeds=1, display_stats=True)
                else: 
                    nsga_algorithm.run(ceil(epoch/3), callback= plot_stat.save_conv_stat)
                    de_algorithm.run(ceil(epoch/3), callback= plot_stat.save_conv_stat)
                    results= experiment(algorithms=algorithms, problems=[(problem, name)], nfe=nfe, seeds=ceil(epoch/3), display_stats=True)
                plot_stat.set_isClassic(isClassic= False)
                if isInit == False:
                    smpso_algorithm.run(1, callback= plot_stat.save_conv_stat)
                    isInit= True
                else:
                    smpso_algorithm.run(ceil(epoch/3), callback= plot_stat.save_conv_stat)
                nsga_algorithm.set_initial_population(solutions)
            statistics["NSGAII"].append(plot_stat.get_stat(nsga_algorithm))
            statistics["DE"].append(plot_stat.get_stat(de_algorithm))
            statistics["SMPSO"].append(plot_stat.get_stat(smpso_algorithm))"""
            cpt+=1
    plot_stat.plot_bxplt_stat_cec(fnct_names=["F1_2D", "F3_2D"])
    return statistics

#==> CUSTOM EXPERIMENT <==#



if __name__ == "__main__":
    #etude_stat_convergence()
    plot_stat= PlotStatistics()
    pattern_observer= PatternObservers()
    benchmark_problem= Problem(2, 1, function= kim)
    benchmark_problem.types[:]= Real(-50, 50)
    benchmark_problem.directions= [Problem.MINIMIZE]
    plot_search_task= PlotSearchSpace(benchmark_function= kim, isClassic= False)
    smpso_algorithm= SMPSO(problem= benchmark_problem, swarm_size= 100, leader= 5)
    de_algorithm= GDE3(problem= benchmark_problem, population_size= 100)
    nsga_algorithm= NSGAII(problem= benchmark_problem, variator= GAOperator(SBX(), CompoundMutation()), population_size= 100)
    smpso_algorithm.run(500, callback= plot_stat.save_stat)
    de_algorithm.run(500, callback= plot_stat.save_stat)
    nsga_algorithm.run(500, callback= plot_stat.save_stat)
    plot_stat.plot_bxplt_stat(algorithm= smpso_algorithm)
    plot_stat.plot_bxplt_stat(algorithm= de_algorithm)
    plot_stat.plot_bxplt_stat(algorithm= nsga_algorithm)
    #etude_stat_convergence()
    #test_convergence()
    
