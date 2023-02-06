from platypus import *
from math import *
import numpy as np
import matplotlib.pyplot as plt

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

def plot_search_space(algorithm):   #Pour Genetic Algorithm
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
        solX = [x.variables[0] for x in algorithm.population] # extract x values from population
        solY = [x.variables[1] for x in algorithm.population] # extract y values from population
        solZ = [x.objectives[0] for x in algorithm.population] # extract z values from population
        surf = ax.scatter(solX,solY,solZ, color='red') # plot population
        plt.show() # show plot"""

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

if __name__ == "__main__":
    benchmark_problem= Problem(2, 1, function= kim)
    benchmark_problem.types[:]= Real(-5, 5)
    benchmark_problem.directions= [Problem.MINIMIZE]
    # ALGORITHMES à TESTER
    #   --> Particle Swarm Optimization 
    """smpso_algorithm= SMPSO(problem= benchmark_problem, swarm_size= 100, leader= 5)
    smpso_algorithm.run(2000, callback= plot_search_spc)
    print("--> Résultat de l'algorithme SMPSO: Particle Swarm Optimization")
    for s in smpso_algorithm.result:
        print(s.objectives)
    print("--------------------------------------------")"""
    #   --> Differential Evolution
    """de_algorithm= GDE3(problem= benchmark_problem, population_size= 100)
    de_algorithm.run(2000, callback= plot_search_space)
    print("--> Résultat de l'algorithme GDE3: Différential Evolution")
    for s in de_algorithm.result:
        print(s.objectives)
    print("--------------------------------------------")"""
    #   --> Genetic Algorithm
    """ga_algorithm= GeneticAlgorithm(problem= benchmark_problem, population_size= 100)
    print("--> Résultat de l'algorithme GeneticAlgorithm Classique")
    ga_algorithm.run(2000, callback= plot_search_space)
    for s in ga_algorithm.result:
        print(s.objectives)
    print("--------------------------------------------")"""
    ga_algorithm= NSGAII(problem= benchmark_problem, population_size= 100)
    print("--> Résultat de l'algorithme NSGAII: Genetic Algorithm")
    ga_algorithm.run(2000, callback= plot_search_space)
    for s in ga_algorithm.result:
        print(s.objectives)
    print("--------------------------------------------")

    
