from GeneticAlgorithm import GeneticAlgorithm
from DifferentialEvolution import DifferentialEvolution
from SMPSO.Particle import Particle
from SMPSO.SMPSOAlgorithm import SMPSOAlgorithm
import numpy as np
import random
from math import *


def decode_float_values(bounds: list[list[float]], number_variables: int, genome: list[int])-> list[float]:
    decoded= list()
    n_bits= len(genome) // number_variables
    largest= 2**n_bits
    for i in range(number_variables):
        start, end= i * n_bits, (i * n_bits) + n_bits
        substring= genome[start:end]
        chars= ''.join([str(s) for s in substring])
        integer= int(chars, 2)
        value= bounds[i][0] + ((integer / largest) * (bounds[i][1] - bounds[i][0]))
        decoded.append(value)
    return decoded

def decoded_particle(particle, bounds, number_variables):
    decoded= list()
    n_bits= len(particle) // number_variables
    largest= 2**n_bits
    for i in range(number_variables):
        value= bounds[i][0] + ((particle[i] / largest) * (bounds[i][1] - bounds[i][0]))
        decoded.append(value)
    return decoded

def ackley_function(individual: list[int])-> float:
    a= 20
    b= 0.2
    c= 2 * np.pi
    number_variables= 2
    bounds= [[-5.0, 5.0], [-5.0, 5.0]]
    values= decode_float_values(bounds, number_variables, individual)
    f = -a * exp(-b * sqrt(1.0/number_variables) * sum([xi**2 for xi in values])) \
            - exp (1.0/number_variables * sum([cos(c*xi) for xi in values])) + a + e
    #f= -a * exp(-b * sqrt(1.0 / number_variables * sum([x**2 for x in values]))) - exp(1.0 / number_variables * sum([cos(c * x) for x in values])) + a + exp(1)
    return f

def ackley(particle):
    a= 20
    b= 0.2
    c= 2 * np.pi
    number_variables= 2
    bounds= [[-5.0, 5.0], [-5.0, 5.0]]
    values= decoded_particle(particle, bounds, number_variables)
    f = -a * exp(-b * sqrt(1.0/number_variables) * sum([xi**2 for xi in values])) \
            - exp (1.0/number_variables * sum([cos(c*xi) for xi in values])) + a + e
    return f



def evaluate(individual):
        return sum(individual)

if __name__ == "__main__":
    random.seed(100)
    max_fitness= 0
    best_individual= []
    population= np.array([[random.randint(0,1) for i in range(20)] for j in range(100)])
    for i in range(population.shape[0]):
        if ackley_function(population[i]) > max_fitness:
            max_fitness= ackley_function(population[i])
            best_individual= population[i]
    print("-->Avant Differential Evolution / Genetic Algorithm: ")
    print("Meilleur Individu= ",best_individual)
    print("Fitness Value associée= ", max_fitness)
    print("----------------------------------------------------")
    
    population= np.array([[random.randint(0,1) for i in range(20)] for j in range(100)])
    genetic_algorithm= GeneticAlgorithm(population= population, evaluate= ackley_function)
    best_individual, best_fitness= genetic_algorithm.run_algorithm(max_nfe= 2000)
    print("-->Après Genetic Algorithm: ")
    print("Meilleur Individu= ",best_individual)
    print("Fitness Value associée= ", best_fitness)
    print("----------------------------------------------------")
    
    #population= np.random.rand(100, 20)
    
    differential_evolution= DifferentialEvolution(population= population, evaluate= ackley_function)
    best_individual, best_fitness= differential_evolution.run_algorithm(max_nfe= 2000)
    print("-->Après Differential Evolution: ")
    print("Meilleur Individu= ",best_individual)
    print("Fitness Value associée= ", best_fitness)
    print("----------------------------------------------------")

    swarm= np.array([Particle(evaluate= ackley, bounds= np.array([[-5.0, 5.0], [-5.0, 5.0]]), seed= 100) for i in range(100)])
    smpso_algorithm= SMPSOAlgorithm(swarm= swarm, evaluate= ackley)
    best_particle= smpso_algorithm.run_algorithm(max_iter= 10)
    print("-->Après SMPSO Algorithm: ")
    print("Meilleur Individu= ",best_particle)
    print("----------------------------------------------------")

