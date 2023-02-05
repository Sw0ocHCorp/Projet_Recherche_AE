import numpy as np
import random
class DifferentialEvolution:
    def __init__(self, population, evaluate):
        self.population = population
        self.l= population.shape[1]
        self.evaluate= evaluate
        self.fitness = np.array([self.evaluate(individual) for individual in self.population])
    
    def mutation(self, population, target_individual, F= 0.5):
        pop= population.copy()
        pop= np.delete(pop, target_individual, axis= 0)
        selected_individuals=random.sample(list(pop), 3)
        mutated_individual= np.round_(np.clip(selected_individuals[0] + F*(selected_individuals[1] - selected_individuals[2]), 0, 1))
        mutated_individual= mutated_individual.astype(int)
        return mutated_individual

    def crossover_selection(self, target_individual, mutated_individual, cr= 0.5):
        target_individual= target_individual.astype(int)
        min_len= min([len(target_individual), len(mutated_individual)])
        child= np.array([], dtype= int)
        for i in range(min_len):
            if random.random() < cr:
                child= np.append(child, mutated_individual[i])
            else:
                child= np.append(child, target_individual[i])
        fit_child= self.evaluate(child)
        fit_parent= self.evaluate(target_individual)
        if fit_child > fit_parent:
            return child
        else:
            return target_individual
    
    def run_algorithm(self, max_nfe= 1000):
        nfe= 0
        while nfe < max_nfe:
            for i in range(len(self.population)):
                mutated_individual= self.mutation(self.population, self.population[i])
                self.population[i]= self.crossover_selection(self.population[i], mutated_individual)
                nfe+= 1
        return self.population[np.argmax(self.fitness)], self.evaluate(self.population[np.argmax(self.fitness)])
        
    
