import random
import numpy as np

class Particle():
    def __init__(self, evaluate, bounds, seed= 0):
        random.seed(seed)
        self.rnd= random.Random(seed)
        self.evaluate= evaluate
        self.bounds= bounds
        self.dim= bounds.shape[0]
        self.position= np.array([])
        self.velocity= np.array([])
        for bound in bounds:
            self.position= np.append(self.position, ((bound[0] - bound[1])*random.random() + bound[1]))
            self.velocity= np.append(self.velocity, ((bound[0] - bound[1])*random.random() + bound[1]))
        self.pb_position= self.position
        self.pb_fitness= self.evaluate(self.pb_position)
        self.fitness= self.evaluate(self.position)

    def update_state(self, gbest_position, w= 0.5, c1= 1, c2= 1.5):
        r1= random.random()
        r2= random.random()
        for dim in range(self.dim):
            self.velocity[dim]= w*self.velocity[dim] + c1*r1*(self.pb_position[dim] - self.position[dim]) + c2*r2*(gbest_position[dim] - self.position[dim])
            if self.velocity[dim] > self.bounds[dim][1]:
                self.velocity[dim]= self.bounds[dim][1]
            elif self.velocity[dim] < self.bounds[dim][0]:
                self.velocity[dim]= self.bounds[dim][0]
            self.position[dim]= self.position[dim] + self.velocity[dim]
        self.fitness= self.evaluate(self.position)
        if self.fitness > self.pb_fitness:
            self.pb_fitness= self.fitness
            self.pb_position= self.position
        return self.fitness
    
    def __repr__(self) -> str:
        return f"Particle: {self.position} | Fitness: {self.fitness}"
        
        
            
        