import random
import numpy as np
from SMPSO.Particle import Particle

class SMPSOAlgorithm():
    def __init__(self, swarm, evaluate):
        self.swarm = swarm
        self.evaluate= evaluate
        self.pb= -100000
        for particle in self.swarm:
            if particle.pb_fitness > self.pb:
                self.pb= particle.pb_fitness
                self.best_particle= particle
        

    def run_algorithm(self, max_iter= 100):
        iteration= 0
        while iteration < max_iter:
            for particle in self.swarm:
                particle.update_state(self.best_particle.position)
                if particle.pb_fitness > self.pb:
                    self.pb= particle.pb_fitness
                    self.best_particle= particle
            iteration+= 1
        return self.best_particle
