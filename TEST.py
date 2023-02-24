from optproblems import *
from math import *

# Modélisation du problème d'optimisation
def kim(individual):
     return (sin(individual[0]) + cos(individual[1]) + 0.016*(individual[0]-5)**2 + 0.008*(individual[1] - 5)**2)

my_problem = Problem(kim)

# Résolution du problème d'optimisation
my_solution = my_problem.

# Obtention de l'optimum global
optimum = my_solution.get_optimum()

print(f"Optimum global : {optimum}")
