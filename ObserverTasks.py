import csv
from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from platypus import *

class PatternObservers():
    def __init__(self, tasks):
        self.tasks= tasks
    
    def do_observers_tasks(self, algorithm):
        for task in self.tasks:
            task.do_task(algorithm)

class CSVWriter():
    def __init__(self):
        self.csv_file= None
    
    def do_task(self, algorithm):
        self.csv_writer.writerow([algorithm.nfe, algorithm.result[0].objectives[0]])
    
    def reset_csv(self, filename= "data.csv"):
        if self.csv_file is not None:
            self.csv_file.close()
        self.filename= filename
        self.delimiter= ";"
        self.csv_file= open(self.filename, 'a', newline='')
        self.csv_writer= csv.writer(self.csv_file, delimiter= self.delimiter)
        self.csv_writer.writerow(["NFE", "Valeur Objective"])

class PlotSearchSpace():
    def __init__(self, benchmark_function, isClassic= True):
        self.isClassic= isClassic
        self.benchmark_function= benchmark_function
        self.title= ""

    def do_task(self, algorithm):   #Pour Genetic Algorithm
        solX= np.array([])
        solY= np.array([])
        solZ= np.array([])
        fig= plt.figure(figsize=(6,6))
        ax= fig.add_subplot(111, projection= "3d")
        x = np.arange(-5, 5, 0.1) # set of float values between
        y = np.arange(-5, 5, 0.1) # -0.5 and 0.5 step 0.1
        X, Y = np.meshgrid(x, y) # dot product between x & y
        Z = [self.benchmark_function([a,b]) for a,b in zip(np.ravel(X),np.ravel(Y))]
        Z = np.array(Z).reshape(X.shape)
        surf = ax.plot_surface(X, Y, Z, alpha=0.3)
        if self.isClassic:
            solX = [x.variables[0] for x in algorithm.population] # extract x values from population
            solY = [x.variables[1] for x in algorithm.population] # extract y values from population
            solZ = [x.objectives[0] for x in algorithm.population] # extract z values from population
        else:
            solX = [x.variables[0] for x in algorithm.particles] # extract x values from population
            solY = [x.variables[1] for x in algorithm.particles] # extract y values from population
            solZ = [x.objectives[0] for x in algorithm.particles] # extract z values from population
        surf = ax.scatter(solX,solY,solZ, color='red') # plot population
        plt.title(self.title)
        plt.show() # show plot"""
    
    def set_title(self, title):
        self.title= title



class PlotStatistics():
    def __init__(self, isClassic= True):
        self.isClassic= isClassic
        self.algo_names= np.array([])

    def do_task(self, algorithm):
        if not hasattr(algorithm, 'statistics'):
            algorithm.statistics= {'nfe': [], 'avg': [], 'min': [], 'max': [], 'std': []}
        algorithm.statistics['nfe'].append(algorithm.nfe)
        if self.isClassic:
            fitness= [s.objectives[0] for s in algorithm.population]
        else:
            fitness= [s.objectives[0] for s in algorithm.particles]
        algorithm.statistics['avg'].append(np.average(fitness))
        if isinstance(algorithm, SMPSO):
            self.algo_names= np.append(self.algo_names, "SMPSO")
        elif isinstance(algorithm, NSGAII):
            self.algo_names= np.append(self.algo_names, "NSGAII")
        elif isinstance(algorithm, GDE3):
            self.algo_names= np.append(self.algo_names, "GDE3")
        elif isinstance(algorithm, GeneticAlgorithm):
            self.algo_names= np.append(self.algo_names, "Genetic Algorithm")
    
    def display_plot_stat(self, algorithm_list):
        for i in range(len(algorithm_list)):
            plt.plot(algorithm_list[i].statistics['nfe'], algorithm_list[i].statistics['avg'], label= self.algo_names[i])
        plt.title("Genetic Algorithm / Differential Evolution / Particle Swarm Optimization")
        plt.xlabel('Number of Function Evaluations')
        plt.ylabel('Average min fitness')
        plt.legend()
        plt.show()
    
    def plot_CEC2005_stat(self, results, indicator, bench_function= 1, types_problems= dict()):
        indicators_result = calculate(results, indicator)
        #print("Best element for Crossover => " + Xover + " and Mutation => " + Mutation + " " + "Benchmark Function " + str(bench_function) + " --> " + str(bench_stats))
        data= dict()
        for key_algo, algo in results.items():
            for key_problem, problem in algo.items():
                for key_indicator, indicator in indicators_result.items():
                    data[(key_algo,key_problem)] = indicators_result[key_algo][key_problem]['bestFitness']
        data= pd.DataFrame(data)
        data.to_csv("data.csv")
        print(data.describe())
        data= data.stack(level= 0).unstack()
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(20, 10))
        data["F1_10D"].plot(ax= ax1)
        ax1.set_title("F1_10D: à " + str(types_problems["F1_10D"]))
        data["F2_10D"].plot(ax= ax2)
        ax2.set_title("F2_10D: à " + str(types_problems["F2_10D"]))
        data["F3_10D"].plot(ax= ax3)
        ax3.set_title("F3_10D: à " + str(types_problems["F3_10D"]))
        data["F4_10D"].plot(ax= ax4)
        ax4.set_title("F4_10D: à " + str(types_problems["F4_10D"]))
        data["F5_10D"].plot(ax= ax5)
        ax5.set_title("F5_10D: à " + str(types_problems["F5_10D"]))
        data["F6_10D"].plot(ax= ax6)
        ax6.set_title("F6_10D: à " + str(types_problems["F6_10D"]))
        data["F7_10D"].plot(ax= ax7)
        ax7.set_title("F7_10D: à " + str(types_problems["F7_10D"]))
        data["F8_10D"].plot(ax= ax8)
        ax8.set_title("F8_10D: à " + str(types_problems["F8_10D"]))
        data["F9_10D"].plot(ax= ax9)
        ax9.set_title("F9_10D: à " + str(types_problems["F9_10D"]))
        fig2, ((ax10, ax11, ax12), (ax13, ax14, ax15), (ax16, ax17, ax18)) = plt.subplots(3, 3, figsize=(20, 10))
        data["F10_10D"].plot(ax= ax10)
        ax10.set_title("F10_10D: à " + str(types_problems["F10_10D"]))
        data["F11_10D"].plot(ax= ax11)
        ax11.set_title("F11_10D: à " + str(types_problems["F11_10D"]))
        data["F12_10D"].plot(ax= ax12)
        ax12.set_title("F12_10D: à " + str(types_problems["F12_10D"]))
        data["F13_10D"].plot(ax= ax13)
        ax13.set_title("F13_10D: à " + str(types_problems["F13_10D"]))
        data["F14_10D"].plot(ax= ax14)
        ax14.set_title("F14_10D: à " + str(types_problems["F14_10D"]))
        data["F15_10D"].plot(ax= ax15)
        ax15.set_title("F15_10D: à " + str(types_problems["F15_10D"]))
        data["F16_10D"].plot(ax= ax16)
        ax16.set_title("F16_10D: à " + str(types_problems["F16_10D"]))
        data["F17_10D"].plot(ax= ax17)
        ax17.set_title("F17_10D: à " + str(types_problems["F17_10D"]))
        data["F18_10D"].plot(ax= ax18)
        ax18.set_title("F18_10D: à " + str(types_problems["F18_10D"]))
        plt.show(block= True)