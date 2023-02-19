import csv
from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from platypus import *
from scipy.spatial import ConvexHull
from platypus.indicators import Hypervolume
from matplotlib.gridspec import GridSpec
from platypus.algorithms import NSGAII, SMPSO, GDE3

class PatternObservers():
    def __init__(self):
        pass
    
    def attach_tasks(self, tasks):
        self.tasks= tasks

    def do_observers_tasks(self, algorithm):
        for task in self.tasks:
            task.do_task(algorithm)
    
    def attach_functions(self, target_functions):
        self.target_functions= target_functions
    
    def do_target_functions(self, algorithm):
        for target_function in self.target_functions:
            target_function(algorithm)

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
    
    def print_variation_fitness(self,data, algos_names):
        file= open("variation_fitness.csv", 'a', newline='')
        csv_writer= csv.writer(file, delimiter= ";")
        csv_writer.writerow(algos_names)
        for i in range(data.shape[0]):
            csv_writer.writerow(data[i,:])
        csv_writer.writerow(["Moyenne Variation Fitness"])
        csv_writer.writerow(algos_names)
        csv_writer.writerow(np.mean(data, axis= 0))


        

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
    def __init__(self):
        #self.cpt=0
        self.algo_names= np.array([])
        self.fnct_name= ""
        self.statistics= {"SMPSO": {}, "DE": {}, "NSGAII": {}}
        self.num_study= 0
        self.saved_nfe= []

    def set_problem(self, problem):
        self.problem= problem
    
    def set_fnct_name(self, fnct_name):
        self.fnct_name= fnct_name

    def set_isClassic(self, isClassic):
        self.isClassic= isClassic
    
    def get_stat(self, algorithm):
        return algorithm.statistics

    def setup_new_analysis(self):
        self.num_study+= 1
    
    def reset_analysis(self):
        self.num_study= 0

    def save_stat(self, algorithm):
        hv= Hypervolume(minimum= [0,0], maximum= [1,1])
        fitness= []
        if not hasattr(algorithm, 'statistics'):
            algorithm.statistics= dict()
        if hasattr(algorithm, "population"):
            hv= Hypervolume(reference_set= algorithm.population)
            fitness= [s.objectives[0] for s in algorithm.population]
            algorithm.statistics[str(algorithm.nfe)]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population}
        if hasattr(algorithm, "particles"):
            hv= Hypervolume(reference_set= algorithm.particles)
            fitness= [s.objectives[0] for s in algorithm.particles]
            algorithm.statistics[str(algorithm.nfe)]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.particles), "population": algorithm.particles}
    
    def save_stat_cec(self, algorithm):
        hv= Hypervolume(minimum= [0,0], maximum= [1,1])
        fitness= []
        if isinstance(algorithm, SMPSO):
            hv= Hypervolume(reference_set= algorithm.particles)
            fitness= [s.objectives[0] for s in algorithm.particles]
            if self.fnct_name not in self.statistics["SMPSO"].keys():
                self.statistics["SMPSO"][self.fnct_name]= {}
                if algorithm.nfe not in self.statistics["SMPSO"][self.fnct_name]:
                    self.statistics["SMPSO"][self.fnct_name][algorithm.nfe]= {}
                    if self.num_study not in self.statistics["SMPSO"][self.fnct_name][algorithm.nfe]:
                        self.statistics["SMPSO"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.particles), "particles": algorithm.particles}
            else:
                if algorithm.nfe not in self.statistics["SMPSO"][self.fnct_name]:
                    self.statistics["SMPSO"][self.fnct_name][algorithm.nfe]= {}
                    if self.num_study not in self.statistics["SMPSO"][self.fnct_name][algorithm.nfe]:
                        self.statistics["SMPSO"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.particles), "particles": algorithm.particles}
                else:
                    if self.num_study not in self.statistics["SMPSO"][self.fnct_name][algorithm.nfe]:
                        self.statistics["SMPSO"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.particles), "particles": algorithm.particles}
        elif isinstance(algorithm, NSGAII):
            hv= Hypervolume(reference_set= algorithm.population)
            fitness= [s.objectives[0] for s in algorithm.population]
            if self.fnct_name not in self.statistics["NSGAII"].keys():
                self.statistics["NSGAII"][self.fnct_name]= {}
                if algorithm.nfe not in self.statistics["NSGAII"][self.fnct_name]:
                    if len(self.saved_nfe) < 4:
                        self.saved_nfe.append(algorithm.nfe)
                    self.statistics["NSGAII"][self.fnct_name][algorithm.nfe]= {}
                    if self.num_study not in self.statistics["NSGAII"][self.fnct_name][algorithm.nfe]:
                        self.statistics["NSGAII"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population, "nfe": algorithm.nfe}
            else:
                if len(self.saved_nfe) < 4:
                    if algorithm.nfe not in self.statistics["NSGAII"][self.fnct_name]:
                        self.saved_nfe.append(algorithm.nfe)
                        self.statistics["NSGAII"][self.fnct_name][algorithm.nfe]= {}
                        if self.num_study not in self.statistics["NSGAII"][self.fnct_name][algorithm.nfe]:
                            self.statistics["NSGAII"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population, "nfe": algorithm.nfe}
                    else:
                        if self.num_study not in self.statistics["NSGAII"][self.fnct_name][algorithm.nfe]:
                            self.statistics["NSGAII"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population, "nfe": algorithm.nfe}
                else:
                    min_diff= POSITIVE_INFINITY
                    target_nfe= 0
                    for nfe in self.saved_nfe:
                        if abs(nfe - algorithm.nfe) < min_diff:
                            min_diff= abs(nfe - algorithm.nfe)
                            target_nfe= nfe
                    if target_nfe not in self.statistics["NSGAII"][self.fnct_name]:
                        self.statistics["NSGAII"][self.fnct_name][target_nfe]= {}
                    if self.num_study not in self.statistics["NSGAII"][self.fnct_name][target_nfe]:
                        self.statistics["NSGAII"][self.fnct_name][target_nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population, "nfe": algorithm.nfe}
                    
        elif isinstance(algorithm, GDE3):
            hv= Hypervolume(reference_set= algorithm.population)
            fitness= [s.objectives[0] for s in algorithm.population]
            if self.fnct_name not in self.statistics["DE"].keys():
                self.statistics["DE"][self.fnct_name]= {}
                if algorithm.nfe not in self.statistics["DE"][self.fnct_name]:
                    self.statistics["DE"][self.fnct_name][algorithm.nfe]= {}
                    if self.num_study not in self.statistics["DE"][self.fnct_name][algorithm.nfe]:
                        self.statistics["DE"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population}
            else:
                if algorithm.nfe not in self.statistics["DE"][self.fnct_name]:
                    self.statistics["DE"][self.fnct_name][algorithm.nfe]= {}
                    if self.num_study not in self.statistics["DE"][self.fnct_name][algorithm.nfe]:
                        self.statistics["DE"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population}
                else:
                    if self.num_study not in self.statistics["DE"][self.fnct_name][algorithm.nfe]:
                        self.statistics["DE"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population}
            
    def plot_bxplt_stat(self, algorithm):
        isInit= False
        data= []
        labels= []
        data_hv= [0]
        fig = plt.figure()
        ax = fig.gca()
        for nfe in algorithm.statistics.keys():
            data.append(algorithm.statistics[nfe]["population_fitness"])
            data_hv.append(algorithm.statistics[nfe]["hv_value"])
            labels.append(str(nfe) + " NFE")
        ax.boxplot(data, labels= labels)
        ticks = ax.get_xticks()
        ax2 = ax.twinx()
        ax2.plot(data_hv, color= "red")
        ax2.tick_params(axis='y', labelcolor="red")
        plt.show()

    def plot_bxplt_stat_cec(self, fnct_names):
        isInit= False
        
        fig = plt.figure()
        grid= GridSpec(4, len(fnct_names), wspace=0.25, hspace=0.5)
        dic= dict()
        for a, fnct in enumerate(fnct_names):
            for algorithm in self.statistics.keys():
                data= []
                hv_data= [0]
                labels= []
                if algorithm not in dic.keys():
                    dic[algorithm]= {}
                dic[algorithm][fnct]= {}
                for nfe in self.statistics[algorithm][fnct].keys():
                    labels.append(str(nfe) + " NFE")
                    fnct_array= []
                    fnct_hv= []
                    for exec in self.statistics[algorithm][fnct][nfe].keys():
                        fnct_array.append(self.statistics[algorithm][fnct][nfe][exec]["population_fitness"])
                        fnct_hv.append(self.statistics[algorithm][fnct][nfe][exec]["hv_value"])
                    data.append(np.mean(fnct_array, axis= 0))
                    hv_data.append(np.mean(fnct_hv))
                dic[algorithm][fnct]= {"fitness": data, "hv": hv_data, "labels": labels}

        for row, algorithm in enumerate(dic.keys()):
            for col, function in enumerate(dic[algorithm].keys()):
                ax= fig.add_subplot(grid[row, col])
                ax.set_title(algorithm + ": " + function)
                ax.boxplot(dic[algorithm][function]["fitness"], labels= dic[algorithm][function]["labels"])
                ticks = ax.get_xticks()
                ax2 = ax.twinx()
                ax2.plot(dic[algorithm][function]["hv"], color= "red")
                ax2.tick_params(axis='y', labelcolor="red")
        plt.show()
        

    def do_task(self, algorithm):
        #self.cpt+=1
        if not hasattr(algorithm, 'statistics'):
            algorithm.statistics= {'nfe': [], 'avg': [], 'min': [], 'max': [], 'std': []}
            if isinstance(algorithm, SMPSO):
                self.algo_names= np.append(self.algo_names, "SMPSO")
            elif isinstance(algorithm, NSGAII):
                self.algo_names= np.append(self.algo_names, "NSGAII")
            elif isinstance(algorithm, GDE3):
                self.algo_names= np.append(self.algo_names, "GDE3")
            elif isinstance(algorithm, GeneticAlgorithm):
                self.algo_names= np.append(self.algo_names, "Genetic Algorithm")
        algorithm.statistics['nfe'].append(algorithm.nfe)
        #algorithm.statistics['nfe'].append(algorithm.nfe)
        if self.isClassic:
            fitness= [s.objectives[0] for s in algorithm.population]
        else:
            fitness= [s.objectives[0] for s in algorithm.particles]
        algorithm.statistics['avg'].append(np.average(fitness))
    
    def display_plot_stat(self, algorithm_list):
        for i in range(len(algorithm_list)):
            plt.plot(algorithm_list[i].statistics['nfe'], algorithm_list[i].statistics['avg'], label= self.algo_names[i])
        plt.title("Genetic Algorithm(NSGA-II) / Differential Evolution / Particle Swarm Optimization")
        plt.xlabel('Number of Function Evaluations')
        plt.ylabel('Average min fitness')
        plt.legend()
        plt.show()
    
    def plot_CEC2005_stat(self, results, indicator, dim, types_problems= dict()):
        indicators_result = calculate(results, indicator)
        #print("Best element for Crossover => " + Xover + " and Mutation => " + Mutation + " " + "Benchmark Function " + str(bench_function) + " --> " + str(bench_stats))
        data= dict()
        for key_algo, algo in results.items():
            for key_problem, problem in algo.items():
                for key_indicator, indicator in indicators_result.items():
                    data[(key_algo,key_problem)] = indicators_result[key_algo][key_problem]['bestFitness']
        data_df= pd.DataFrame(data)
        csv_writer= CSVWriter()
        algos_names= np.array(["NSGAII", "DE", "SMPSO"])
        data_variation=np.empty((0, len(algos_names)))
        for i in range(1, 20):
            row= np.array([])
            for algo in algos_names:
                row= np.append(row, data_df[algo]["F" + str(i) + "_" + str(dim) + "D"].max() - data_df[algo]["F" + str(i) + "_" + str(dim) + "D"].min())
            data_variation= np.vstack((data_variation, row))
        test= data_variation.shape[0]
        test2= data_variation.shape[1]
        csv_writer.print_variation_fitness(data= data_variation, algos_names= algos_names)
        #print(str(key_algo) + " " + str(key_problem) + "Diff MinMax ==> " + str(data_df[key_algo]["F1_" + str(dim) + 'D'].max() - data_df["F1_" + str(dim) + 'D'].min()))
        data_df.to_csv("data.csv")
        print(data_df.describe())
        data_df= data_df.stack(level= 0).unstack()
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(20, 10))
        data_df["F1_" + str(dim) + 'D'].plot(ax= ax1)
        ax1.set_title("F1_" + str(dim) + "D: à " + str(types_problems["F1_" + str(dim) + "D"]))
        data_df["F2_" + str(dim) + "D"].plot(ax= ax2)
        ax2.set_title("F2_" + str(dim) + "D: à " + str(types_problems["F2_" + str(dim) + "D"]))
        data_df["F3_" + str(dim) + "D"].plot(ax= ax3)
        ax3.set_title("F3_"  + str(dim) + "D: à " + str(types_problems["F3_"  + str(dim) + "D"]))
        data_df["F4_" + str(dim) + "D"].plot(ax= ax4)
        ax4.set_title("F4_" + str(dim) + "D: à " + str(types_problems["F4_"  + str(dim) + "D"]))
        data_df["F5_" + str(dim) + "D"].plot(ax= ax5)
        ax5.set_title("F5_" + str(dim) + "D: à " + str(types_problems["F5_"  + str(dim) + "D"]))
        data_df["F6_" + str(dim) + "D"].plot(ax= ax6)
        ax6.set_title("F6_" + str(dim) + "D: à " + str(types_problems["F6_"  + str(dim) + "D"]))
        data_df["F7_" + str(dim) + "D"].plot(ax= ax7)
        ax7.set_title("F7_" + str(dim) + "D: à " + str(types_problems["F7_"  + str(dim) + "D"]))
        data_df["F8_" + str(dim) + "D"].plot(ax= ax8)
        ax8.set_title("F8_" + str(dim) + "D: à " + str(types_problems["F8_"  + str(dim) + "D"]))
        data_df["F9_" + str(dim) + "D"].plot(ax= ax9)
        ax9.set_title("F9_" + str(dim) + "D: à " + str(types_problems["F9_"  + str(dim) + "D"]))
        fig2, ((ax10, ax11, ax12), (ax13, ax14, ax15), (ax16, ax17, ax18)) = plt.subplots(3, 3, figsize=(20, 10))
        data_df["F10_" + str(dim) + "D"].plot(ax= ax10)
        ax10.set_title("F10_" + str(dim) + "D: à " + str(types_problems["F10_" + str(dim) + "D"]))
        data_df["F11_" + str(dim) + "D"].plot(ax= ax11)
        ax11.set_title("F11_" + str(dim) + "D: à " + str(types_problems["F11_" + str(dim) + "D"]))
        data_df["F12_" + str(dim) + "D"].plot(ax= ax12)
        ax12.set_title("F12_" + str(dim) + "D: à " + str(types_problems["F12_" + str(dim) + "D"]))
        data_df["F13_" + str(dim) + "D"].plot(ax= ax13)
        ax13.set_title("F13_" + str(dim) + "D: à " + str(types_problems["F13_" + str(dim) + "D"]))
        data_df["F14_" + str(dim) + "D"].plot(ax= ax14)
        ax14.set_title("F14_" + str(dim) + "D: à " + str(types_problems["F14_" + str(dim) + "D"]))
        data_df["F15_" + str(dim) + "D"].plot(ax= ax15)
        ax15.set_title("F15_" + str(dim) + "D: à " + str(types_problems["F15_" + str(dim) + "D"]))
        data_df["F16_" + str(dim) + "D"].plot(ax= ax16)
        ax16.set_title("F16_" + str(dim) + "D: à " + str(types_problems["F16_" + str(dim) + "D"]))
        data_df["F17_" + str(dim) + "D"].plot(ax= ax17)
        ax17.set_title("F17_" + str(dim) + "D: à " + str(types_problems["F17_" + str(dim) + "D"]))
        data_df["F18_" + str(dim) + "D"].plot(ax= ax18)
        ax18.set_title("F18_" + str(dim) + "D: à " + str(types_problems["F18_" + str(dim) + "D"]))
        plt.show(block= True)