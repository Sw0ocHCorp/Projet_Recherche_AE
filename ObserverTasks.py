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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.optimize as opt


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
        self.cpt= 0
        self.prev_gen= None
        self.prev_fitness= None

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
    
    def set_exec(self, exec):
        self.exec= exec

    def save_stat(self, algorithm):
        self.cpt+= 1
        if self.cpt % 5 == 0:
            self.cpt=0
            sol = opt.minimize(algorithm.problem.function, np.zeros(algorithm.problem.nvars), method= "SLSQP", options={'maxiter': 500, 'disp': False})
            domin_points= nondominated(algorithm.result)
            isOptimal= False
            for domin_point in domin_points:
                if self.is_optimal(domin_point.objectives, sol.fun, 0.1):
                    isOptimal= True
            hv= Hypervolume(minimum= [0,0], maximum= [1,1])
            fitness= []
            problem_direction= ""
            algo_name= ""
            if algorithm.problem.directions[0] == -1:
                problem_direction= "A Minimiser"
            else:
                problem_direction= "A Maximiser"
            
            if isinstance(algorithm, SMPSO):
                algo_name= "SMPSO"
                hv= Hypervolume(reference_set= algorithm.particles)
                fitness= [s.objectives[0] for s in algorithm.particles]
                if algorithm.nfe not in self.statistics[algo_name].keys():
                    self.statistics[algo_name][algorithm.nfe]= {}
                    if self.exec not in self.statistics[algo_name][algorithm.nfe].keys():
                        self.statistics[algo_name][algorithm.nfe][self.exec]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.particles), "particles": algorithm.particles, "isOptimal": isOptimal, "direction": problem_direction, "domin_point":domin_points}
                else:
                    if self.exec not in self.statistics[algo_name][algorithm.nfe].keys():
                        self.statistics[algo_name][algorithm.nfe][self.exec]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.particles), "particles": algorithm.particles, "isOptimal": isOptimal, "direction": problem_direction, "domin_point":domin_points}

            else:
                if isinstance(algorithm, NSGAII):
                    algo_name= "NSGAII"
                elif isinstance(algorithm, GDE3):
                    algo_name= "DE"
                hv= Hypervolume(reference_set= algorithm.population)
                fitness= [s.objectives[0] for s in algorithm.population]
                if algorithm.nfe not in self.statistics[algo_name].keys():
                    self.statistics[algo_name][algorithm.nfe]= {}
                    if self.exec not in self.statistics[algo_name][algorithm.nfe].keys():
                        self.statistics[algo_name][algorithm.nfe][self.exec]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population, "isOptimal": isOptimal, "direction": problem_direction, "domin_point":domin_points}
                else:
                    if self.exec not in self.statistics[algo_name][algorithm.nfe].keys():
                        self.statistics[algo_name][algorithm.nfe][self.exec]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population, "isOptimal": isOptimal, "direction": problem_direction, "domin_point":domin_points}
        
    def save_stat_cec(self, algorithm):
        self.cpt+= 1
        if self.cpt % 5 == 0:
            test= algorithm.problem.CECProblem.get_optimal_solutions()
            self.cpt=0
            sol= Solution(algorithm.problem)
            sol.variables= algorithm.problem.CECProblem.get_optimal_solutions()[0].phenome
            best_fit= algorithm.problem.evaluate(sol)
            """if algorithm.problem.CECProblem.min_bounds == None:
                sol = opt.minimize(algorithm.problem.function, np.zeros(algorithm.problem.nvars), method= "SLSQP", options={'maxiter': 500, 'disp': False})
            else:
                sol = opt.minimize(algorithm.problem.function, np.zeros(algorithm.problem.nvars), method= "SLSQP", bounds= opt.Bounds(lb= algorithm.problem.CECProblem.min_bounds, ub= algorithm.problem.CECProblem.max_bounds), options={'maxiter': 500, 'disp': False})
            """
            hv= Hypervolume(minimum= [0,0], maximum= [1,1])
            fitness= []
            pop= []
            #benchmark_optimum = opt.minimize(algorithm.problem.function, [100, 100], method='SLSQP', options={'maxiter': 100, 'disp': True})
            isOptimal= False
            domin_points= nondominated(algorithm.result)
            if isinstance(algorithm, SMPSO):
                for domin_point in domin_points:
                    if isinstance(domin_point.objectives, float):
                        if isinstance(sol.objectives[0], float):
                            if self.is_optimal(domin_point.objectives, sol.objectives[0], int(abs(sol.objectives[0]) / 50)):
                                isOptimal= True
                                break
                        else:
                            if self.is_optimal(domin_point.objectives, sol.objectives[0][0], int(abs(sol.objectives[0][0]) / 50)):
                                isOptimal= True
                                break
                    else:
                        if isinstance(sol.objectives[0], float):
                            if self.is_optimal(domin_point.objectives[0], sol.objectives[0], int(abs(sol.objectives[0]) / 50)):
                                isOptimal= True
                                break
                        else:
                            if self.is_optimal(domin_point.objectives[0], sol.objectives[0][0], int(abs(sol.objectives[0][0]) / 50)):
                                isOptimal= True
                                break
                hv= Hypervolume(reference_set= algorithm.particles)
                fitness= [s.objectives[0] for s in algorithm.particles]
                if self.fnct_name not in self.statistics["SMPSO"].keys():
                    self.statistics["SMPSO"][self.fnct_name]= {}
                    self.statistics["SMPSO"][self.fnct_name]["optimum"]= sol
                    if algorithm.nfe not in self.statistics["SMPSO"][self.fnct_name]:
                        self.statistics["SMPSO"][self.fnct_name][algorithm.nfe]= {}
                        if self.num_study not in self.statistics["SMPSO"][self.fnct_name][algorithm.nfe]:
                            self.statistics["SMPSO"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.particles), "particles": algorithm.particles, "domin_point": domin_points[0], "problem_direction": algorithm.problem.directions[0], "isOptimal": isOptimal}
                else:
                    if algorithm.nfe not in self.statistics["SMPSO"][self.fnct_name]:
                        self.statistics["SMPSO"][self.fnct_name][algorithm.nfe]= {}
                        if self.num_study not in self.statistics["SMPSO"][self.fnct_name][algorithm.nfe]:
                            self.statistics["SMPSO"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.particles), "particles": algorithm.particles, "domin_point": domin_points[0], "problem_direction": algorithm.problem.directions[0], "isOptimal": isOptimal}
                    else:
                        if self.num_study not in self.statistics["SMPSO"][self.fnct_name][algorithm.nfe]:
                            self.statistics["SMPSO"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.particles), "particles": algorithm.particles, "domin_point": domin_points[0], "problem_direction": algorithm.problem.directions[0], "isOptimal": isOptimal}

            elif isinstance(algorithm, NSGAII):
                for domin_point in domin_points:
                    if isinstance(domin_point.objectives, float):
                        if isinstance(sol.objectives[0], float):
                            if self.is_optimal(domin_point.objectives, sol.objectives[0], int(abs(sol.objectives[0]) / 50)):
                                isOptimal= True
                                break
                        else:
                            if self.is_optimal(domin_point.objectives, sol.objectives[0][0], int(abs(sol.objectives[0][0]) / 50)):
                                isOptimal= True
                                break
                    else:
                        if isinstance(sol.objectives[0], float):
                            if self.is_optimal(domin_point.objectives[0], sol.objectives[0], int(abs(sol.objectives[0]) / 50)):
                                isOptimal= True
                                break
                        else:
                            if self.is_optimal(domin_point.objectives[0], sol.objectives[0][0], int(abs(sol.objectives[0][0]) / 50)):
                                isOptimal= True
                                break
                hv= Hypervolume(reference_set= algorithm.population)
                fitness= [s.objectives[0] for s in algorithm.population]
                if self.fnct_name not in self.statistics["NSGAII"].keys():
                    self.statistics["NSGAII"][self.fnct_name]= {}
                    self.statistics["NSGAII"][self.fnct_name]["optimum"]= sol
                    if algorithm.nfe not in self.statistics["NSGAII"][self.fnct_name]:
                        if len(self.saved_nfe) < 4:
                            self.saved_nfe.append(algorithm.nfe)
                        self.statistics["NSGAII"][self.fnct_name][algorithm.nfe]= {}
                        if self.num_study not in self.statistics["NSGAII"][self.fnct_name][algorithm.nfe]:
                            self.statistics["NSGAII"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population, "domin_point": domin_point, "nfe": algorithm.nfe, "problem_direction": algorithm.problem.directions[0], "isOptimal": isOptimal}
                else:
                    if len(self.saved_nfe) < 4:
                        if algorithm.nfe not in self.statistics["NSGAII"][self.fnct_name]:
                            self.saved_nfe.append(algorithm.nfe)
                            self.statistics["NSGAII"][self.fnct_name][algorithm.nfe]= {}
                            if self.num_study not in self.statistics["NSGAII"][self.fnct_name][algorithm.nfe]:
                                self.statistics["NSGAII"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population, "domin_point": domin_point, "nfe": algorithm.nfe, "problem_direction": algorithm.problem.directions[0], "isOptimal": isOptimal}
                        else:
                            if self.num_study not in self.statistics["NSGAII"][self.fnct_name][algorithm.nfe]:
                                self.statistics["NSGAII"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population, "domin_point": domin_point, "nfe": algorithm.nfe, "problem_direction": algorithm.problem.directions[0], "isOptimal": isOptimal}
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
                            self.statistics["NSGAII"][self.fnct_name][target_nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population, "domin_point": domin_point, "nfe": algorithm.nfe, "problem_direction": algorithm.problem.directions[0], "isOptimal": isOptimal}

            elif isinstance(algorithm, GDE3):
                for domin_point in domin_points:
                    if isinstance(domin_point.objectives, float):
                        if isinstance(sol.objectives[0], float):
                            if self.is_optimal(domin_point.objectives, sol.objectives[0], int(abs(sol.objectives[0]) / 50)):
                                isOptimal= True
                                break
                        else:
                            if self.is_optimal(domin_point.objectives, sol.objectives[0][0], int(abs(sol.objectives[0][0]) / 50)):
                                isOptimal= True
                                break
                    else:
                        if isinstance(sol.objectives[0], float):
                            if self.is_optimal(domin_point.objectives[0], sol.objectives[0], int(abs(sol.objectives[0]) / 50)):
                                isOptimal= True
                                break
                hv= Hypervolume(reference_set= algorithm.population)
                fitness= [s.objectives[0] for s in algorithm.population]
                if self.fnct_name not in self.statistics["DE"].keys():
                    self.statistics["DE"][self.fnct_name]= {}
                    self.statistics["DE"][self.fnct_name]["optimum"]= sol
                    if algorithm.nfe not in self.statistics["DE"][self.fnct_name]:
                        self.statistics["DE"][self.fnct_name][algorithm.nfe]= {}
                        if self.num_study not in self.statistics["DE"][self.fnct_name][algorithm.nfe]:
                            self.statistics["DE"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population, "domin_point": domin_point, "problem_direction": algorithm.problem.directions[0], "isOptimal": isOptimal}
                else:
                    if algorithm.nfe not in self.statistics["DE"][self.fnct_name]:
                        self.statistics["DE"][self.fnct_name][algorithm.nfe]= {}
                        if self.num_study not in self.statistics["DE"][self.fnct_name][algorithm.nfe]:
                            self.statistics["DE"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population, "domin_point": domin_point, "problem_direction": algorithm.problem.directions[0], "isOptimal": isOptimal}
                    else:
                        if self.num_study not in self.statistics["DE"][self.fnct_name][algorithm.nfe]:
                            self.statistics["DE"][self.fnct_name][algorithm.nfe][self.num_study]= {"population_fitness": fitness, "hv_value": hv.calculate(algorithm.population), "population": algorithm.population, "domin_point": domin_point, "problem_direction": algorithm.problem.directions[0], "isOptimal": isOptimal}

    def is_optimal(self, solution, benchmark_optimum, success_tol):
        """Check if the given solution is optimal for the given benchmark function."""
        if isinstance(solution, float):
            if abs(abs(solution) - abs(benchmark_optimum)) <= success_tol:
                return True
            else:
                return False
        else:
            if abs(abs(solution[0]) - abs(benchmark_optimum)) <= success_tol:
                return True
            else:
                return False

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

    def plot_sr(self):
        fig= plt.figure()
        dic= dict()
        max_labels= 0
        for algorithm in self.statistics.keys():
            if len(self.statistics[algorithm].keys()) > 0:
                print("------------------")
                optimum= []
                labels= []
                problem_direction= ""
                if algorithm not in dic.keys():
                    dic[algorithm]= {}
                for nfe in self.statistics[algorithm].keys():
                    cpt_sr= 0
                    if nfe != "optimum":
                        labels.append(nfe)
                        for exec in self.statistics[algorithm][nfe].keys():
                            if self.statistics[algorithm][nfe][exec]["isOptimal"]:
                                cpt_sr+= 1
                            problem_direction= self.statistics[algorithm][nfe][exec]["direction"]
                        optimum.append(cpt_sr / len(self.statistics[algorithm][nfe].keys()))
                        print("SR -> " + algorithm + " Avec " + str(nfe) + " NFE= " + str(cpt_sr / 10))
                if len(labels) > max_labels:
                    max_labels= len(labels)
                dic[algorithm]= {"labels": labels, "optimum": optimum, "problem_direction": problem_direction}
        
        for row, algorithm in enumerate(dic.keys()):
            if len(self.statistics[algorithm].keys()) > 0:
                ax= fig.add_subplot()
                ax.set_title("SR " + algorithm + " => " + dic[algorithm]["problem_direction"])
                ax.plot(dic[algorithm]["labels"], dic[algorithm]["optimum"])
        plt.show()


    def plot_diversity(self):
        fig= plt.figure()
        dic= dict()
        max_labels= 0
        for algorithm in self.statistics.keys():
            if len(self.statistics[algorithm].keys()) > 0:
                x_values= []
                y_values= []
                fitness_data= []
                labels= []
                optimum= []
                domin_point= []
                problem_direction= ""
                if algorithm not in dic.keys():
                    dic[algorithm]= {}
                dic[algorithm]= {}
                for nfe in self.statistics[algorithm].keys():
                    if nfe != "optimum":
                        cpt_sr= 0
                        x_array= []
                        y_array= []
                        fit_array= []
                        labels.append(str(nfe) + " NFE")
                        if algorithm == "SMPSO":
                            for exec in self.statistics[algorithm][nfe].keys():
                                if self.statistics[algorithm][nfe][exec]["isOptimal"] == True:
                                    cpt_sr+= 1
                                if len(self.statistics[algorithm][nfe][exec]["particles"][0].variables) > 2:
                                    data= [particle.variables for particle in self.statistics[algorithm][nfe][exec]["particles"]]
                                    data.append(self.statistics[algorithm][nfe][exec]["domin_point"].variables)
                                    transformed_data= TSNE(n_components=2, learning_rate='auto', 
                                                            init='random', perplexity=3).fit_transform(np.array(data))
                                    x_array.append(transformed_data[:,0].tolist())
                                    y_array.append(transformed_data[:,1].tolist())
                                    domin_point= transformed_data[-1, :].tolist()
                                else:
                                    x_array.append([particle.variables[0] for particle in self.statistics[algorithm][nfe][exec]["particles"]])
                                    y_array.append([particle.variables[1] for particle in self.statistics[algorithm][nfe][exec]["particles"]])
                                    domin_point= self.statistics[algorithm][nfe][exec]["domin_point"][0].variables
                                fit_array.append(self.statistics[algorithm][nfe][exec]["population_fitness"])
                                problem_direction= self.statistics[algorithm][nfe][exec]["direction"]
                        else:
                            for exec in self.statistics[algorithm][nfe].keys():
                                if self.statistics[algorithm][nfe][exec]["isOptimal"] == True:
                                    cpt_sr+= 1
                                if len(self.statistics[algorithm][nfe][exec]["population"][0].variables) > 2:
                                    data= [indiv.variables for indiv in self.statistics[algorithm][nfe][exec]["population"]]
                                    data.append(self.statistics[algorithm]["optimum"])
                                    data.append(self.statistics[algorithm][nfe][exec]["domin_point"][0].variables)
                                    transformed_data= TSNE(n_components=2, learning_rate='auto', 
                                                            init='random', perplexity=3).fit_transform(np.array(data))
                                    x_array.append(transformed_data[:,0].tolist())
                                    y_array.append(transformed_data[:,1].tolist())
                                    optimum= transformed_data[-2, :].tolist()
                                    domin_point= transformed_data[-1, :].tolist()
                                else:
                                    x_array.append([particle.variables[0] for particle in self.statistics[algorithm][nfe][exec]["population"]])
                                    y_array.append([particle.variables[1] for particle in self.statistics[algorithm][nfe][exec]["population"]])
                                    domin_point= self.statistics[algorithm][nfe][exec]["domin_point"][0].variables
                                fit_array.append(self.statistics[algorithm][nfe][exec]["population_fitness"])
                                problem_direction= self.statistics[algorithm][nfe][exec]["direction"]
                        x_values.append(np.mean(x_array, axis= 0))
                        y_values.append(np.mean(y_array, axis= 0))
                        fitness_data.append(np.mean(fit_array, axis= 0))
                if len(labels) > max_labels:
                    max_labels= len(labels)
                dic[algorithm]= {"x_values": x_values, "y_values": y_values, "fitness": fitness_data, "labels": labels, "problem_direction": problem_direction, "optimum": optimum, "domin_point": domin_point}
        grid= GridSpec(4, max_labels, wspace=0.25, hspace=0.5)
        for row, algorithm in enumerate(dic.keys()):
            for col, label in enumerate(dic[algorithm]["labels"]):
                if len(self.statistics[algorithm].keys()) > 0:
                    ax= fig.add_subplot(grid[row, col])  
                    ax.set_title(algorithm + " => " + dic[algorithm]["problem_direction"] + ": " + label) 
                    scatplot= ax.scatter(dic[algorithm]["x_values"][col], dic[algorithm]["y_values"][col], color= "blue")
                    ax.scatter(dic[algorithm]["domin_point"][0], dic[algorithm]["domin_point"][1], color= "red")
                    #ax.scatter(dic[algorithm][fnct_name]["domin_point"][0], dic[algorithm][fnct_name]["domin_point"][1], color= "purple")
        plt.show()
    

    def plot_sr_stat(self, fnct_names):
        fig = plt.figure()
        grid= GridSpec(4, len(fnct_names), wspace=0.25, hspace=0.5)
        dic= dict()
        for a, fnct in enumerate(fnct_names):
            for algorithm in self.statistics.keys():
                print("------------------")
                optimum= []
                labels= []
                problem_direction= ""
                if algorithm not in dic.keys():
                    dic[algorithm]= {}
                dic[algorithm][fnct]= {}
                for nfe in self.statistics[algorithm][fnct].keys():
                    cpt_sr= 0
                    if nfe != "optimum":
                        labels.append(nfe)
                        for exec in self.statistics[algorithm][fnct][nfe].keys():
                            if self.statistics[algorithm][fnct][nfe][exec]["isOptimal"]:
                                cpt_sr+= 1
                            if self.statistics[algorithm][fnct][nfe][exec]["problem_direction"] == -1:
                                problem_direction= "A Minimiser"
                            else:
                                problem_direction= "A Maximiser"
                        optimum.append(cpt_sr / len(self.statistics[algorithm][fnct][nfe].keys()))
                        print("SR -> " + algorithm + " " + fnct + " Avec " + str(nfe) + " NFE= " + str(cpt_sr / 10))
                dic[algorithm][fnct]= {"labels": labels, "optimum": optimum, "problem_direction": problem_direction}
            
        for row, algorithm in enumerate(dic.keys()):
            for col, function in enumerate(dic[algorithm].keys()):
                ax= fig.add_subplot(grid[row, col])
                ax.set_title("SR " + algorithm + ": " + function + " => " + dic[algorithm][function]["problem_direction"])
                ax.plot(dic[algorithm][function]["labels"], dic[algorithm][function]["optimum"])
        plt.show()
                

    def plot_bxplt_stat_cec(self, fnct_names):
        isInit= False
        fig = plt.figure()
        grid= GridSpec(4, len(fnct_names), wspace=0.25, hspace=0.5)
        dic= dict()
        statistics= dict()
        for a, fnct in enumerate(fnct_names):
            for algorithm in self.statistics.keys():
                data= []
                hv_data= [0]
                sr_data= [0]
                labels= []
                problem_direction= ""
                optimum= [0]
                if algorithm not in dic.keys():
                    dic[algorithm]= {}
                    statistics[algorithm]= {}
                dic[algorithm][fnct]= {}
                statistics[algorithm][fnct]= {}
                for key in self.statistics[algorithm][fnct].keys():
                    if key == "optimum":
                        continue
                    else:
                        statistics[algorithm][fnct][key]= self.statistics[algorithm][fnct][key]
                for nfe in sorted(statistics[algorithm][fnct].keys()):
                    labels.append(str(nfe) + " NFE")
                    fnct_array= []
                    fnct_hv= []
                    fnct_sr= []
                    test= statistics[algorithm][fnct][nfe].keys()
                    for exec in statistics[algorithm][fnct][nfe].keys():
                        fnct_array.append(statistics[algorithm][fnct][nfe][exec]["population_fitness"])
                        #fnct_hv.append(self.statistics[algorithm][fnct][nfe][exec]["hv_value"])
                        fnct_sr.append(statistics[algorithm][fnct][nfe][exec]["isOptimal"])
                        if statistics[algorithm][fnct][nfe][exec]["problem_direction"] == -1:
                            problem_direction= "A Minimiser"
                        else:
                            problem_direction= "A Maximiser"
                    data.append(np.mean(fnct_array, axis= 0))
                    optimum.append(self.statistics[algorithm][fnct]["optimum"].objectives[0])
                    #hv_data.append(np.mean(fnct_hv))
                    if len(fnct_sr) > 0:
                        sr_data.append(np.sum(fnct_sr) / len(fnct_sr))
                    else:
                        sr_data.append(0)
                #dic[algorithm][fnct]= {"fitness": data, "metric": hv_data, "labels": labels, "problem_direction": problem_direction}
                dic[algorithm][fnct]= {"fitness": data, "metric": sr_data, "labels": labels, "problem_direction": problem_direction, "target": optimum}

        for row, algorithm in enumerate(dic.keys()):
            for col, function in enumerate(dic[algorithm].keys()):
                ax= fig.add_subplot(grid[row, col])
                ax.set_title(algorithm + ": " + function + " => " + dic[algorithm][function]["problem_direction"])
                bxplot= ax.boxplot(dic[algorithm][function]["fitness"], labels= dic[algorithm][function]["labels"])
                ax.plot(dic[algorithm][function]["target"], color= "green", label= "Optimum Fitness")
                plt.legend()
                q5= np.quantile(dic[algorithm][function]["fitness"], 0.05, axis= 1)
                for i in range(len(q5)):
                    ax.annotate(str(round(bxplot['caps'][i*2].get_ydata()[0], 2)), (i+1, bxplot['caps'][i*2].get_ydata()[0]*0.95), color= "blue")
                    ax.annotate(str(round(bxplot['caps'][(i*2)+1].get_ydata()[0], 2)), (i+1, bxplot['caps'][(i*2)+1].get_ydata()[0]*1.05), color= "blue")
                ticks = ax.get_xticks()
                ax2 = ax.twinx()
                ax2.plot(dic[algorithm][function]["metric"], color= "red", label= "SR")

                for i in range(1, len(dic[algorithm][function]["metric"])):
                    ax2.annotate(str(round(dic[algorithm][function]["metric"][i], 2)), (i, dic[algorithm][function]["metric"][i]- 0.1), color= "purple")
                ax2.tick_params(axis='y', labelcolor="red")
                plt.legend()
        plt.show()

    def plot_population_diversity(self, fnct_names): 
        fig_list= [plt.figure() for i in range(len(fnct_names))]   
        grid_list= [GridSpec(4, 5, wspace=0.25, hspace=0.5) for i in range(len(fnct_names))]
        dic= dict()
        statistics= dict()
        cpt_sr= 0
        for a, fnct in enumerate(fnct_names):
            for algorithm in self.statistics.keys():
                x_values= []
                y_values= []
                fitness_data= []
                labels= []
                optimum= []
                domin_point= []
                problem_direction= ""
                if algorithm not in dic.keys():
                    dic[algorithm]= {}
                    statistics[algorithm]= {}
                dic[algorithm][fnct]= {}
                statistics[algorithm][fnct]= {}
                for key in self.statistics[algorithm][fnct].keys():
                    if key == "optimum":
                        continue
                    else:
                        statistics[algorithm][fnct][key]= self.statistics[algorithm][fnct][key]
                for nfe in sorted(statistics[algorithm][fnct].keys()):
                    cpt_sr= 0
                    x_array= []
                    y_array= []
                    fit_array= []
                    labels.append(str(nfe) + " NFE")
                    if algorithm == "SMPSO":
                        for exec in statistics[algorithm][fnct][nfe].keys():
                            if statistics[algorithm][fnct][nfe][exec]["isOptimal"] == True:
                                cpt_sr+= 1
                            if len(statistics[algorithm][fnct][nfe][exec]["particles"][0].variables) > 2:
                                data= [particle.variables for particle in self.statistics[algorithm][fnct][nfe][exec]["particles"]]
                                data.append(self.statistics[algorithm][fnct]["optimum"].variables)
                                transformed_data= TSNE(n_components=2, learning_rate='auto', 
                                                        init='random', perplexity=3).fit_transform(np.array(data))
                                x_array.append(transformed_data[0:-1,0].tolist())
                                y_array.append(transformed_data[0:-1,1].tolist())
                                optimum= transformed_data[-1, :].tolist()
                            else:
                                x_array.append([particle.variables[0] for particle in self.statistics[algorithm][fnct][nfe][exec]["particles"]])
                                y_array.append([particle.variables[1] for particle in self.statistics[algorithm][fnct][nfe][exec]["particles"]])
                                optimum= self.statistics[algorithm][fnct]["optimum"].variables
                                domin_point= statistics[algorithm][fnct][nfe][exec]["domin_point"].variables
                            fit_array.append(statistics[algorithm][fnct][nfe][exec]["population_fitness"])
                            if statistics[algorithm][fnct][nfe][exec]["problem_direction"] == -1:
                                problem_direction= "A Minimiser"
                            else:
                                problem_direction= "A Maximiser"
                    else:
                        for exec in statistics[algorithm][fnct][nfe].keys():
                            if statistics[algorithm][fnct][nfe][exec]["isOptimal"] == True:
                                cpt_sr+= 1
                            if len(statistics[algorithm][fnct][nfe][exec]["population"][0].variables) > 2:
                                data= [indiv.variables for indiv in self.statistics[algorithm][fnct][nfe][exec]["population"]]
                                data.append(self.statistics[algorithm][fnct]["optimum"].variables)
                                transformed_data= TSNE(n_components=2, learning_rate='auto', 
                                                        init='random', perplexity=3).fit_transform(np.array(data))
                                x_array.append(transformed_data[0:-1,0].tolist())
                                y_array.append(transformed_data[0:-1,1].tolist())
                                optimum= transformed_data[-1, :].tolist()
                            else:
                                x_array.append([particle.variables[0] for particle in self.statistics[algorithm][fnct][nfe][exec]["population"]])
                                y_array.append([particle.variables[1] for particle in self.statistics[algorithm][fnct][nfe][exec]["population"]])
                                optimum= self.statistics[algorithm][fnct]["optimum"].variables
                                domin_point= statistics[algorithm][fnct][nfe][exec]["domin_point"].variables
                            fit_array.append(statistics[algorithm][fnct][nfe][exec]["population_fitness"])
                            if statistics[algorithm][fnct][nfe][exec]["problem_direction"] == -1:
                                problem_direction= "A Minimiser"
                            else:
                                problem_direction= "A Maximiser"
                    x_values.append(np.mean(x_array, axis= 0))
                    y_values.append(np.mean(y_array, axis= 0))
                    fitness_data.append(np.mean(fit_array, axis= 0))
                dic[algorithm][fnct]= {"x_values": x_values, "y_values": y_values, "fitness": fitness_data, "labels": labels, "problem_direction": problem_direction, "optimum_point": optimum, "domin_point": domin_point}
        

        for num_fnct, fnct_name in enumerate(fnct_names):  
            for row, algorithm in enumerate(dic.keys()):
                for col, label in enumerate(dic[algorithm][fnct_name]["labels"]):
                    ax= fig_list[num_fnct].add_subplot(grid_list[num_fnct][row, col])  
                    ax.set_title(algorithm + " " + fnct_name + " => " + dic[algorithm][fnct_name]["problem_direction"] + ": " + label) 
                    scatplot= ax.scatter(dic[algorithm][fnct_name]["x_values"][col], dic[algorithm][fnct_name]["y_values"][col], color= "blue")
                    ax.scatter(dic[algorithm][fnct_name]["optimum_point"][0], dic[algorithm][fnct_name]["optimum_point"][1], color= "red")
                    #ax.scatter(dic[algorithm][fnct_name]["domin_point"][0], dic[algorithm][fnct_name]["domin_point"][1], color= "purple")
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
        plt.variableslabel('Number of Function Evaluations')
        plt.ylabel('Average min fitness')
        plt.legend()
        plt.show()
    
    def plot_CEC2005_stat(self, results, indicator, dim, types_problems= dict()):
        indicators_result = calculate(results= results, indicators= indicator)
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
        ax1.set_xlabel("Executions")
        ax1.set_ylabel("Best Fitness")
        data_df["F2_" + str(dim) + "D"].plot(ax= ax2)
        ax2.set_title("F2_" + str(dim) + "D: à " + str(types_problems["F2_" + str(dim) + "D"]))
        ax2.set_xlabel("Executions")
        ax2.set_ylabel("Best Fitness")
        data_df["F3_" + str(dim) + "D"].plot(ax= ax3)
        ax3.set_title("F3_"  + str(dim) + "D: à " + str(types_problems["F3_"  + str(dim) + "D"]))
        ax3.set_xlabel("Executions")
        ax3.set_ylabel("Best Fitness")
        data_df["F4_" + str(dim) + "D"].plot(ax= ax4)
        ax4.set_title("F4_" + str(dim) + "D: à " + str(types_problems["F4_"  + str(dim) + "D"]))
        ax4.set_xlabel("Executions")
        ax4.set_ylabel("Best Fitness")
        data_df["F5_" + str(dim) + "D"].plot(ax= ax5)
        ax5.set_title("F5_" + str(dim) + "D: à " + str(types_problems["F5_"  + str(dim) + "D"]))
        ax5.set_xlabel("Executions")
        ax5.set_ylabel("Best Fitness")
        data_df["F6_" + str(dim) + "D"].plot(ax= ax6)
        ax6.set_title("F6_" + str(dim) + "D: à " + str(types_problems["F6_"  + str(dim) + "D"]))
        ax6.set_xlabel("Executions")
        ax6.set_ylabel("Best Fitness")
        data_df["F7_" + str(dim) + "D"].plot(ax= ax7)
        ax7.set_title("F7_" + str(dim) + "D: à " + str(types_problems["F7_"  + str(dim) + "D"]))
        ax7.set_xlabel("Executions")
        ax7.set_ylabel("Best Fitness")
        data_df["F8_" + str(dim) + "D"].plot(ax= ax8)
        ax8.set_title("F8_" + str(dim) + "D: à " + str(types_problems["F8_"  + str(dim) + "D"]))
        ax8.set_xlabel("Executions")
        ax8.set_ylabel("Best Fitness")
        data_df["F9_" + str(dim) + "D"].plot(ax= ax9)
        ax9.set_title("F9_" + str(dim) + "D: à " + str(types_problems["F9_"  + str(dim) + "D"]))
        ax9.set_xlabel("Executions")
        ax9.set_ylabel("Best Fitness")
        fig.tight_layout()
        fig2, ((ax10, ax11, ax12), (ax13, ax14, ax15), (ax16, ax17, ax18)) = plt.subplots(3, 3, figsize=(20, 10))
        data_df["F10_" + str(dim) + "D"].plot(ax= ax10)
        ax10.set_title("F10_" + str(dim) + "D: à " + str(types_problems["F10_" + str(dim) + "D"]))
        ax10.set_xlabel("Executions")
        ax10.set_ylabel("Best Fitness")
        data_df["F11_" + str(dim) + "D"].plot(ax= ax11)
        ax11.set_title("F11_" + str(dim) + "D: à " + str(types_problems["F11_" + str(dim) + "D"]))
        ax11.set_xlabel("Executions")
        ax11.set_ylabel("Best Fitness")
        data_df["F12_" + str(dim) + "D"].plot(ax= ax12)
        ax12.set_title("F12_" + str(dim) + "D: à " + str(types_problems["F12_" + str(dim) + "D"]))
        ax12.set_xlabel("Executions")
        ax12.set_ylabel("Best Fitness")
        data_df["F13_" + str(dim) + "D"].plot(ax= ax13)
        ax13.set_title("F13_" + str(dim) + "D: à " + str(types_problems["F13_" + str(dim) + "D"]))
        ax13.set_xlabel("Executions")
        ax13.set_ylabel("Best Fitness")
        data_df["F14_" + str(dim) + "D"].plot(ax= ax14)
        ax14.set_title("F14_" + str(dim) + "D: à " + str(types_problems["F14_" + str(dim) + "D"]))
        ax14.set_xlabel("Executions")
        ax14.set_ylabel("Best Fitness")
        data_df["F15_" + str(dim) + "D"].plot(ax= ax15)
        ax15.set_title("F15_" + str(dim) + "D: à " + str(types_problems["F15_" + str(dim) + "D"]))
        ax15.set_xlabel("Executions")
        ax15.set_ylabel("Best Fitness")
        data_df["F16_" + str(dim) + "D"].plot(ax= ax16)
        ax16.set_title("F16_" + str(dim) + "D: à " + str(types_problems["F16_" + str(dim) + "D"]))
        ax16.set_xlabel("Executions")
        ax16.set_ylabel("Best Fitness")
        data_df["F17_" + str(dim) + "D"].plot(ax= ax17)
        ax17.set_title("F17_" + str(dim) + "D: à " + str(types_problems["F17_" + str(dim) + "D"]))
        ax17.set_xlabel("Executions")
        ax17.set_ylabel("Best Fitness")
        data_df["F18_" + str(dim) + "D"].plot(ax= ax18)
        ax18.set_title("F18_" + str(dim) + "D: à " + str(types_problems["F18_" + str(dim) + "D"]))
        ax18.set_xlabel("Executions")
        ax18.set_ylabel("Best Fitness")
        fig2.tight_layout()
        plt.show(block= True)