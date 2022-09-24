import random
import json
from typing import Dict, List, Tuple, Set
from mobile import PROJECT_ROOT
from mobile.utils import *
from mobile.heuristics import *
import matplotlib.pyplot as plt

with open(PROJECT_ROOT/"output"/"runs"/"charlottesville_city"/"tradeoff.json") as f:
    data = json.load(f)
    algs = data.keys()
excluded = {'most_coverage'}

def gen_plot_k(k: int):
    alg_k_facilities = {}
    for alg, runs in data.items():
        
        if alg in excluded: continue
        
        for selected_fac in runs:
            if len(selected_fac[0]) == k:
                alg_k_facilities[alg] = [selected_fac[0], selected_fac[1]]
                break
    #print(alg_k_facilities)
    
    alg_percentile_objective = {}
    for alg, selected in alg_k_facilities.items():
        alg_percentiles = []
        alg_objectives = []
        
        selected_fac = selected[0]
        assignment = selected[1]
        
        for percentile in range(800, 1001):
            alg_percentiles.append(percentile/10)
            alg_objectives.append(calculate_objective(assignment, percentile/10))
        
        alg_percentile_objective[alg] = [alg_percentiles, alg_objectives]
    
    #plt.plot(alg_percentiles, alg_objectives, label = alg)
    
    fig = plt.figure()

    for alg, objectives in alg_percentile_objective.items():
        
        alg_percentiles = objectives[0]
        alg_objectives = objectives[1]
        plt.plot(alg_percentiles, alg_objectives, label = alg)
    
    plt.xlabel("Percentile of Clients Covered")
    plt.ylabel("Objective Value (km)")
    
    plt.legend()
    
    fig.set_figheight(5)
    fig.set_figwidth(8)
    
    fig.savefig(PROJECT_ROOT/"output"/"plots"/"charlottesville_city"/f"percentile_coverage_{k}.png", dpi = 300)
    
for k in range(3, 11):
    print(k)
    gen_plot_k(k)