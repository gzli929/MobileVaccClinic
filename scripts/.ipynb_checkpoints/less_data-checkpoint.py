import random
from typing import Dict, List, Tuple, Set
from mobile import PROJECT_ROOT
from mobile.utils import *
from mobile.config import LOCATIONS, CLIENT_LOCATIONS, HOME_SHIFT
from mobile.heuristics import fpt
import json
import random
import time
import ray
import numpy as np
import scipy
import scipy.stats
import scipy.special
from joblib import Parallel, delayed

#print(len(LOCATIONS))
#print(LOCATIONS[0].keys())
#print(list(CLIENT_LOCATIONS.values())[0])

def simulate_dropped_out(q: float = 1):
    loc_pid_dropped = {}
    for c, loc in CLIENT_LOCATIONS.items():
        for l in loc:
            if random.random() <= q:
                if l not in loc_pid_dropped.keys():
                    loc_pid_dropped[l] = [c]
                else:
                    loc_pid_dropped[l].append(c)
    
    return loc_pid_dropped

def cover_approx_q(neighbors, k: int, q: float = 1):
    
    l = 0.1
    h = 10
    
    facilities = []
    objective = 10005
    
    #alpha = 0
    
    #for i in range(1, len(CLIENT_LOCATIONS)+1):
    #    alpha += 1/i
    
    #print(alpha)
    
    alpha = 1
    loc_pid_dropped = simulate_dropped_out(q)
    
    possible_locs = set(loc for loc in loc_pid_dropped.keys())
    new_neighbors = {}
    for loc, neighbor in neighbors.items():
        if loc in possible_locs:
            new_neighbors[loc] = []
            for n in neighbor:
                if n[0] <= h:
                    if n[1] in possible_locs:
                        new_neighbors[loc].append(n)
                else:
                    break
    
    while h-l > 1e-3:
        r = (l+h)/2
        
        print(r)
        
        sol = set_cover_softmax_q(new_neighbors, loc_pid_dropped, radius = r, top= 10, times = 40)
        
        if len(sol) <= alpha * k:
            h = (l+h)/2
            facilities = sol
            objective = r
        else:
            l = (l+h)/2
        
    return facilities, objective

def set_cover_softmax_q(neighbors, loc_pid_dropped, radius: float, top: int = 1, times: int = 1):

    radius_dict = {}

    for loc, neighbor in tqdm.tqdm(neighbors.items()):
        radius_dict[loc] = []
        for n in neighbor:
            if n[0] <= radius:
                ngbr = n[1]
                radius_dict[loc] += loc_pid_dropped[ngbr]
            else:
                break
    
    for loc, pids in tqdm.tqdm(radius_dict.items()):
        radius_dict[loc] = set(pids)
    
    total_length = len(set(i for v in loc_pid_dropped.values() for i in v))
    #radius_dict_id = ray.put(radius_dict)
    
    results = []
    #@ray.remote
    #def process(radius_dict):
        #print('starting process')
    for i in tqdm.tqdm(range(times)):
        covered = set()
        chosen = set()

        while len(covered) != total_length:

            max_coverage = []

            for loc in radius_dict.keys():

                if loc not in chosen:

                    individuals_covered = radius_dict[loc] - covered
                    max_coverage.append((len(individuals_covered), loc, individuals_covered))

            max_coverage = sorted(max_coverage, reverse = True)

            if max_coverage[0][0] == 0:
                break

            choice = max_coverage[scipy.stats.boltzmann.rvs(lambda_=0.8, N=top)]

            covered = covered.union(choice[2])
            chosen.add(choice[1])
            #nprint(len(covered))

        #return (len(chosen), chosen, covered)
        results.append((len(chosen), chosen, covered))

    #print("here")
    #results = [ray.get(process.remote(radius_dict_id)) for _ in range(times)]
    results = sorted(results)
    print(len(results[0][1]))
    
    return results[0][1]


with open(PROJECT_ROOT/'output'/'runs'/'charlottesville_city'/'neighbors.json', 'r') as f:
    neighbors = {int(key): val for key, val in json.load(f)['neighbors'].items()}

data = {}
'''for q in range(50, 101, 5):
    fac, radius_obj = cover_approx_q(neighbors, 5, q/100)
    assignments = assign_facilities(fac)
    obj = calculate_objective(assignments)
    print(q, fac, radius_obj, obj)
    data[q] = {"facilities": list(fac), "obj_value": obj, "radius": radius_obj}
    print(data)'''

q = 75
fac, radius_obj = cover_approx_q(neighbors, 10, q/100)
assignments = assign_facilities(fac)
obj = calculate_objective(assignments)
obj_95 = calculate_objective(assignments, 95)
data[q] = {"facilities": list(fac), "obj_value": obj, "obj_value_95": obj_95, "radius": radius_obj}
print(data)

with open(PROJECT_ROOT/ 'output'/ 'runs'/ 'charlottesville_city' / f'less_data_k_10_75.json', 'w') as f:
    json.dump(data, f)
    
    
    