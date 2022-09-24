from mobile import PROJECT_ROOT
import json
import tqdm
from os import path
from mobile.utils import *
from mobile.heuristics import *
from mobile.config import LOCATIONS, CLIENT_LOCATIONS

county_name = "albemarle"

with open(PROJECT_ROOT/'output'/'runs'/county_name/'neighbors.json', 'r') as f:
    neighbors = {int(key): val for key, val in json.load(f)['neighbors'].items()}

def cover_approx_k(neighbors, k: int, lower_bound: float, upper_bound: float):
    
    l = lower_bound
    h = upper_bound
    
    facilities = []
    objective = 10005
    
    alpha = 1
    
    while h-l > 1e-3:
        r = (l+h)/2
        
        sol = set_cover_softmax_k(neighbors, alpha*k, radius = r)
        #sol = set_cover_softmax_k(neighbors, alpha*k, radius = r)
        
        print(r, len(sol))
        
        if len(sol) <= alpha * k and len(sol)!=0:
            h = (l+h)/2
            facilities = sol
            objective = r
        else:
            l = (l+h)/2
        
    return facilities, objective

def set_cover_softmax_k(neighbors, k, radius: float, top: int = 1, times: int = 1):

    radius_dict = {}

    for l, neighbor in tqdm.tqdm(neighbors.items()):

        temp = []

        for n in neighbor:

            if n[0] <= radius:                
                ngbr = n[1]
                temp += list(LOCATIONS[ngbr]['pid'])
            else:
                break
        
        radius_dict[l] = set(temp)
    
    total_length = len(CLIENT_LOCATIONS)
    #radius_dict_id = ray.put(radius_dict)
    
    results = []
    #@ray.remote
    #def process(radius_dict):
        #print('starting process')
    
    for i in range(times):
        covered = set()
        chosen = set()
        finished = True

        while len(covered) != total_length:
            
            if len(chosen) > k:
                finished = False
                break
            
            max_coverage = []

            for loc in radius_dict.keys():

                if loc not in chosen:

                    individuals_covered = radius_dict[loc] - covered
                    max_coverage.append((len(individuals_covered), loc, individuals_covered))

            max_coverage = sorted(max_coverage, reverse = True)

            if max_coverage[0][0] == 0:
                finished = False
                break

            choice = max_coverage[scipy.stats.boltzmann.rvs(lambda_=0.8, N=top)]

            covered = covered.union(choice[2])
            chosen.add(choice[1])
            #print(len(covered))

        #return (len(chosen), chosen, covered)
        if finished:
            results.append((len(chosen), chosen, covered))


    if len(results) == 0:
        return []
    
    results = sorted(results)
    return results[0][1]


k = 20
lower_bound = 2
upper_bound = 4


print(k)

fac, obj = cover_approx_k(neighbors, k, lower_bound, upper_bound)
with open(PROJECT_ROOT/'output'/'runs'/county_name/f"full_albe_cover_{k}.json", 'w') as f:
    json.dump({"k": k, "facilities": list(fac), "obj_radius": obj, "assignments": assign_facilities(fac)}, f)

#k = 5
#lower_bound = 9
#upper_bound = 12

'''
k = 6
lower_bound = 8.5
upper_bound = 11

k = 7
lower_bound = 8.5
upper_bound = 11

k = 8
lower_bound = 6
upper_bound = 10

k = 9
lower_bound = 6
upper_bound = 8

k = 10
lower_bound = 5
upper_bound = 7

k = 11
lower_bound = 5
upper_bound = 6.5

k = 12
lower_bound = 5
upper_bound = 6.5

k = 13
lower_bound = 5
upper_bound = 6

k = 14
lower_bound = 5
upper_bound = 6

k = 15
lower_bound = 4
upper_bound = 5

k = 16
lower_bound = 4
upper_bound = 5

k = 17
lower_bound = 4
upper_bound = 5

k = 18
lower_bound = 3.5
upper_bound = 4.5

k = 19
lower_bound = 3
upper_bound = 4

k = 20
lower_bound = 2.5
upper_bound = 3.5
'''
