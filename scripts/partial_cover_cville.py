from mobile import PROJECT_ROOT
import json
import tqdm
from mobile.utils import *
from mobile.heuristics import *
from mobile.config import aggregate_data

import scipy.stats

county_name = "charlottesville_city"

LOCATIONS, CLIENT_LOCATIONS = aggregate_data(county_name = county_name, aggregation = 1, radius = 0.025, home_work_only=False)
neighbors_parallel = generate_sorted_list_parallel(LOCATIONS)


"""
Partial ClientCover
"""
def partial_cover_approx(LOCATIONS, CLIENT_LOCATIONS, neighbors, k: int, partial):
    
    l = 0.01
    h = 3
    
    facilities = []
    objective = 10005
    
    alpha = 1
    
    while h-l > 1e-3:
        r = (l+h)/2
        
        sol = partial_set_cover_softmax(LOCATIONS, CLIENT_LOCATIONS, neighbors, alpha*k, partial, radius = r)
        
        print(r, len(sol))
        
        if len(sol) <= alpha * k and len(sol)!=0:
            h = (l+h)/2
            facilities = sol
            objective = r
        else:
            l = (l+h)/2
        
    return facilities, objective

def partial_set_cover_softmax(LOCATIONS, CLIENT_LOCATIONS, neighbors, k, partial:float, radius: float, top: int = 1, times: int = 1):

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
    results = []
    
    for i in range(times):
        covered = set()
        chosen = set()
        finished = True

        while len(covered) < partial*total_length:
            
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

data = {}
for i in range(3, 11):
    fac, obj = partial_cover_approx(LOCATIONS, CLIENT_LOCATIONS, neighbors_parallel, i, .95)
    assign = assign_facilities(LOCATIONS, CLIENT_LOCATIONS, fac)
    obj_val = calculate_objective(LOCATIONS, assign, 95)
    print(i, fac, obj_val)
    
    data[i] = {"facilities": list(fac), "assignments": assign, "obj_95": obj_val}

with open(PROJECT_ROOT/"output"/"runs"/county_name/"partial_cover_tradeoff.json", "w") as f:
    json.dump(data, f)
