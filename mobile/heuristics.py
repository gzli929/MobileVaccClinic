import random
from typing import Dict, List, Tuple, Set
from mobile.utils import *
import time
import ray
import numpy as np
import scipy
import scipy.stats
import scipy.special
from joblib import Parallel, delayed

"""
ClientCover Algorithm
"""
def cover_approx(LOCATIONS, CLIENT_LOCATIONS, neighbors, k: int, top = 1, times = 1, lower_bound = 0.1, upper_bound = 8):
    """
    Performs binary search over the objective/radius search space and applies greedy set cover
    
    PARAMETERS
    ----------
    LOCATIONS: List[Dict[str, T]]
    CLIENT_LOCATIONS: Dict[int, List[int]]
    neighbors: Dict[int, List[List[float, int]]]
        dictionary mapping a location in LOCATIONS to an asecending sorted List of its distance to all other locations
    k : int
        number of facilities to be opened
    
    RETURNS
    ----------
    facilities : List[int]
        contains facility indices that are to be opened and placed
    objective: float
        the radius chosen by binary search
    """
    
    l = lower_bound
    h = upper_bound
    
    facilities = []
    objective = 10005
    
    alpha = 1
    
    while h-l > 1e-3:
        r = (l+h)/2
        
        sol = set_cover_softmax(LOCATIONS, CLIENT_LOCATIONS, neighbors, alpha*k, radius = r, top = top, times = times)
        
        print(r, len(sol))
        
        if len(sol) <= alpha * k and len(sol)!=0:
            h = (l+h)/2
            facilities = sol
            objective = r
        else:
            l = (l+h)/2
        
    return facilities, objective

def set_cover_softmax(LOCATIONS, CLIENT_LOCATIONS, neighbors, k, radius: float, top: int = 1, times: int = 1):
    """
    Helper algorithm for ClientCover that can take in the argument top and times and select from the top choices instead of only selecting the maximum for greedy
    """

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

"""
FPT Algorithm
"""
def fpt(LOCATIONS, CLIENT_LOCATIONS, k: int, s: int):
    """
    Picks the s activity locations that cover the most clients (through a set cover approximation)
    Assumes the number of locations visited by clients is bounded by a constant
    Run k-supplier on all combination sets of locations that will be covered by facilities. Select the guess and its open facilities with the smallest objective value.
    
    PARAMETERS
    ----------
    LOCATIONS: List[Dict[str, T]]
    CLIENT_LOCATIONS: Dict[int, List[int]]
    k : int
        number of facilities to be opened
    s : int
        number of activity locations examined
    
    RETURNS
    ----------
    facilities : List[int]
        contains facility indices that are open
    assignments : List[Tuple[int, int]]
        list of the visited location and opened facility pair that produces the smallest pairwise distance for each client
    """
    
    potential_facility_locations = cover_most(LOCATIONS, CLIENT_LOCATIONS, s)
    
    #Remove homes from the client_location lists
    client_locations_excluded = []
    for person in CLIENT_LOCATIONS.values():
        new_list = [p for p in person[1:] if p in potential_facility_locations]
        if len(new_list)>0:
            client_locations_excluded.append(new_list)
    
    locations = [i for i in range(len(LOCATIONS)) if not LOCATIONS[i]['home']]
    
    G, loc_map, c_loc_map = precompute_distances(LOCATIONS, client_locations_excluded, locations)
    
    ray.init(ignore_reinit_error=True)
    
    @ray.remote
    def process(guess):
        facilities = k_supplier(LOCATIONS, list(guess), locations, k)
        obj_value = assign_client_facilities(G, loc_map, c_loc_map, client_locations_excluded, facilities)
        return obj_value, facilities
    
    futures = [process.remote(guess) for guess in powerset(list(potential_facility_locations))]
    results = ray.get(futures)

    min_obj_guess: Tuple[int, List[int]] = min(results)
    return min_obj_guess, assign_facilities(LOCATIONS, CLIENT_LOCATIONS, min_obj_guess[1])


"""
HomeCenter Heuristic
"""
def center_of_homes(LOCATIONS, CLIENT_LOCATIONS, k: int):
    """
    Opens facilities based only on home-locations for each client
    
    PARAMETERS
    ----------
    LOCATIONS: List[Dict[str, T]]
    CLIENT_LOCATIONS: Dict[int, List[int]]
    k : int
        number of facilities to be opened
    
    RETURNS
    ----------
    facilities : List[int]
        contains facility indices that are open
    assignments : List[Tuple[int, int]]
        list of the visited location and opened facility pair that produces the smallest pairwise distance for each client
    """
    
    potential_facility_locations = [key for key in range(len(LOCATIONS)) if not LOCATIONS[key]['home']]
    homes = set(locs[0] for locs in CLIENT_LOCATIONS.values())
    
    facilities = k_supplier(LOCATIONS, list(homes), potential_facility_locations, k)
    
    return facilities, assign_facilities(LOCATIONS, CLIENT_LOCATIONS, facilities)


"""
MostActivity Heuristic
"""
def most_populous(LOCATIONS, CLIENT_LOCATIONS, k: int):
    """
    Selects the top k potential facilities that have the most unique pid visitors
    
    PARAMETERS
    ----------
    LOCATIONS: List[Dict[str, T]]
    CLIENT_LOCATIONS: Dict[int, List[int]]
    k: int
        number of facilities to be opened
    
    RETURNS
    ---------
    facilities : List[int]
        contains facility indices that are open
    assignments : List[Tuple[int, int]]
        list of the visited location and opened facility pair that produces the smallest pairwise distance for each client
    """
    
    facilities = [i for i in range(k)]
    return facilities, assign_facilities(LOCATIONS, CLIENT_LOCATIONS, facilities)
