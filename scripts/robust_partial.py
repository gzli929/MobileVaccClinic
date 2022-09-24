from mobile import PROJECT_ROOT
import json
import tqdm
from os import path
from mobile.utils import *
from mobile.heuristics import *
from mobile.config import LOCATIONS, CLIENT_LOCATIONS


def robust_k_supplier(clients: List[int], locations: List[int], k: int, p: int):
    
    l = 0
    r = 5

    to_ret = -1
    EPSILON = 10**(-4)
    
    disk_data = create_disk_data(clients, locations)
    
    latest_success = []
    
    while r-l > EPSILON:
    
        mid = l + (r - l) / 2
        
        norm_disks, expand_disks = filter_disks(disk_data, mid)
        is_success, selected_fac = _select_disks(norm_disks, expand_disks, k, p)
        
        if is_success:
            to_ret = mid
            latest_success = selected_fac
            r = mid
        else:
            l = mid
    
    return to_ret, latest_success

def create_disk_data(clients: List[int], locations:List[int]):
    disk_data = {}
    for l in locations:
        sorted_dist_clients = []
        for c in clients:
            dist = calculate_distance(l, c)
            sorted_dist_clients.append((dist, c))
        
        disk_data[l] = sorted(sorted_dist_clients)
        
    return disk_data

def filter_disks(disk_data, r: float):
    normal_disks = {}
    expanded_disks = {}
    for l in disk_data.keys():
        for dist, c in disk_data[l]:
            if dist <= r:
                if l in normal_disks:
                    normal_disks[l].append(c)
                else:
                    normal_disks[l] = [c]
            if dist <= 3*r:
                if l in expanded_disks:
                    expanded_disks[l].append(c)
                else:
                    expanded_disks[l] = [c]
    return normal_disks, expanded_disks

def _select_disks(norm_disks, expand_disks, k: int, p: int):
    
    selected_facilities = []
    covered = set()
    
    for i in range(k):
        max_add = (0, 0)
        for n in norm_disks:
            new_add = set(norm_disks[n])-covered
            if len(new_add) > max_add[0]:
                max_add = (len(new_add), n)
        
        if max_add[0] == 0:
            break
    
        chosen_fac = max_add[1]
        covered = covered.union(expand_disks[chosen_fac])
        selected_facilities.append(chosen_fac)
    
    return len(covered) >= p, selected_facilities

def partial_center_of_homes(k:int, partial:float):
    potential_facility_locations = [key for key in range(len(LOCATIONS)) if not LOCATIONS[key]['home']]
    homes = set(locs[0] for locs in CLIENT_LOCATIONS.values())
    
    radius, fac = robust_k_supplier(homes, potential_facility_locations, k, math.ceil(partial*len(homes)))
    return fac, assign_facilities(fac)


def partial_cover_approx(neighbors, k: int, partial):
    
    l = 0.025
    h = 3
    
    facilities = []
    objective = 10005
    
    alpha = 1
    
    while h-l > 1e-3:
        r = (l+h)/2
        
        sol = partial_set_cover_softmax(neighbors, alpha*k, partial, radius = r)
        
        print(r, len(sol))
        
        if len(sol) <= alpha * k and len(sol)!=0:
            h = (l+h)/2
            facilities = sol
            objective = r
        else:
            l = (l+h)/2
        
    return facilities, objective

def partial_set_cover_softmax(neighbors, k, partial:float, radius: float, top: int = 1, times: int = 1):

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

'''
data = {3:{"facilities":[73, 1035, 249], "assignments": assign_facilities([73, 1035, 249]), "obj_95": 2.9028132394187462},
        4:{"facilities":[688, 136, 872, 470], "assignments": assign_facilities([688, 136, 872, 470]), "obj_95": 3.0136841784872073},
        5:{"facilities":[308, 369, 621, 470, 398], "assignments": assign_facilities([308, 369, 621, 470, 398]), "obj_95": 3.073129129907178},
        6:{"facilities":[308, 558, 861, 1844, 621, 398], "assignments": assign_facilities([308, 558, 861, 1844, 621, 398]), "obj_95":2.639562170000805},
        7:{"facilities":[308, 1035, 38117, 603, 621, 578, 874], "assignments": assign_facilities([308, 1035, 38117, 603, 621, 578, 874]), "obj_95": 2.58830365785502},
        8:{"facilities":[688, 315, 1347, 861, 142, 578, 874, 696], "assignments": assign_facilities([688, 315, 1347, 861, 142, 578, 874, 696]), "obj_95":2.3598801260648345},
        9:{"facilities":[133, 30398, 38225, 7494, 236, 2542, 367, 38130, 909], "assignments":assign_facilities([133, 30398, 38225, 7494, 236, 2542, 367, 38130, 909]), "obj_95":1.61503058313516},
        10:{"facilities":[572, 30398, 1635, 7494, 1844, 1094, 367, 536, 38130, 594], "assignments": assign_facilities([572, 30398, 1635, 7494, 1844, 1094, 367, 536, 38130, 594]), "obj_95":1.5784857437016278}}
    


for i in range(3, 21):
    fac, assign = partial_center_of_homes(i, .95)
    obj_val = calculate_objective(assign, 95)
    print(i, fac, obj_val)
    
    data[i] = {"facilities": list(fac), "assignments": assign, "obj_95": obj_val}
    
with open(PROJECT_ROOT/"output"/"runs"/"albemarle"/"partial_center_tradeoff.json", "w") as f:
    json.dump(data, f)

'''
county_name = "albemarle"
with open(PROJECT_ROOT/'output'/'runs'/county_name/'neighbors.json', 'r') as f:
    neighbors = {int(key): val for key, val in json.load(f)['neighbors'].items()}

data = {}

for i in range(11, 21):
    fac, obj_radius = partial_cover_approx(neighbors, i, .95)
    assign = assign_facilities(fac)
    obj_val = calculate_objective(assign, 95)
    print(i, fac, obj_val)
    
    data[i] = {"facilities": list(fac), "assignments": assign, "obj_95": obj_val}
    
    
with open(PROJECT_ROOT/"output"/"runs"/county_name/"partial_cover_tradeoff_2.json", "w") as f:
    json.dump(data, f)
