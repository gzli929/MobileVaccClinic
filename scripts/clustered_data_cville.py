from mobile import PROJECT_ROOT
import json
import tqdm
from os import path
from mobile.utils import *
from mobile.heuristics import *
from mobile.config import aggregate_data

county_name = "charlottesville_city"

LOCATIONS, CLIENT_LOCATIONS = aggregate_data(county_name = county_name, aggregation = 1, radius = 0.025, home_work_only=False)

def generate_new_information(LOCATIONS, CLIENT_LOCATIONS, neighbors, radius):
    
    print("Clustering")
    
    cluster_dict = {}
    for n, ngbrs in tqdm.tqdm(neighbors.items()):
        index = 0
        cluster_value = []
        while index < len(ngbrs) and ngbrs[index][0] <= radius:
            cluster_value.append(ngbrs[index][1])
            index+=1
        cluster_dict[n] = cluster_value
    
    print("Choosing")
    
    cover = set()
    chosen_points = set()
    while len(cover) < len(LOCATIONS):
        max_choice = (0, set(), -1)
        for key, val in cluster_dict.items():
            if key not in chosen_points:
                set_choice = set(val) - cover
                if len(set_choice)>max_choice[0]:
                    max_choice = (len(set_choice), set_choice, key)
        if max_choice[0] == 0:
            print("not completely covered")
            break
        cover = cover.union(max_choice[1])
        chosen_points.add(max_choice[2])
    
    loc_pid_clustered = {}
    for point in chosen_points:
        pids_to_loc = LOCATIONS[point]['pid']
        for l in cluster_dict[point]:
            pids_to_loc += LOCATIONS[l]['pid']
        loc_pid_clustered[point] = set(pids_to_loc)
    for point in range(len(LOCATIONS)):
        if LOCATIONS[point]['home']:
            loc_pid_clustered[point] = set(LOCATIONS[point]['pid'])

    
    return chosen_points, loc_pid_clustered


def cover_approx_cluster(LOCATIONS, CLIENT_LOCATIONS, neighbors, k: int, cluster_radius: float, loc_pid_clustered, lower_bound, upper_bound):
    
    l = lower_bound
    h = upper_bound
    
    facilities = []
    objective = 10005
    
    alpha = 1
    
    while h-l > 1e-3:
        r = (l+h)/2
        
        sol = set_cover_softmax_cluster(LOCATIONS, CLIENT_LOCATIONS, neighbors, loc_pid_clustered, alpha*k, radius = r)
        # top=10, times = 40
        print(r, len(sol))
        
        if len(sol) <= alpha * k and len(sol)!=0:
            h = (l+h)/2
            facilities = sol
            objective = r
        else:
            l = (l+h)/2
        
    return facilities, objective

def set_cover_softmax_cluster(LOCATIONS, CLIENT_LOCATIONS, neighbors, loc_pid_clustered, k, radius: float, top: int = 1, times: int = 1):

    radius_dict = {}

    for loc, neighbor in tqdm.tqdm(neighbors.items()):
        temp = []
        for n in neighbor:
            if n[0] <= radius:
                ngbr = n[1]
                temp += list(loc_pid_clustered[ngbr])
            else:
                break
        radius_dict[loc] = set(temp)
    
    total_length = len(CLIENT_LOCATIONS)
    #radius_dict_id = ray.put(radius_dict)
    
    results = []

    for i in range(times):
        covered = set()
        chosen = set()
        finished = True

        while len(covered) != total_length:
            
            if len(chosen)>k:
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


bounds = {0.1: (0.5, 1.5), 0.2: (0.5, 1.5), 0.3:(0.5, 1.5), 0.4:(0.5, 1.5), 0.5:(0.25, 1), 0.6:(0.25, 1),
            0.15: (0.5, 1.5), 0.25:(0.5, 1.25), 0.35:(0.5, 1.25), 0.45:(0.25, 1), 0.55:(0.25, 1)}

k = 10
cluster_radius_fac = {}
neighbors = generate_sorted_list_parallel(LOCATIONS)

for i in range(10, 63):
    cluster_radius = i/100

    chosen_points, loc_pid_clustered = generate_new_information(LOCATIONS, CLIENT_LOCATIONS, neighbors, cluster_radius)

    print("New Neighbors")
    new_neighbors = {}
    for n, val in tqdm.tqdm(neighbors.items()):
        if n in chosen_points:
            new_val = [v for v in val if v[1] in chosen_points or LOCATIONS[v[1]]['home']]
            new_neighbors[n] = new_val

    if cluster_radius in bounds.keys():
        lower = bounds[cluster_radius][0]
        upper = bounds[cluster_radius][1]
    else:
        lower = 0.1
        upper = 1.5
    
    fac, obj = cover_approx_cluster(LOCATIONS, CLIENT_LOCATIONS, new_neighbors, k, cluster_radius, loc_pid_clustered, lower, upper)
    
    assign = assign_facilities(LOCATIONS, CLIENT_LOCATIONS, fac)
    objective_value = calculate_objective(LOCATIONS, assign)
    print(cluster_radius, fac, obj, objective_value)

    cluster_radius_fac[cluster_radius] = {"k": k, "facilities": list(fac), "obj_radius": obj, "objective_value": objective_value, "assignments": assign}
    
    for radius, info in cluster_radius_fac.items():
        print(radius, info["facilities"], info["obj_radius"],  info["objective_value"])

with open(PROJECT_ROOT/'output'/'runs'/county_name/f'cluster_exp_cover_full.json', 'w') as f:
    json.dump(cluster_radius_fac, f)
