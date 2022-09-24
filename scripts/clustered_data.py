from mobile import PROJECT_ROOT
import json
import tqdm
from os import path
from mobile.utils import *
from mobile.heuristics import *
from mobile.config import LOCATIONS, CLIENT_LOCATIONS

county_name = "charlottesville_city"

with open(PROJECT_ROOT/'output'/'runs'/county_name/'neighbors.json', 'r') as f:
    neighbors = {int(key): val for key, val in json.load(f)['neighbors'].items()}

def generate_new_information(neighbors, radius):
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
        
    '''coverage_matching = {}
    for cluster in cluster_dict.keys():
        overlap = [member for member in cluster_dict[cluster] if member in chosen_points]
        coverage_matching[cluster] = overlap'''
    
    loc_pid_clustered = {}
    for point in chosen_points:
        pids_to_loc = LOCATIONS[point]['pid']
        for l in cluster_dict[point]:
            pids_to_loc += LOCATIONS[l]['pid']
        loc_pid_clustered[point] = set(pids_to_loc)
    for point in range(len(LOCATIONS)):
        if LOCATIONS[point]['home']:
            loc_pid_clustered[point] = set(LOCATIONS[point]['pid'])
    
    '''client_loc_clustered = {}
    for c in CLIENT_LOCATIONS:
        new_client = []
        for loc in CLIENT_LOCATIONS[c]:
            if LOCATIONS[loc]['home']:
                new_client += [loc]
            else:
                new_client += coverage_matching[loc]
        client_loc_clustered[c] = set(new_client)'''
    
    return chosen_points, loc_pid_clustered


def cover_approx_cluster(neighbors, k: int, cluster_radius: float, loc_pid_clustered, lower_bound, upper_bound):
    
    l = lower_bound
    h = upper_bound
    
    facilities = []
    objective = 10005
    
    alpha = 1
    
    while h-l > 1e-3:
        r = (l+h)/2
        
        #sol = set_cover_softmax_cluster(neighbors, loc_pid_clustered, radius = r)
        #top=10, times = 40
        sol = set_cover_softmax_cluster(neighbors, loc_pid_clustered, alpha*k, radius = r)
        # top=10, times = 40
        print(r, len(sol))
        
        if len(sol) <= alpha * k and len(sol)!=0:
            h = (l+h)/2
            facilities = sol
            objective = r
        else:
            l = (l+h)/2
        
    return facilities, objective

def set_cover_softmax_cluster(neighbors, loc_pid_clustered, k, radius: float, top: int = 1, times: int = 1):

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

def cover_most_cluster(s: int, loc_pid_clustered):
    """
    Helper method for FPT: returns the set of activity locations of size s that cover the most clients
    Used with aggregate activity locations
    
    aggregation : int
    the version of aggregation selected
    0 --> none
    1 --> set cover: aggregation with repeats in coverage
    """
    
    covered = set()
    selected = []
    for i in range(s):
        most_coverage = max([(len(loc_pid_clustered[l] - covered), l) for l in chosen_points])
        selected.append(most_coverage[1])
        covered = covered.union(loc_pid_clustered[most_coverage[1]])
    print(f"COVERAGE OF CLIENTS BY {s} LOCATIONS: ", len(covered)/len(CLIENT_LOCATIONS.keys()))
    return selected

def fpt_cluster(k: int, s: int, loc_pid_clustered, chosen_points, client_loc_clustered):
    
    potential_facility_locations = cover_most_cluster(s, loc_pid_clustered)
    
    #Remove homes from the client_location lists
    #TODO: Perhaps create mapping for the indices of people before exclusion and after?
    client_locations_excluded = []
    for person in client_loc_clustered.values():
        new_list = [p for p in person if p in potential_facility_locations]
        if len(new_list)>0:
            client_locations_excluded.append(new_list)
    
    locations = list(chosen_points)
    
    G, loc_map, c_loc_map = precompute_distances(client_locations_excluded, locations)
    
    ray.init(ignore_reinit_error=True)
    
    @ray.remote
    def process(guess):
        facilities = k_supplier(list(guess), locations, k)
        obj_value = assign_client_facilities(G, loc_map, c_loc_map, client_locations_excluded, facilities)
        return obj_value, facilities
    
    futures = [process.remote(guess) for guess in powerset(list(potential_facility_locations))]
    results = ray.get(futures)

    min_obj_guess: Tuple[int, List[int]] = min(results)
    return min_obj_guess, assign_facilities(min_obj_guess[1])

bounds = {0.1: (0.5, 1.5), 0.2: (0.5, 1.5), 0.3:(0.5, 1.5), 0.4:(0.5, 1.5), 0.5:(0.25, 1), 0.6:(0.25, 1),
            0.15: (0.5, 1.5), 0.25:(0.5, 1.25), 0.35:(0.5, 1.25)}

k = 10
cluster_radius_fac = {}
#30
for i in [45, 55]:
    cluster_radius = i/100
    
    filename = PROJECT_ROOT/'output'/'runs'/county_name/f'cluster_data_{cluster_radius}.json'
    
    if path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            cluster_dict = data['cluster_dict']
            chosen_points = set(data['chosen_points'])
            coverage_matching = {int(key): value for key,value in data['coverage_matching'].items()}
            loc_pid_clustered = {int(key): set(value) for key,value in data['loc_pid_clustered'].items()}
            client_loc_clustered = {int(key): set(value) for key,value in data['client_loc_clustered'].items()}
            new_neighbors = {int(key): value for key,value in data['new_neighbors'].items()}
    
    else:
        chosen_points, loc_pid_clustered = generate_new_information(neighbors, cluster_radius)
    
        print("New Neighbors")
        new_neighbors = {}
        for n, val in tqdm.tqdm(neighbors.items()):
            if n in chosen_points:
                new_val = [v for v in val if v[1] in chosen_points or LOCATIONS[v[1]]['home']]
                new_neighbors[n] = new_val
        
        '''with open(filename, "w") as f:
            json.dump({"cluster_dict":cluster_dict, "chosen_points": list(chosen_points), "coverage_matching": coverage_matching,
                        "loc_pid_clustered": {key: list(val) for key, val in loc_pid_clustered.items()},
                        "client_loc_clustered": {key: list(val) for key,val in client_loc_clustered.items()}, "new_neighbors": new_neighbors}, f)'''
        
    if cluster_radius in bounds.keys():
        lower = bounds[cluster_radius][0]
        upper = bounds[cluster_radius][1]
    else:
        lower = 0.1
        upper = 2.5
    
    fac, obj = cover_approx_cluster(new_neighbors, k, cluster_radius, loc_pid_clustered, lower, upper)
    
    assign = assign_facilities(fac)
    objective_value = calculate_objective(assign)
    print(cluster_radius, fac, obj, objective_value)
    
    '''fac, assign = fpt_cluster(k, 15, loc_pid_clustered, chosen_points, client_loc_clustered)
    cluster_radius_fac[cluster_radius] = {"k": k, "facilities": list(fac), "assignments": assign}'''
    
    cluster_radius_fac[cluster_radius] = {"k": k, "facilities": list(fac), "obj_radius": obj, "objective_value": objective_value, "assignments": assign}
    
    '''fac, assign = most_populous_cluster(10, loc_pid_clustered)
    data[cluster_radius] = {"facilities": fac, "assignments": assign}
    print(cluster_radius, calculate_objective(assign))'''
    
    for radius, info in cluster_radius_fac.items():
        print(radius, info["facilities"], info["obj_radius"],  info["objective_value"])

with open(PROJECT_ROOT/'output'/'runs'/county_name/f'cluster_exp_smooth12_{k}.json', 'w') as f:
    json.dump(cluster_radius_fac, f)