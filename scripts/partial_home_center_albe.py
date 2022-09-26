from mobile import PROJECT_ROOT
import json
from mobile.utils import *
from mobile.heuristics import *
from mobile.config import aggregate_data

county_name = "albemarle"

LOCATIONS, CLIENT_LOCATIONS = aggregate_data(county_name = county_name, aggregation = 1, radius = 0.025, home_work_only=False)

"""
Partial HomeCenter
"""
def robust_k_supplier(LOCATIONS, clients: List[int], locations: List[int], k: int, p: int):
    
    l = 0
    r = 8

    to_ret = -1
    EPSILON = 10**(-4)
    
    disk_data = create_disk_data(LOCATIONS, clients, locations)
    
    latest_success = []
    
    while r-l > EPSILON:
    
        mid = l + (r - l) / 2
        
        norm_disks, expand_disks = filter_disks(disk_data, mid)
        is_success, selected_fac = _select_disks(norm_disks, expand_disks, k, p)
        
        print(mid, is_success)
        
        if is_success:
            to_ret = mid
            latest_success = selected_fac
            r = mid
        else:
            l = mid
    
    return to_ret, latest_success

def create_disk_data(LOCATIONS, clients: List[int], locations:List[int]):
    disk_data = {}
    for l in locations:
        sorted_dist_clients = []
        for c in clients:
            dist = calculate_distance(LOCATIONS, l, c)
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

def partial_center_of_homes(LOCATIONS, CLIENT_LOCATIONS, k:int, partial:float):
    potential_facility_locations = [key for key in range(len(LOCATIONS)) if not LOCATIONS[key]['home']]
    homes = set(locs[0] for locs in CLIENT_LOCATIONS.values())
    
    radius, fac = robust_k_supplier(LOCATIONS, homes, potential_facility_locations, k, math.ceil(partial*len(homes)))
    return fac, assign_facilities(LOCATIONS, CLIENT_LOCATIONS, fac)


data = {}

for i in range(5, 21):
    fac, assign = partial_center_of_homes(LOCATIONS, CLIENT_LOCATIONS, i, .95)
    obj_val = calculate_objective(LOCATIONS, assign, 95)
    print(i, fac, obj_val)
    
    data[i] = {"facilities": list(fac), "assignments": assign, "obj_95": obj_val}
    
    
with open(PROJECT_ROOT/"output"/"runs"/ county_name /"partial_center_tradeoff.json", "w") as f:
    json.dump(data, f)
