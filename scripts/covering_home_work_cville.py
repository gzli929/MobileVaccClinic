import random
import json
from typing import Dict, List, Tuple, Set
from mobile import PROJECT_ROOT
from mobile.utils import *
from mobile.config import aggregate_data
from mobile.heuristics import *

#print("STARTING...")

county_name = "charlottesville_city"

LOCATIONS, CLIENT_LOCATIONS = aggregate_data(county_name = county_name, aggregation = 1, radius = 0.025, home_work_only=False)
LOCATIONS_limited, CLIENT_LOCATIONS_limited = aggregate_data(county_name = county_name, aggregation = 1, radius = 0.025, home_work_only=True)

neighbors_parallel = generate_sorted_list_parallel(LOCATIONS_limited)

def calculate_distance_limited(LOCATIONS_limited, LOCATIONS, loc, fac):
    # fac comes from the locations_limited set of data
    
    coord1_row = LOCATIONS[loc]
    coord2_row = LOCATIONS_limited[fac]
    coord1 = (coord1_row['latitude'], coord1_row['longitude'])
    coord2 = (coord2_row['latitude'], coord2_row['longitude'])
    return geopy.distance.great_circle(coord1, coord2).km


def calculate_objective_limited(LOCATIONS_limited, LOCATIONS, assignments, percentile:float=100):
    if len(assignments) == 0: return 0
    
    obj_val = sorted([calculate_distance_limited(LOCATIONS_limited, LOCATIONS, loc, fac) for loc, fac in assignments])
    ind = math.floor(len(obj_val)*percentile/100) -1
    
    #If no clients are selected to be covered, then the objective is 0
    if ind < 0: return 0
    
    return obj_val[ind]

def assign_facilities_limited(LOCATIONS_limited, LOCATIONS, CLIENT_LOCATIONS, facilities):
    
    if len(facilities) == 0: return []
    
    assignments: List[Tuple[int, int]] = []
    
    for key in CLIENT_LOCATIONS.keys():
        possible_assignments = [(calculate_distance_limited(LOCATIONS_limited, LOCATIONS, loc, fac), loc, fac) for loc in CLIENT_LOCATIONS[key] for fac in facilities]
        
        min_loc = min(possible_assignments)
        assignments.append((min_loc[1], min_loc[2]))
   
    return assignments


runs = {}
k_limits = {
    3: (2, 4),
    4: (2, 4),
    5: (2, 3.5),
    6: (1.5, 2.5),
    7: (1, 2.5),
    8: (0.5, 2),
    9: (0.5, 2),
    10: (0.5, 2)
}

for k in range(8, 9):
    facilities, objective = cover_approx(LOCATIONS_limited, CLIENT_LOCATIONS_limited, neighbors_parallel, k, lower_bound = k_limits[k][0], upper_bound = k_limits[k][1], top=10, times=10)
    print(k, objective)
    assignment_full = assign_facilities_limited(LOCATIONS_limited, LOCATIONS, CLIENT_LOCATIONS, facilities)
    true_objective = calculate_objective_limited(LOCATIONS_limited, LOCATIONS, assignment_full)
    runs[k] = {"facilities": list(facilities), "assignments": list(assignment_full), "objective": true_objective}


with open(PROJECT_ROOT/'output'/'runs'/county_name/f"cover_home_work_3.json", 'w') as f:
    json.dump(runs, f)

