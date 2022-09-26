from mobile import PROJECT_ROOT
import json
import tqdm
from os import path
from mobile.utils import *
from mobile.heuristics import cover_approx
from mobile.config import aggregate_data

county_name = "albemarle"

LOCATIONS, CLIENT_LOCATIONS = aggregate_data(county_name = county_name, aggregation = 1, radius = 0.025, home_work_only=False)

neighbors_parallel = generate_sorted_list_parallel(LOCATIONS)

run_storage = {}
k_limits = {
    6: (9, 12),
    7: (7, 10),
    8: (7, 10),
    9: (5, 8),
    10: (5, 8),
    11: (5, 7.5),
    12: (4, 7),
    13: (4, 6.5),
    14: (3, 8),
    15: (3, 5.5),
    16: (2.5, 5.5),
    17: (2.5, 5),
    18: (2.5, 5),
    19: (2.5, 4.5),
    20: (2.5, 4.5)
}

for k in range(6, 21):
    facilities, obj = cover_approx(LOCATIONS, CLIENT_LOCATIONS, neighbors_parallel, k, lower_bound=k_limits[k][0], upper_bound=k_limits[k][1])
    print(k, obj)
    assignment = assign_facilities(LOCATIONS, CLIENT_LOCATIONS, facilities)
    opt_obj = calculate_objective(LOCATIONS, assignment)
    
    run_storage[k] = {"facilities": list(facilities), "assignment": list(assignment), "objective": opt_obj}
    

with open(PROJECT_ROOT/'output'/'runs'/county_name/f"full_albe_cover.json", 'w') as f:
    json.dump(run_storage, f)