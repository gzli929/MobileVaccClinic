from mobile import PROJECT_ROOT
import json
import tqdm
from os import path
from mobile.utils import *
from mobile.heuristics import cover_approx
from mobile.config import aggregate_data

county_name = "charlottesville_city"

LOCATIONS, CLIENT_LOCATIONS = aggregate_data(county_name = county_name, aggregation = 1, radius = 0.025, home_work_only=False)

neighbors_parallel = generate_sorted_list_parallel(LOCATIONS)

run_storage = {}
k_limits = {
    3: (1, 2.5),
    4: (1, 2.5),
    5: (1, 2.5),
    6: (1, 2),
    7: (1, 2),
    8: (0.5, 1.5),
    9: (0.5, 1.5),
    10: (0.5, 1.5)
}

for k in range(3, 11):
    facilities, obj = cover_approx(LOCATIONS, CLIENT_LOCATIONS, neighbors_parallel, k, lower_bound=k_limits[k][0], upper_bound=k_limits[k][1], top = 10, times=10)
    print(k, obj)
    assignment = assign_facilities(LOCATIONS, CLIENT_LOCATIONS, facilities)
    opt_obj = calculate_objective(LOCATIONS, assignment)
    
    run_storage[k] = {"facilities": list(facilities), "assignment": list(assignment), "objective": opt_obj}
    

with open(PROJECT_ROOT/'output'/'runs'/county_name/f"full_cville_cover.json", 'w') as f:
    json.dump(run_storage, f)