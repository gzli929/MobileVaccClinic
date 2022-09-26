from mobile.config import aggregate_data
from mobile.utils import *
from mobile.heuristics import *
from mobile import PROJECT_ROOT

import json

county_name = "charlottesville_city"

LOCATIONS, CLIENT_LOCATIONS = aggregate_data(county_name = county_name, aggregation = 1, radius = 0.025, home_work_only=False)

"""
Run budget tradeoff for HomeCenter, MostActivity, and FPT (cover done separately in cville_covering_full.py script)
"""

data = {}

for alg in [center_of_homes, most_populous]:
    data[alg.__name__] = []
    
    for num_facility in range(3,11):
        (fac, asgn) = alg(LOCATIONS, CLIENT_LOCATIONS, num_facility)
        obj = calculate_objective(LOCATIONS, asgn, 95)
        
        data[alg.__name__].append([fac, asgn, obj])
         
data['fpt20'] = []

for k in range(3,11):
    (min_obj_guess, asgn) = fpt(LOCATIONS, CLIENT_LOCATIONS, k, 20)
    fac = min_obj_guess[1]
    obj = calculate_objective(LOCATIONS, asgn, 95)
    data['fpt20'].append([fac, asgn, obj])


with open(PROJECT_ROOT / 'output' / 'runs' / county_name / 'tradeoff.json', 'w') as f:
    json.dump(data, f)