from mobile.config import LOCATIONS, CLIENT_LOCATIONS, HOME_SHIFT
from mobile.utils import *
from mobile.heuristics import *
from mobile import PROJECT_ROOT

import numpy as np
import scipy
import scipy.stats
import scipy.special
import ray
import time
import geopy
import json


data = {}

"""
neighbors = generate_sorted_list()
data['cover_approx'] = []

ray.init(ignore_reinit_error=True)

neighbors_id = ray.put(neighbors)

@ray.remote
def process(neighbors, k):
    (fac, obj_fake) = cover_approx(neighbors, k)
    #asgn = assign_facilities(fac)
    #obj = calculate_objective(asgn, 95)
    fac = list(fac)
    return [fac, "to do", obj_fake]

data['cover_approx'] = [ray.get(process.remote(neighbors_id, k)) for k in range(3, 11, 1)]
"""

for alg in [center_of_homes, most_populous]:
    data[alg.__name__] = []
    
    for num_facility in range(5,6,1):
        (fac, asgn) = alg(num_facility)
        obj = calculate_objective(asgn, 95)
        
        data[alg.__name__].append([fac, asgn, obj])

print(len(CLIENT_LOCATIONS))

'''data['fpt20'] = []

for k in range(5,21,1):
    (min_obj_guess, asgn) = fpt(k, 20)
    fac = min_obj_guess[1]
    obj = calculate_objective(asgn, 95)
    data['fpt20'].append([fac, asgn, obj])
    print(k, obj)'''

with open(PROJECT_ROOT / 'output' / 'runs' / 'albemarle' / 'tradeoff_5.json', 'w') as f:
        json.dump(data, f)