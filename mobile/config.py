from collections import namedtuple
from os import path
import json
from mobile import PROJECT_ROOT
from typing import Dict, List, Tuple, Set
import pandas as pd
import geopy.distance

"""
address: namedtuple containing the client index, visited location, and potential facility

HOME_SHIFT : to adjust the lids of residential locations
"""

address = namedtuple('address', ['index', 'location', 'facility'])
HOME_SHIFT=1000000000

############################################## DATA LOADING #######################################################

def create_data_input(county_name: str = 'charlottesville_city', home_work_only = False):
    """
    PARAMETERS
    ------------
    county_name: str
    home_work_only: bool
    
    RETURNS
    ------------
    LOCATIONS: List[Dict[str, T]]
        list of locations sorted in descending order of activity; each location represented with a dictionary of the following fields:
            lid_ind: int
                the index of the location in the LOCATIONS list
            longitude: float
            latitude: float
            activity: int
                the number of clients that visit this location
            pid: List[int]
                a list of the pids of the clients that visit the location
            home: bool
                indicator for whether this location is a home
    CLIENT_LOCATIONS: Dict[int, List[int]]
        dictionary matching each pid to a list of locations (with home locations at the beginning) that they visit (represented by the index/lid_ind in LOCATIONS)
    
    """
    #Read in both the activity and residence locations
    
    df_activity = pd.read_csv(PROJECT_ROOT/ 'data'/ 'raw'/ county_name / f"usa_va_{county_name}_activity_locations.csv").rename({"alid": "lid"}, axis = 'columns')
    df_residence = pd.read_csv(PROJECT_ROOT/ 'data'/ 'raw'/ county_name/ f"usa_va_{county_name}_residence_locations.csv")

    #Shift the residence lid
    df_residence['lid'] = df_residence['rlid'] + HOME_SHIFT
    locations = pd.concat([df_activity[['lid', 'longitude', 'latitude']], df_residence[['lid', 'longitude', 'latitude']]]).reset_index(drop = True)

    #Read in the client visited locations data (adults only for now)
    client_locations = pd.read_csv(PROJECT_ROOT/ 'data'/ 'raw'/ county_name/ f"usa_va_{county_name}_adult_activity_location_assignment_week.csv")
    if home_work_only:
        client_locations = client_locations.drop(client_locations[(client_locations["activity_type"]!=1) & (client_locations["activity_type"]!=2)].index, axis=0)

    #Get the coordinates of all the residential locations
    home_coords = set(df_residence[['latitude', 'longitude']].apply(tuple, axis=1).tolist())

    #Shift lids for residential areas
    client_locations['coord'] = client_locations[['latitude', 'longitude']].apply(lambda x: (x.latitude, x.longitude), axis = 1)
    client_locations.loc[client_locations.coord.isin(home_coords), 'lid'] += HOME_SHIFT
    
    #Find popularity of locations, which locations are visited by which individuals
    assignments = client_locations.copy()
    assignments = assignments.groupby(['lid'])['pid'].apply(set).reset_index(name = 'pid')
    assignments = assignments.set_index('lid')
    assignments['activity'] = assignments['pid'].apply(lambda x: len(x))

    locations['activity'] = locations['lid'].apply(lambda x: assignments.at[x, 'activity'] if x in assignments.index else 0)
    locations['pid'] = locations['lid'].apply(lambda x: list(assignments.at[x, 'pid']) if x in assignments.index else [])
    locations = locations.sort_values(by = 'activity', ascending = False).reset_index(drop = True)

    client_locations = client_locations.groupby(['pid'])['lid'].apply(set).reset_index(name = 'lid')

    #Replace lid with the index of the lid in locations
    def filter(x):
        return_list = []
        for i in x:
        #Insert home locations at the front of the list
            if (i>HOME_SHIFT):
                return_list.insert(0, int(locations.loc[locations.lid==i].index[0]))
            else:
                return_list.append(int(locations.loc[locations.lid==i].index[0]))
        return return_list

    client_locations['lid'] = client_locations['lid'].apply(lambda x: filter(x))

    return locations.to_dict('index'), client_locations.to_dict('index')


def write_data_input(county_name: str = 'charlottesville_city', home_work_only=False):
    """
    Calls create_data_input and writes to a JSON file
    """
    
    if home_work_only:
        filename = "original_1.json"
    else:
        filename = "original.json"
    
    with open(PROJECT_ROOT / 'data' / 'processed' / county_name / filename, 'w') as f:
        LOCATIONS, CLIENT_LOCATIONS = create_data_input(county_name, home_work_only)
        data = {"LOCATIONS": LOCATIONS, "CLIENT_LOCATIONS": CLIENT_LOCATIONS}
        json.dump(data, f)

def read_data_input(county_name: str = 'charlottesville_city', home_work_only = False):
    """
    Checks if JSON file with data exists: if not, it calls create_data_input.
    Otherwise, it reads from the file and returns LOCATIONS and CLIENT_LOCATIONS (see create_data_input for more details)
    """
    
    if home_work_only:
        filename = "original_1.json"
    else:
        filename = "original.json"
    
    directory_path = PROJECT_ROOT / 'data' / 'processed' / county_name / filename
    
    if not path.exists(directory_path):
        write_data_input(county_name, home_work_only=home_work_only)
    
    file = open(directory_path, 'r')
    data = json.load(file)

    LOCATIONS = []
    for key, val in data["LOCATIONS"].items():
        val["lid_ind"] = int(key)
        val["home"] = val["lid"] >= HOME_SHIFT
        LOCATIONS.append(val)
    
    CLIENT_LOCATIONS = {int(value['pid']): value['lid'] for ind, value in data["CLIENT_LOCATIONS"].items()}
    return LOCATIONS, CLIENT_LOCATIONS

############################################## AGGREGATION #################################################



def radius_cover(LOCATIONS, CLIENT_LOCATIONS, radius: float, county_name: str = 'charlottesville_city', home_work_only=False):
    """
    Helper method for set_cover_aggregation: generates balls for the given radius and stores chosen locations and radius-coverage mappings in a JSON file
    
    PARAMETERS
    --------------
    LOCATIONS: List[Dict[str, T]]
        list of locations from read_data_input
    CLIENT_LOCATIONS: Dict[int, List[int]
        map of client location assignments from read_data_input
    radius:
        radius used to form clustered location sets such that for any given location, there is a set containing all locations within radius km from it
    county_name: str
        'charlottesville_city' or 'albemarle'
    home_work_only: bool
        indicates whether the location is a home
    """
    
    radius_dict = {}
    LOCATIONS_act = [(ind, value) for ind,value in enumerate(LOCATIONS) if not LOCATIONS[ind]['home']]
    
    for i in range(len(LOCATIONS_act)):
        loc1 = LOCATIONS_act[i][0]
        radius_dict[loc1] = []
        
        for j in range(len(LOCATIONS_act)):
            loc2 = LOCATIONS_act[j][0]
            
            coord_1 = (LOCATIONS[loc1]['latitude'], LOCATIONS[loc1]['longitude'])
            coord_2 = (LOCATIONS[loc2]['latitude'], LOCATIONS[loc2]['longitude'])
            
            dist = geopy.distance.great_circle(coord_1, coord_2).km
            
            if dist < radius:
                radius_dict[loc1].append(loc2)

    cover = set()
    chosen = set()
    
    while len(cover) < len(LOCATIONS_act):
        max_choice = (0, set(), -1)
        
        for key, val in radius_dict.items():
            if key not in chosen:
                set_choice = set(val)-cover
                if len(set_choice)>max_choice[0]:
                    max_choice = (len(set_choice), set_choice, key)
        
        cover = cover.union(max_choice[1])
        chosen.add(max_choice[2])
    
    if home_work_only:
        filename = f'radius_cover_{int(1000*radius)}_1.json'
    else:
        filename = f'radius_cover_{int(1000*radius)}.json'
    
    with open(PROJECT_ROOT/ 'data'/ 'processed'/ county_name / filename, 'w') as f:
        data = {"radius": radius, "radius_dict": radius_dict, "chosen": list(chosen)}
        json.dump(data, f)

def set_cover_aggregation(LOCATIONS, CLIENT_LOCATIONS, county_name: str = 'charlottesville_city', radius: float = 0.01, home_work_only=False):
    """
    Creates balls of given radius around each location
    Picks location balls to cover all possible facility locations through set cover approximation
    Creates and returns a version of LOCATIONS and CLIENT_LOCATIONS that are based on the aggregate locs
    
    PARAMETERS
    ------------------
    LOCATIONS: List[Dict[str, T]]
        list of unclustered locations from read_data_input
    CLIENT_LOCATIONS: Dict[int, List[int]]
        dictionary of client location assignments without clustering from read_data_input
    county_name: str
        'charlottesville_city' or 'albemarle'
    radius: float
        radius to construct location sets for clustering
    home_work_only: bool
        indicates whether the location is a home
    
    
    RETURNS
    ------------------
    LOCATIONS: List[Dict[str, T]]
        list of locations sorted in descending order of activity; each location represented with a dictionary of the following fields:
            lid_ind: int
                the index of the location in the LOCATIONS list
            longitude: float
            latitude: float
            activity: int
                the number of clients that visit this location
            pid: List[int]
                a list of the pids of the clients that visit the location
            home: bool
                indicator for whether this location is a home
    CLIENT_LOCATIONS: Dict[int, List[int]]
        dictionary matching each pid to a list of locations that they visit (represented by the index/lid_ind in LOCATIONS)
    """
    
    if home_work_only:
        filename = f"radius_cover_{int(1000*radius)}_1.json"
    else:
        filename = f"radius_cover_{int(1000*radius)}.json"
    
    file_radius = open(PROJECT_ROOT/ 'data'/ 'processed'/ county_name / filename, 'r')
    data_radius = json.load(file_radius)
    
    #Dict[int, List[int]] ==> {loc: locations covered by a ball of given radius centered at loc}
    cluster_dict = {int(ind): val for ind, val in data_radius["radius_dict"].items()}
    
    #Set[int] ==> contains the picked loc centers (for which the balls are constructed)
    chosen_points = set(data_radius["chosen"])
    
    #--------------------------------- LOCATIONS ------------------------------# 
    
    LOCATIONS_act = [(ind, value) for ind, value in enumerate(LOCATIONS) if not LOCATIONS[ind]['home']]
    
    #Reverse map the lids to the renumbered lid_ind
    reverse_lid_index = {}
    for i, loc in enumerate(LOCATIONS_act):
        reverse_lid_index[loc[0]] = i

    activity_locations = []
    for point in chosen_points:
        
        pid_set = set()
        member_list = cluster_dict[point]
        
        for loc in cluster_dict[point]:
            pid_set = pid_set.union(LOCATIONS[loc]['pid'])
        
        activity_locations.append({"lid_ind" : point,
                     "longitude" : LOCATIONS[point]['longitude'] ,
                     "latitude" : LOCATIONS[point]['latitude'],
                     "activity" : len(pid_set),
                     "pid" : list(pid_set),
                     "home" : False})
        
    residential_locations = []
    LOCATIONS_res = [(ind, value) for ind,value in enumerate(LOCATIONS) if LOCATIONS[ind]['home']]
    
    for loc in LOCATIONS_res:
        
        locations_dict = LOCATIONS[loc[0]]
        
        residential_locations.append({"lid_ind": locations_dict['lid_ind'],
                    "longitude" : locations_dict['longitude'],
                    "latitude" : locations_dict['latitude'],
                    "activity" : locations_dict['activity'],
                    "pid" : locations_dict["pid"],
                    "home" : True
                    })
    
    LOCATIONS_agg = sorted(activity_locations + residential_locations, key = lambda x: x["activity"], reverse = True)
    
    ind_to_reindex = {}
    for ind, loc in enumerate(LOCATIONS_agg):
        ind_to_reindex[loc['lid_ind']] = ind
        loc['lid_ind'] = ind
    
    #------------------------- CLIENT LOCATIONS --------------------------------#
    
    #Maps each location to chosen locations covering it
    coverage_matching = {}
    for cluster in cluster_dict.keys():
        overlap = [member for member in cluster_dict[cluster] if member in chosen_points]
        coverage_matching[cluster] = overlap
    
    CLIENT_LOCATIONS_agg = {}
    for key, val in CLIENT_LOCATIONS.items():
        new_lid_list = []
        for loc in val:
            
            if not LOCATIONS[loc]['home']:
                for elem in coverage_matching[loc]:
                    #Avoid repeats in client locations
                    if ind_to_reindex[elem] not in new_lid_list:
                        new_lid_list.append(ind_to_reindex[elem])
            else:
                new_lid_list.append(ind_to_reindex[loc])
        
        CLIENT_LOCATIONS_agg[key] = new_lid_list
    
    return LOCATIONS_agg, CLIENT_LOCATIONS_agg


def aggregate_data(county_name: str = 'charlottesville_city', aggregation: int = 1, radius: float = 0.01, home_work_only=False):
    """
    PARAMETERS
    ------------
    county_name: str
        'charlottesville_city' or 'albemarle'
    aggregation: int
        0 : default, no aggregation
        1 : set cover aggregation
    radius: float
        clustering radius to form sets
    home_work_only: bool
        only consider home and work locations in client movement patterns (used for information loss experiments)
    
    RETURNS
    ------------
    LOCATIONS: List[Dict[str, T]]
        list of locations sorted in descending order of activity; each location represented with a dictionary of the following fields:
            lid_ind: int
                the index of the location in the LOCATIONS list
            longitude: float
            latitude: float
            activity: int
                the number of clients that visit this location
            pid: List[int]
                a list of the pids of the clients that visit the location
            home: bool
                indicator for whether this location is a home
    CLIENT_LOCATIONS: Dict[int, List[int]]
        dictionary matching each pid to a list of locations that they visit (represented by the index/lid_ind in LOCATIONS)
    """
    
    LOCATIONS, CLIENT_LOCATIONS = read_data_input(county_name, home_work_only = home_work_only)
        
    elif aggregation == 1:
        
        if home_work_only:
            filename = f"aggregation_{aggregation}_{int(1000*radius)}_1.json"
        else:
            filename = f"aggregation_{aggregation}_{int(1000*radius)}.json"
        
        aggregation_file = PROJECT_ROOT/'data'/'processed'/county_name/filename
        
        ###Check if combination has already been attempted###
        if path.exists(aggregation_file):
            with open(aggregation_file, 'r') as f:
                data = json.load(f)
                LOCATIONS_1 = data['LOCATIONS']
                CLIENT_LOCATIONS_1 = {int(key):value for key,value in data['CLIENT_LOCATIONS'].items()}
                return LOCATIONS_1, CLIENT_LOCATIONS_1
        
        else:
            
            if home_work_only:
                filename = f"radius_cover_{int(1000*radius)}_1.json"
            else:
                filename = f"radius_cover_{int(1000*radius)}.json"
            
            radius_file = PROJECT_ROOT / 'data'/ 'processed'/ county_name / filename

            if not path.exists(radius_file):
                radius_cover(LOCATIONS, CLIENT_LOCATIONS, radius, county_name, home_work_only=home_work_only)
            
            LOCATIONS_1, CLIENT_LOCATIONS_1 = set_cover_aggregation(LOCATIONS, CLIENT_LOCATIONS, county_name, radius, home_work_only=home_work_only)
            
            ###Store output in json file###
            with open(aggregation_file, 'w') as f:
                json.dump({"LOCATIONS": LOCATIONS_1, "CLIENT_LOCATIONS": CLIENT_LOCATIONS_1}, f)
            
            return LOCATIONS_1, CLIENT_LOCATIONS_1
        
    else:
        
        for loc_dict in LOCATIONS:
            del loc_dict['lid']
        
        return LOCATIONS, CLIENT_LOCATIONS

