


import sys
import json
import math
import prior
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from shapely.geometry import Polygon
import ai2thor.wsgi_server
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


'''

'''

sys.path.append("/home/winky/Documents/mycode/1CSK-NAV")
from utils.ai2thor_collect import *


all_obj_json = '/home/winky/Documents/mycode/1CSK-NAV/script/all_objs_prior.json'
all_obj_dict = json.load(open(all_obj_json))

objects = [x for x in all_obj_dict.keys() if all_obj_dict[x]['type'] == 'object']
furnitures = [x for x in all_obj_dict.keys() if all_obj_dict[x]['type'] == 'furniture']

dataset_10k = prior.load_dataset('procthor-10k')

all_scene = {}

def find_inner_corners(room_polygon, walls_in_room):
    inner_corners = []
    n = len(room_polygon)

    for i in range(n):
        prev_point = room_polygon[(i-1+n)%n] # previous vertex
        curr_point = room_polygon[i] # current vertex
        next_point = room_polygon[(i+1)%n] # next vertex

        v1 = [curr_point[0] - prev_point[0], curr_point[1] - prev_point[1]]
        v2 = [next_point[0] - curr_point[0], next_point[1] - curr_point[1]]

        cross_product = v1[0]*v2[1] - v1[1]*v2[0]

        ind = 0
        if cross_product < 0:
            curr_point = (round(curr_point[0], 2), round(curr_point[1], 2))
            # print(curr_point)
            for wall_ in walls_in_room:
                # print(walls_in_room)
                if curr_point in wall_:
                    ind += 1
            if ind >= 2:
                inner_corners.append(curr_point)
    return inner_corners

def get_corners_info(walls, room_regions_dict, distance = 1.):
    corners_info = {}
    index = 0
    for roomid in walls.keys():
        room_polygon = room_regions_dict[roomid]
        corners_info[roomid] = {}
        walls_in_room = []
        for wall in walls[roomid]:
            walls_in_room.append([(round(float(wall.split('|')[2]),2), round(float(wall.split('|')[3]),2)),
                                  (round(float(wall.split('|')[4]),2), round(float(wall.split('|')[5]),2))])
        corners_coor = find_inner_corners(room_polygon, walls_in_room)

        for corner in corners_coor:
            corners_info[roomid]['corner|' + str(index)] = {}
            corners_info[roomid]['corner|' + str(index)]['pos'] = corner
            corners_info[roomid]['corner|' + str(index)]['connect_walls'] = []
            for wall in walls[roomid]:
                if corner in [(round(float(wall.split('|')[2]),2), round(float(wall.split('|')[3]),2)),
                              (round(float(wall.split('|')[4]),2), round(float(wall.split('|')[5]),2))]:
                    corners_info[roomid]['corner|' + str(index)]['connect_walls'].append(wall)

            suqate_corner = Polygon([(corner[0] - distance, corner[1] - distance),
                                     (corner[0] + distance, corner[1] - distance),
                                     (corner[0] + distance, corner[1] + distance),
                                     (corner[0] - distance, corner[1] + distance)])
            corner_regions = Polygon(room_polygon).intersection(suqate_corner)
            corners_info[roomid]['corner|' + str(index)]['region'] = corner_regions
            index += 1
            
    return corners_info


def get_distance(pos1, pos2):
    dis = math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
    return dis

def distance_from_point_to_line(point, line_start, line_end):
    # Line represented as line_start to line_end
    # Point represented as point
    line = np.array(line_end) - np.array(line_start)
    point_from_start = np.array(point) - np.array(line_start)
    t = np.dot(line, point_from_start) / np.dot(line, line)
    t = np.clip(t, 0, 1)  # Constrain t to [0, 1]
    closest_point = np.array(line_start) + t * line
    distance = np.linalg.norm(closest_point - np.array(point))
    return distance

'''
ROOM
STRUCTURE
    - door
    - wall
    - corner
FURNITURE
OBJECT
'''

'''
contain_in room-obj/furniture/wall/door/door/corner | furniture/obj-cornerr 
near obj/furniture
on_under object-furniture
attach paint?wall | furniture-wall
meet wall-wall | corner-wall |
'''

for i in tqdm(range(0,1500)):
    house = dataset_10k['train'][i]
    controller_ = Controller(scene=house,
                             platform=CloudRendering
                             )
    event_data = controller_.last_event.metadata
    
    obj_df = show_house_objects_table(event_data['objects'],
                                      house['rooms'])
    walls = {}
    for idx, row in obj_df.iterrows():
        obj_uuid = row['objectId']
        obj_name = row['obj_name']

        if obj_name == 'wall':
            walls[obj_uuid] = [[float(obj_uuid.split('|')[2]),float(obj_uuid.split('|')[3])],
                            [float(obj_uuid.split('|')[4]),float(obj_uuid.split('|')[5])]]
    
    all_scene['train|'+str(i)] = {}

    all_scene['train|'+str(i)]['rooms'] = {}
    all_scene['train|'+str(i)]['structures'] = {}
    all_scene['train|'+str(i)]['furnitures'] = {}
    all_scene['train|'+str(i)]['objects'] = {}

    
    walls_in_room = {}
    rooms_info = {}
    for room in house['rooms']:
        rooms_info[room['id']] = [[x['x'], x['z']] for x in room['floorPolygon']]
    # corners
    for wall in walls:
        if 'room|' + wall.split('|')[1] not in walls_in_room.keys():
            walls_in_room['room|' + wall.split('|')[1]] = [wall]
        else:
            walls_in_room['room|' + wall.split('|')[1]].append(wall)
    # print(" walls_in_room", walls_in_room)        
    corners_info = get_corners_info(walls_in_room, rooms_info)

    for room in corners_info.keys():
        for corner in corners_info[room].keys():
            all_scene['train|'+str(i)]['structures'][corner] = {}
            all_scene['train|'+str(i)]['structures'][corner]['uuid'] = corner
            all_scene['train|'+str(i)]['structures'][corner]['name'] = 'corner'
            all_scene['train|'+str(i)]['structures'][corner]['contain_in'] = [room]
            all_scene['train|'+str(i)]['structures'][corner]['meet'] = corners_info[room][corner]['connect_walls']


    

    for idx, row in obj_df.iterrows():
        obj_uuid = row['objectId']
        obj_name = row['obj_name']
        obj_type = row['objectType']
        # room
        if obj_type == 'Floor':
            all_scene['train|'+str(i)]['rooms'][obj_uuid] = {}
            all_scene['train|'+str(i)]['rooms'][obj_uuid]['uuid'] = obj_uuid
            all_scene['train|'+str(i)]['rooms'][obj_uuid]['name'] = obj_name

        # doors
        elif obj_type == 'Doorway':
            all_scene['train|'+str(i)]['structures'][obj_uuid] = {}
            all_scene['train|'+str(i)]['structures'][obj_uuid]['uuid'] = obj_uuid # door|8|10	
            all_scene['train|'+str(i)]['structures'][obj_uuid]['name'] = obj_name # door
            all_scene['train|'+str(i)]['structures'][obj_uuid]['contain_in'] = row['inRoomOf']
            all_scene['train|'+str(i)]['structures'][obj_uuid]['near'] = []
            all_scene['train|'+str(i)]['structures'][obj_uuid]['pos'] = [row['position[x]'], row['position[z]']]
            
            room1 = 'room|'+obj_uuid.split('|')[1]
            room2 = 'room|'+obj_uuid.split('|')[2]

            if room1 in rooms_info.keys() and room2 in rooms_info.keys():
                # print( walls_in_room)
                wall1 = walls_in_room[room1]
                wall2 = walls_in_room[room2]
                for w1 in wall1:
                    for w2 in wall2:
                        if w1[7:] == w2[7:]:
                            all_scene['train|'+str(i)]['structures'][obj_uuid]['contain_in'].append(w1)
                            all_scene['train|'+str(i)]['structures'][obj_uuid]['contain_in'].append(w2)
            else:
                if room1 in rooms_info.keys():
                    obj_position = [row['position[x]'], row['position[z]']]
                    # calculate the min distance to the walls in the room
                    walls_in_same_room = walls_in_room[room1]
                    # Compute the distance to each wall
                    distances = {}
                    for wall in walls_in_same_room:
                        _, _, x1, y1, x2, y2 = wall.split("|")
                        line_start = [float(x1), float(y1)]
                        line_end = [float(x2), float(y2)]
                        distance = distance_from_point_to_line(obj_position, line_start, line_end)
                        distances[wall] = distance
                    closest_wall = min(distances, key=distances.get)
                    all_scene['train|'+str(i)]['structures'][obj_uuid]['contain_in'].append(closest_wall)

        # walls
        elif obj_type == 'Wall':
            all_scene['train|'+str(i)]['structures'][obj_uuid] = {}
            all_scene['train|'+str(i)]['structures'][obj_uuid]['uuid'] = obj_uuid # wall|9|13.65|13.65|17.55|13.65
            all_scene['train|'+str(i)]['structures'][obj_uuid]['name'] = obj_name # wall
            all_scene['train|'+str(i)]['structures'][obj_uuid]['contain_in'] = row['inRoomOf']
            all_scene['train|'+str(i)]['structures'][obj_uuid]['meet'] = []
            walls[obj_uuid] = [[float(obj_uuid.split('|')[2]),float(obj_uuid.split('|')[3])],
                               [float(obj_uuid.split('|')[4]),float(obj_uuid.split('|')[5])]]
            walls_in_same_room = walls_in_room[row['inRoomOf'][0]]
                # Compute the distance to each wall
            for wall in walls_in_same_room:
                if (obj_uuid.split('|')[2] == wall.split('|')[2] and obj_uuid.split('|')[3] == wall.split('|')[3]) or \
                   (obj_uuid.split('|')[4] == wall.split('|')[2] and obj_uuid.split('|')[5] == wall.split('|')[3]) or \
                   (obj_uuid.split('|')[2] == wall.split('|')[4] and obj_uuid.split('|')[3] == wall.split('|')[5]) or \
                   (obj_uuid.split('|')[4] == wall.split('|')[4] and obj_uuid.split('|')[5] == wall.split('|')[5]) :
                    if wall != obj_uuid:
                        all_scene['train|'+str(i)]['structures'][obj_uuid]['meet'].append(wall)
        
        # objects/furnitures
        elif obj_name.lower() in objects:
            all_scene['train|'+str(i)]['objects'][obj_uuid] = {}
            all_scene['train|'+str(i)]['objects'][obj_uuid]['uuid'] = obj_uuid
            all_scene['train|'+str(i)]['objects'][obj_uuid]['name'] = obj_name
            all_scene['train|'+str(i)]['objects'][obj_uuid]['contain_in'] = row['inRoomOf']
            all_scene['train|'+str(i)]['objects'][obj_uuid]['near'] = []
            all_scene['train|'+str(i)]['objects'][obj_uuid]['on_under'] = []
            all_scene['train|'+str(i)]['objects'][obj_uuid]['pos'] = [row['position[x]'], row['position[z]']]

            obj_region = Polygon(row['locate'])
            obj_area = obj_region.area

            if row['parentReceptacles'] != None:
                for parent in row['parentReceptacles']:
                    if parent.split('|')[0] != 'Floor':
                        all_scene['train|'+str(i)]['objects'][obj_uuid]['on_under'].append(parent)
                        all_scene['train|'+str(i)]['objects'][obj_uuid]['near'].append(parent)

            for cor in corners_info[row['inRoomOf'][0]].keys():
                cor_region = corners_info[row['inRoomOf'][0]][cor]['region']
                intersection = obj_region.intersection(cor_region)
                if not intersection.is_empty:
                    if obj_area < 0.6:
                        if intersection.area > obj_area/2:
                            all_scene['train|'+str(i)]['objects'][obj_uuid]['contain_in'].append(cor)
                    else:
                        if intersection.area > 0.6:
                            all_scene['train|'+str(i)]['objects'][obj_uuid]['contain_in'].append(cor)
                        # all_scene['train|'+str(i)]['objects'][obj_uuid]['near'].append(cor)

            all_scene['train|'+str(i)]['objects'][obj_uuid]['attach'] = []

        # furnitures
        elif obj_name.lower() in furnitures:
            all_scene['train|'+str(i)]['furnitures'][obj_uuid] = {}
            all_scene['train|'+str(i)]['furnitures'][obj_uuid]['uuid'] = obj_uuid
            all_scene['train|'+str(i)]['furnitures'][obj_uuid]['name'] = obj_name
            all_scene['train|'+str(i)]['furnitures'][obj_uuid]['contain_in'] = row['inRoomOf']
            all_scene['train|'+str(i)]['furnitures'][obj_uuid]['near'] = []
            all_scene['train|'+str(i)]['furnitures'][obj_uuid]['on_under'] = []
            all_scene['train|'+str(i)]['furnitures'][obj_uuid]['pos'] = [row['position[x]'], row['position[z]']]

            if row['parentReceptacles'] != None:
                for parent in row['parentReceptacles']:
                    if parent.split('|')[0] != 'Floor':
                        all_scene['train|'+str(i)]['furnitures'][obj_uuid]['on_under'].append(parent)
                        all_scene['train|'+str(i)]['furnitures'][obj_uuid]['near'].append(parent)

            all_scene['train|'+str(i)]['furnitures'][obj_uuid]['attach'] = []
            

            obj_region = Polygon(row['locate'])
            obj_area = obj_region.area

            if obj_name != 'painting':
                for cor in corners_info[row['inRoomOf'][0]].keys():
                    cor_region = corners_info[row['inRoomOf'][0]][cor]['region']
                    intersection = obj_region.intersection(cor_region)
                    if not intersection.is_empty:
                        if obj_area < 0.6:
                            if intersection.area > (2*obj_area)/3:
                                all_scene['train|'+str(i)]['furnitures'][obj_uuid]['contain_in'].append(cor)
                        else:
                            if intersection.area > 0.6:
                                all_scene['train|'+str(i)]['furnitures'][obj_uuid]['contain_in'].append(cor)



            if obj_name == 'painting':
                obj_position = all_scene['train|'+str(i)]['furnitures'][obj_uuid]['pos']
                # calculate the min distance to the walls in the room
                walls_in_same_room = walls_in_room[row['inRoomOf'][0]]
                # Compute the distance to each wall
                distances = {}
                for wall in walls_in_same_room:
                    _, _, x1, y1, x2, y2 = wall.split("|")
                    line_start = [float(x1), float(y1)]
                    line_end = [float(x2), float(y2)]
                    distance = distance_from_point_to_line(obj_position, line_start, line_end)
                    distances[wall] = distance
                closest_wall = min(distances, key=distances.get)
                all_scene['train|'+str(i)]['furnitures'][obj_uuid]['attach'] = [closest_wall]
            else:
                obj_position = all_scene['train|'+str(i)]['furnitures'][obj_uuid]['pos']
                walls_in_same_room = walls_in_room[row['inRoomOf'][0]]
                # Compute the distance to each wall
                distances = {}
                for wall in walls_in_same_room:
                    _, _, x1, y1, x2, y2 = wall.split("|")
                    line_start = [float(x1), float(y1)]
                    line_end = [float(x2), float(y2)]
                    distance = distance_from_point_to_line(obj_position, line_start, line_end)
                    distances[wall] = distance
                closest_wall = min(distances, key=distances.get)
                all_scene['train|'+str(i)]['furnitures'][obj_uuid]['attach'] = [closest_wall]
                closest_wall = min(distances, key=distances.get)
                if distances[closest_wall] < 0.1:
                    if closest_wall not in all_scene['train|'+str(i)]['furnitures'][obj_uuid]['attach']:
                        all_scene['train|'+str(i)]['furnitures'][obj_uuid]['attach'].append(closest_wall)
    for obj1 in all_scene['train|'+str(i)]['objects'].keys():
        for obj2 in all_scene['train|'+str(i)]['objects'].keys():
            if obj1 != obj2:
                dis = get_distance(all_scene['train|'+str(i)]['objects'][obj1]['pos'], all_scene['train|'+str(i)]['objects'][obj2]['pos'])
                if dis < 1.:
                    all_scene['train|'+str(i)]['objects'][obj1]['near'].append(obj2)
                    all_scene['train|'+str(i)]['objects'][obj2]['near'].append(obj1)
                    # print(obj1,'near' ,obj2)
    controller_.stop()




        
    

    
data_json = json.dumps(all_scene, indent=4)
with open('all_objs_1500.json', 'w') as f:
    f.write(data_json)           
    
    






