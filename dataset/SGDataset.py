


import json
import torch
import random
import networkx as nx
from tqdm import tqdm
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
'''
class NodeType(Enum):
    ROOM = "room"
    FURNITURE = "furniture"
    OBJECT = "object"

class EdgeType(Enum):
    ONTOP_UNDER = "onTop_under"  1 "on_under"
    IN_CONTAINS = "in_contains" 2 "contains_in"
    NEAR = "near" 3 "near"
    NONE_RELATIONSHIP 4 
    
others = ['kitchen', 'living_room', 'bedroom', 'bathroom',
          "top-bottom relationship","close relationship", "contains in"] 
'''

node_name_emb_dict_json_pth = '/home/winky/Documents/mycode/1CSK-NAV/dataset/obj_emb.json'
node_name_emb_dict = json.load(open(node_name_emb_dict_json_pth, 'r'))

obj_type_json = '/home/winky/Documents/mycode/1CSK-NAV/script/all_objs.json'
obj_type_dict = json.load(open(obj_type_json, 'r'))

class SGDataset(InMemoryDataset):
    def __init__(self, 
                 graphs,
                 root = '../data', 
                 transform=None, 
                 pre_transform=None):
        self.graphs = graphs # a list of graphs
        
        self.root = root if root else '../data'
        
        super(SGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
            
    @property
    def raw_file_names(self):
        return ['scene_graphs.pt']
    
    @property
    def processed_file_names(self):
        return ['scene_graphs.pt']

    
    def process(self):
        
        samples = []
        
        for graph_k in tqdm(self.graphs):
            data = Data()
            
            G = nx.Graph()
            graph_ = self.graphs[graph_k]
            
            # new nodes
            for obj_ in graph_.keys():
                G.add_node(obj_)
                G.nodes[obj_]['nodeType'] = graph_[obj_]['type']
                G.nodes[obj_]['fea'] = node_name_emb_dict[graph_[obj_]["name"]]["semantic_emb"]
                # print(obj_)
                # print(node_name_emb_dict[graph_[obj_]["name"]]["semantic_emb"])
                
            # print(graph_.keys())
            # new edges
            all_furniture = set()
            for obj_ in graph_.keys():
                if graph_[obj_]['type'] == 'furniture':
                    all_furniture.add(obj_)
            
            
            for obj_ in graph_.keys():
                if obj_[0:4] != 'room':
                    # print(obj_)
                    relevant_objs = []
                    for obj2_ in graph_[obj_]["on_under"]:
                        if not G.has_edge(obj_, obj2_):
                            relevant_objs.append(obj2_)
                            G.add_edge(obj_, obj2_)
                            edge_fea = node_name_emb_dict["top-bottom relationship"]["semantic_emb"]
                            # edge_should_sample_for_loss = 0 # ...
                            G.edges[obj_, obj2_]['fea'] = edge_fea
                            # print(edge_fea)
                            G.edges[obj_, obj2_]['fea_gpt'] = 0 # ...
                            
                            G.edges[obj_, obj2_]['label'] = 1
                            if (graph_[obj_]['type'] == 'object' and graph_[obj2_]['type'] == 'furniture') or (graph_[obj2_]['type'] == 'object' and graph_[obj_]['type'] == 'furniture'):
                                G.edges[obj_, obj2_]['edge_should_sample_for_loss'] = 1
                            else:
                                G.edges[obj_, obj2_]['edge_should_sample_for_loss'] = 0
                                
                    for obj2_ in graph_[obj_]["contains_in"]:
                        if not G.has_edge(obj_, obj2_):
                            relevant_objs.append(obj2_)
                            G.add_edge(obj_, obj2_)
                            edge_fea = node_name_emb_dict["contains in"]["semantic_emb"]
                            G.edges[obj_, obj2_]['fea'] = edge_fea
                            # print(edge_fea)
                            G.edges[obj_, obj2_]['fea_gpt'] = 0 # ...
                            G.edges[obj_, obj2_]['edge_should_sample_for_loss'] = 0
                            G.edges[obj_, obj2_]['label'] = 1
                                
                    for obj2_ in graph_[obj_]["near"]:
                        if not G.has_edge(obj_, obj2_):
                            relevant_objs.append(obj2_)
                            G.add_edge(obj_, obj2_)
                            edge_fea = node_name_emb_dict["contains in"]["semantic_emb"]
                            # edge_should_sample_for_loss = 0 # ...
                            G.edges[obj_, obj2_]['fea'] = edge_fea
                            # print(edge_fea)
                            G.edges[obj_, obj2_]['fea_gpt'] = 0 # ...
                            G.edges[obj_, obj2_]['label'] = 1
                            if (graph_[obj_]['type'] == 'object' and graph_[obj2_]['type'] == 'furniture') or (graph_[obj2_]['type'] == 'object' and graph_[obj_]['type'] == 'furniture'):
                                G.edges[obj_, obj2_]['edge_should_sample_for_loss'] = 1
                            else:
                                G.edges[obj_, obj2_]['edge_should_sample_for_loss'] = 0
                    
                    
                        
                    if graph_[obj_]['type'] == 'object':
                        non_relevant_objs = all_furniture - set(relevant_objs) - {obj_}
                        
                        # select_fur_num = min(8,len(all_furniture))
                        
                        negative_samples = random.sample(non_relevant_objs, min(8, len(non_relevant_objs)))
                        for neg_obj in negative_samples:
                            if not G.has_edge(obj_, neg_obj):
                                edge_fea = node_name_emb_dict["no relevent"]["semantic_emb"]
                                # 从 graph_.keys() 减去 relevant_objs 中随机选取10个作为negative
                                G.add_edge(obj_, neg_obj)
                                G.edges[obj_, neg_obj]['fea'] = edge_fea
                                # print(edge_fea)
                                G.edges[obj_, neg_obj]['fea_gpt'] = 0 # ...
                                G.edges[obj_, neg_obj]['label'] = 0
                                G.edges[obj_, neg_obj]['edge_should_sample_for_loss'] = 1
                # else:
                        
            # store nodes
            node_list = []
            node_nums = {}
            node_num = 0
            
            for node_id, node_fea in G.nodes(data=True):
                node_nums[node_id] = node_num
                node_num += 1
                # print(node_id)
                # print(node_fea)
                node_list.append(node_fea['fea'])
            
            data.x = torch.Tensor(np.array(node_list))
            
            # store edges
            edge_list = [[],[]]
            edge_fea_list = []
            edge_label_list = []
            edge_should_sample_for_loss_list = []
            
            for src_node, dst_node, edge_fea in G.edges(data=True):
                edge_list[0].append(node_nums[src_node])
                edge_list[1].append(node_nums[dst_node])
                edge_label_list.append(edge_fea['label'])
                edge_fea_list.append(edge_fea['fea'])
                edge_should_sample_for_loss_list.append(edge_fea['edge_should_sample_for_loss'])
            
            data.y = torch.Tensor(np.array(edge_label_list))
            data.edge_index = torch.LongTensor(np.array(edge_list))
            data.edge_attr = torch.Tensor(np.array(edge_fea_list))
            data.should_sample_for_loss = torch.Tensor(np.array(edge_should_sample_for_loss_list))
            
            samples.append(data)

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])



#  edge_features = data.edge_attr[indeces,:]
class SGDataset_wType(InMemoryDataset):
    def __init__(self, 
                 graphs,
                 root = '../data', 
                 transform=None, 
                 pre_transform=None):
        self.graphs = graphs # a list of graphs
        
        self.root = root if root else '../data'
        
        super(SGDataset_wType, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
            
    @property
    def raw_file_names(self):
        return ['scene_graphs_wType.pt']
    
    @property
    def processed_file_names(self):
        return ['scene_graphs_wType.pt']

    
    def process(self):
        
        samples = []
        node_name_mapping = {}
        
        for graph_k in tqdm(self.graphs):
            data = Data()
            node_name_mapping[graph_k] = {}
            G = nx.Graph()
            graph_ = self.graphs[graph_k]
            
            # new nodes
            for obj_ in graph_.keys():
                G.add_node(obj_)
                G.nodes[obj_]['nodeType'] = graph_[obj_]['type']
                G.nodes[obj_]['fea'] = node_name_emb_dict[graph_[obj_]["name"]]["semantic_emb"]
                # print(obj_)
                # print(node_name_emb_dict[graph_[obj_]["name"]]["semantic_emb"])
                
            # print(graph_.keys())
            # new edges
            all_furniture = set()
            all_objs = set(graph_.keys())
            for obj_ in graph_.keys():
                if graph_[obj_]['type'] == 'furniture':
                    all_furniture.add(obj_)
            
            
            for obj_ in graph_.keys():
                if obj_[0:4] != 'room':
                    # print(obj_)
                    relevant_objs = []
                    for obj2_ in graph_[obj_]["on_under"]:
                        if not G.has_edge(obj_, obj2_):
                            relevant_objs.append(obj2_)
                            G.add_edge(obj_, obj2_)
                            edge_fea = node_name_emb_dict["top-bottom relationship"]["semantic_emb"]
                            # edge_should_sample_for_loss = 0 # ...
                            G.edges[obj_, obj2_]['fea'] = edge_fea
                            # print(edge_fea)
                            G.edges[obj_, obj2_]['fea_gpt'] = 0 # ...
                            
                            G.edges[obj_, obj2_]['label'] = 1
                            if (graph_[obj_]['type'] == 'object' and graph_[obj2_]['type'] == 'furniture') or (graph_[obj2_]['type'] == 'object' and graph_[obj_]['type'] == 'furniture'):
                                G.edges[obj_, obj2_]['edge_should_sample_for_loss'] = 1
                            else:
                                G.edges[obj_, obj2_]['edge_should_sample_for_loss'] = 0
                                
                    for obj2_ in graph_[obj_]["contains_in"]:
                        if not G.has_edge(obj_, obj2_):
                            relevant_objs.append(obj2_)
                            G.add_edge(obj_, obj2_)
                            edge_fea = node_name_emb_dict["contains in"]["semantic_emb"]
                            G.edges[obj_, obj2_]['fea'] = edge_fea
                            # print(edge_fea)
                            G.edges[obj_, obj2_]['fea_gpt'] = 0 # ...
                            G.edges[obj_, obj2_]['edge_should_sample_for_loss'] = 0
                            G.edges[obj_, obj2_]['label'] = 1
                                
                    for obj2_ in graph_[obj_]["near"]:
                        if not G.has_edge(obj_, obj2_):
                            relevant_objs.append(obj2_)
                            G.add_edge(obj_, obj2_)
                            edge_fea = node_name_emb_dict["contains in"]["semantic_emb"]
                            # edge_should_sample_for_loss = 0 # ...
                            G.edges[obj_, obj2_]['fea'] = edge_fea
                            # print(edge_fea)
                            G.edges[obj_, obj2_]['fea_gpt'] = 0 # ...
                            G.edges[obj_, obj2_]['label'] = 1
                            if (graph_[obj_]['type'] == 'object' and graph_[obj2_]['type'] == 'furniture') or (graph_[obj2_]['type'] == 'object' and graph_[obj_]['type'] == 'furniture'):
                                G.edges[obj_, obj2_]['edge_should_sample_for_loss'] = 1
                            else:
                                G.edges[obj_, obj2_]['edge_should_sample_for_loss'] = 0
                    
                    
                        
                    if graph_[obj_]['type'] == 'object':
                        non_relevant_objs = all_objs - set(relevant_objs) - {obj_}
                        
                        # select_fur_num = min(8,len(all_furniture))
                        
                        negative_samples = random.sample(non_relevant_objs, min(12, len(non_relevant_objs)))
                        for neg_obj in negative_samples:
                            if not G.has_edge(obj_, neg_obj):
                                edge_fea = node_name_emb_dict["no relevent"]["semantic_emb"]
                                # 从 graph_.keys() 减去 relevant_objs 中随机选取10个作为negative
                                G.add_edge(obj_, neg_obj)
                                G.edges[obj_, neg_obj]['fea'] = edge_fea
                                # print(edge_fea)
                                G.edges[obj_, neg_obj]['fea_gpt'] = 0 # ...
                                G.edges[obj_, neg_obj]['label'] = 0
                                if graph_[neg_obj]['type'] == 'furniture':
                                    G.edges[obj_, neg_obj]['edge_should_sample_for_loss'] = 1
                                else:
                                    G.edges[obj_, neg_obj]['edge_should_sample_for_loss'] = 0
                # else:
                        
            # store nodes
            node_list = []
            node_nums = {}
            node_num = 0
            
            for node_id, node_fea in G.nodes(data=True):
                node_nums[node_id] = {}
                node_nums[node_id]['mapnode'] = node_num
                node_num += 1

                node_nums[node_id]["type"] = graph_[node_id]['type']
                # print(node_id)
                # print(node_fea)
                fea_ = []

                fea_.extend(node_fea['fea'])
                fea_.extend(node_name_emb_dict[node_fea['nodeType']]["semantic_emb"])
                node_list.append(np.array(fea_))
            
            data.x = torch.Tensor(np.array(node_list))
            
            # store edges
            edge_list = [[],[]]
            edge_fea_list = []
            edge_label_list = []
            edge_should_sample_for_loss_list = []
            
            for src_node, dst_node, edge_fea in G.edges(data=True):
                edge_list[0].append(node_nums[src_node]['mapnode'])
                edge_list[1].append(node_nums[dst_node]['mapnode'])
                edge_label_list.append(edge_fea['label'])
                edge_fea_list.append(edge_fea['fea'])
                edge_should_sample_for_loss_list.append(edge_fea['edge_should_sample_for_loss'])
            
            data.y = torch.Tensor(np.array(edge_label_list))
            # print(np.array(edge_list).dtype)
            # print(np.array(edge_list))
            data.edge_index = torch.LongTensor(np.array(edge_list))
            # TODO: edge_attr
            data.edge_attr = torch.Tensor(np.array(edge_fea_list))
            data.should_sample_for_loss = torch.Tensor(np.array(edge_should_sample_for_loss_list))
            
            samples.append(data)
            node_name_mapping[graph_k] = node_nums 


        data_json = json.dumps(node_name_mapping, indent=4)
        with open('node_name_mapping_single.json', 'w') as f:
            f.write(data_json)

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])
        
        




node_name_emb_dict_json_pth = '/home/winky/Documents/mycode/1CSK-NAV/dataset/obj_house_emb.json'
node_name_emb_dict = json.load(open(node_name_emb_dict_json_pth, 'r'))
node_name_emb_dict['kitchen']
class SGDataset_wType_house(InMemoryDataset):
    def __init__(self, 
                 graphs,
                 root = '../data', 
                 num_data = '',
                 transform=None, 
                 pre_transform=None):
        self.graphs = graphs # a list of graphs
        
        self.num_data = num_data
        
        self.root = root if root else '../data'
        
        super(SGDataset_wType_house, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
            
    @property
    def raw_file_names(self):
        return ['scene_graphs_wType_house'+self.num_data+'.pt']
    
    @property
    def processed_file_names(self):
        return ['scene_graphs_wType_house'+self.num_data+'.pt']

    
    def process(self):
        
        samples = []
        node_name_mapping = {}
        
        for graph_k in tqdm(self.graphs):
            data = Data()
            node_name_mapping[graph_k] = {}
            G = nx.Graph()
            graph_ = self.graphs[graph_k]
            
            # new nodes
            all_objs = set()
            for k_ in graph_.keys():
                for obj_ in graph_[k_].keys():
                    G.add_node(obj_)
                    all_objs.add(obj_)
                    G.nodes[obj_]['nodeType'] = k_
                    G.nodes[obj_]['fea'] = node_name_emb_dict[graph_[k_][obj_]["name"].lower()]["semantic_emb"]
                    # print(obj_)
                    # print(node_name_emb_dict[graph_[obj_]["name"]]["semantic_emb"])
                
            # print(graph_.keys())
            # new edges
            all_furniture = set()
            
            all_walls = set()
            all_rooms = set()

            for obj_ in graph_['furnitures'].keys():
                all_furniture.add(obj_)
        
            for obj_ in graph_['objects'].keys():
                all_objs.add(obj_)
                    
            for obj_ in graph_['structures'].keys():
                all_walls.add(obj_)
                
            for obj_ in graph_['rooms'].keys():
                all_rooms.add(obj_)
                
            # room contains
            for room in graph_['rooms'].keys():
                for obj in graph_['objects'].keys():
                    if room in graph_['objects'][obj]['contain_in']:
                        if not G.has_edge(room, obj):
                            G.add_edge(room, obj)
                            edge_fea = node_name_emb_dict["contains in"]["semantic_emb"]
                            G.edges[room, obj]['fea'] = edge_fea
                            G.edges[room, obj]['fea_gpt'] = 0
                            G.edges[room, obj]['label'] = 1
                            G.edges[room, obj]['edge_should_sample_for_loss'] = 1
                            
                for fur in graph_['furnitures'].keys():
                    if room in graph_['furnitures'][fur]['contain_in']:
                        if not G.has_edge(room, fur):
                            G.add_edge(room, fur)
                            edge_fea = node_name_emb_dict["contains in"]["semantic_emb"]
                            G.edges[room, fur]['fea'] = edge_fea
                            G.edges[room, fur]['fea_gpt'] = 0
                            G.edges[room, fur]['label'] = 1
                            G.edges[room, fur]['edge_should_sample_for_loss'] = 0
                
                            
                for structure in graph_['structures'].keys():
                    if room in graph_['structures'][structure]['contain_in']:
                        if not G.has_edge(room, structure):
                            G.add_edge(room, structure)
                            edge_fea = node_name_emb_dict["contains in"]["semantic_emb"]
                            G.edges[room, structure]['fea'] = edge_fea
                            G.edges[room, structure]['fea_gpt'] = 0
                            G.edges[room, structure]['label'] = 1
                            G.edges[room, structure]['edge_should_sample_for_loss'] = 0

            # obj: on_under, contains_in (corner), near 
            '''
            ['kitchen', 'living_room', 'bedroom', 'bathroom',
             "top-bottom relationship","close relationship", "contains in", "no relevent", "meet",
             "object", "furniture", "room","structure"] 
            ''' 
            for obj_ in graph_['objects'].keys():
                relevant_objs = []
                for obj2_ in graph_['objects'][obj_]['on_under']:
                    if obj2_ in G:
                        relevant_objs.append(obj2_)
                        G.add_edge(obj_, obj2_)
                        edge_fea = node_name_emb_dict["top-bottom relationship"]["semantic_emb"]
                        G.edges[obj_, obj2_]['fea'] = edge_fea
                        G.edges[obj_, obj2_]['fea_gpt'] = 0
                        G.edges[obj_, obj2_]['label'] = 1
                        G.edges[obj_, obj2_]['edge_should_sample_for_loss'] = 1
                    
                for obj2_ in graph_['objects'][obj_]['near']:
                    if obj2_ in G:
                        relevant_objs.append(obj2_)
                        if not G.has_edge(obj_, obj2_):
                            relevant_objs.append(obj2_)
                            G.add_edge(obj_, obj2_)
                            edge_fea = node_name_emb_dict["close relationship"]["semantic_emb"]
                            G.edges[obj_, obj2_]['fea'] = edge_fea
                            G.edges[obj_, obj2_]['fea_gpt'] = 0
                            G.edges[obj_, obj2_]['label'] = 1
                            
                            if obj2_  in graph_['objects'].keys():
                                G.edges[obj_, obj2_]['edge_should_sample_for_loss'] = 0
                            else:
                                G.edges[obj_, obj2_]['edge_should_sample_for_loss'] = 1
                
                for obj2_ in graph_['objects'][obj_]['contain_in']:
                    if obj2_ in G:
                        relevant_objs.append(obj2_)
                        if not G.has_edge(obj_, obj2_):
                            G.add_edge(obj_, obj2_)
                            edge_fea = node_name_emb_dict["contains in"]["semantic_emb"]
                            G.edges[obj_, obj2_]['fea'] = edge_fea
                            G.edges[obj_, obj2_]['fea_gpt'] = 0
                            G.edges[obj_, obj2_]['label'] = 1
                            G.edges[obj_, obj2_]['edge_should_sample_for_loss'] = 1
                        
                # 随机采样负样本
                non_relevants =  all_objs - set(relevant_objs) - {obj_}
                # 随机采10个
                negative_samples = random.sample(non_relevants, min(12, len(non_relevants)))
                for neg_obj in negative_samples:
                    if neg_obj in G:
                        if not G.has_edge(obj_, neg_obj):
                            edge_fea = node_name_emb_dict["no relevent"]["semantic_emb"]
                            G.add_edge(obj_, neg_obj)
                            G.edges[obj_, neg_obj]['fea'] = edge_fea
                            G.edges[obj_, neg_obj]['fea_gpt'] = 0
                            G.edges[obj_, neg_obj]['label'] = 0
                            if neg_obj in graph_['objects'].keys():
                                G.edges[obj_, neg_obj]['edge_should_sample_for_loss'] = 0
                            else:
                                G.edges[obj_, neg_obj]['edge_should_sample_for_loss'] = 1

            
            # furniture: near, contains_in (corner), attach(wall)
            for fur_ in graph_['furnitures'].keys():
                for obj in graph_['furnitures'][fur]['near']:
                    # obj in G.nodes
                    if obj in G:
                        if not G.has_edge(fur_, obj):
                            G.add_edge(fur_, obj)
                            edge_fea = node_name_emb_dict["close relationship"]["semantic_emb"]
                            G.edges[fur_, obj]['fea'] = edge_fea
                            G.edges[fur_, obj]['fea_gpt'] = 0
                            G.edges[fur_, obj]['label'] = 1
                            G.edges[fur_, obj]['edge_should_sample_for_loss'] = 0
                        
                for cor in graph_['furnitures'][fur]['contain_in']:
                    if not G.has_edge(fur_, cor):
                        if cor in G:
                            G.add_edge(fur_, cor)
                            edge_fea = node_name_emb_dict["close relationship"]["semantic_emb"]
                            G.edges[fur_, cor]['fea'] = edge_fea
                            G.edges[fur_, cor]['fea_gpt'] = 0
                            G.edges[fur_, cor]['label'] = 1
                            G.edges[fur_, cor]['edge_should_sample_for_loss'] = 0
                        
                for wall in graph_['structures'].keys():
                    if not G.has_edge(fur_, wall):
                        G.add_edge(fur_, wall)
                        edge_fea = node_name_emb_dict["attach to"]["semantic_emb"]
                        G.edges[fur_, wall]['fea'] = edge_fea
                        G.edges[fur_, wall]['fea_gpt'] = 0
                        G.edges[fur_, wall]['label'] = 1
                        G.edges[fur_, wall]['edge_should_sample_for_loss'] = 0
            
            # wall: meet(wall) 
            # corner: meet(wall)
            for wall in graph_['structures'].keys():
                if wall[0:4] != 'door':
                    for wall2 in graph_['structures'][wall]['meet']:
                        if not G.has_edge(wall, wall2):
                            G.add_edge(wall, wall2)
                            edge_fea = node_name_emb_dict["meet"]["semantic_emb"]
                            G.edges[wall, wall2]['fea'] = edge_fea
                            G.edges[wall, wall2]['fea_gpt'] = 0
                            G.edges[wall, wall2]['label'] = 1
                            G.edges[wall, wall2]['edge_should_sample_for_loss'] = 0
                

                        
            # store nodes
            node_list = []
            node_nums = {}
            node_num = 0
            
            for node_id, node_fea in G.nodes(data=True):
                node_nums[node_id] = {}
                node_nums[node_id]['mapping_id'] = node_num
                if node_id in graph_['rooms'].keys():
                    node_nums[node_id]["type"] = 'room'
                elif node_id in graph_['furnitures'].keys():
                    node_nums[node_id]["type"] = 'furniture'
                elif node_id in graph_['objects'].keys():
                    node_nums[node_id]["type"] = 'object'
                elif node_id in graph_['structures'].keys():
                    node_nums[node_id]["type"] = 'structure'
                node_num += 1
                # print(node_id)
                # print(node_fea)
                fea_ = []

                # if 'fea' in node_fea.keys():
                fea_.extend(node_fea['fea'])
                # else:
                #     print(node_id)
                #     print(node_fea.keys())
                #     print(graph_['objects'].keys())
                fea_.extend(node_name_emb_dict[node_fea['nodeType']]["semantic_emb"])
                node_list.append(np.array(fea_))
            
            data.x = torch.Tensor(np.array(node_list))
            
            # store edges
            edge_list = [[],[]]
            edge_fea_list = []
            edge_label_list = []
            edge_should_sample_for_loss_list = []
            
            for src_node, dst_node, edge_fea in G.edges(data=True):
                edge_list[0].append(node_nums[src_node]['mapping_id'])
                edge_list[1].append(node_nums[dst_node]['mapping_id'])
                edge_label_list.append(edge_fea['label'])
                edge_fea_list.append(edge_fea['fea'])
                edge_should_sample_for_loss_list.append(edge_fea['edge_should_sample_for_loss'])
            
            data.y = torch.Tensor(np.array(edge_label_list))
            data.edge_index = torch.LongTensor(np.array(edge_list))
            data.edge_attr = torch.Tensor(np.array(edge_fea_list))
            data.should_sample_for_loss = torch.Tensor(np.array(edge_should_sample_for_loss_list))
            
            samples.append(data)
            node_name_mapping[graph_k] = node_nums 


        data_json = json.dumps(node_name_mapping, indent=4)
        with open('node_name_mapping_house'+self.num_data+'.json', 'w') as f:
            f.write(data_json)
        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])


'''
for 3RSCAN
'''
obj_describe_pth = '/home/winky/Documents/dataset/3DSSG/classes.txt'

describe_txt = {}
import sys
sys.path.append('/home/winky/Documents/mycode/1CSK-NAV')

import dataset.scan3r.dataobj_class as dataobj_class
with open(obj_describe_pth, 'r') as file:
    for line in file:
        columns = line.split('\t')
        # print(len(columns))
        describe_txt[columns[1]] = columns
        
node_name_emb_dict = json.load(open('/home/winky/Documents/mycode/1CSK-NAV/dataset/scan3r/describe_emb.json', 'r'))
class SGDataset_3rscan(InMemoryDataset):
    def __init__(self, 
                 graphs,
                 root = '../data3scan', 
                 num_data = '',
                 transform=None, 
                 pre_transform=None):
        self.graphs = graphs # a list of graphs
        
        self.num_data = num_data
        
        self.root = root if root else '../data3scan'
        
        super(SGDataset_3rscan, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
            
    @property
    def raw_file_names(self):
        return ['scene_graphs_3scan'+self.num_data+'.pt']
    
    @property
    def processed_file_names(self):
        return ['scene_graphs_3scan'+self.num_data+'.pt']

    
    def process(self):
        
        samples = []
        node_name_mapping = {}
        
        for graph_k in tqdm(self.graphs.keys()):
            data = Data()
            node_name_mapping[graph_k] = {}
            G = nx.Graph()
            graph_ = self.graphs[graph_k]
            
            # new nodes
            all_objs = set()
            # k_ 为scene id
            not_obj_id = []
            objs_id = []
            for obj_id in graph_['objects']:
                G.add_node(obj_id)
                all_objs.add(obj_id)
                G.nodes[obj_id]['nodeType'] = graph_['objects'][obj_id]['type']
                if G.nodes[obj_id]['nodeType'] != 'object':
                    not_obj_id.append(obj_id)
                else:
                    objs_id.append(obj_id)
                G.nodes[obj_id]['fea0'] = node_name_emb_dict[graph_['objects'][obj_id]["label"]]
                
                if G.nodes[obj_id]['nodeType'] == 'object':
                    G.nodes[obj_id]['fea1'] = node_name_emb_dict['movable object']
                elif G.nodes[obj_id]['nodeType'] == 'furniture':
                    G.nodes[obj_id]['fea1'] = node_name_emb_dict['static furniture']
                else:
                    G.nodes[obj_id]['fea1'] = node_name_emb_dict['room structure']
                G.nodes[obj_id]['name'] = graph_['objects'][obj_id]["label"]
                G.nodes[obj_id]['id'] = obj_id
                G.nodes[obj_id]['fea2'] = node_name_emb_dict[describe_txt[graph_['objects'][obj_id]["label"]][2]]
                    
            
            for obj_id in graph_['objects']:
                for obj2_id in graph_['objects'][obj_id]['attach']:
                    G.add_edge(obj_id, obj2_id)
                    edge_fea = node_name_emb_dict["attach"]
                    G.edges[obj_id, obj2_id]['fea'] = edge_fea
                    G.edges[obj_id, obj2_id]['fea_gpt'] = 0
                    G.edges[obj_id, obj2_id]['label'] = 1
                    
                    if (graph_['objects'][obj_id]['type'] == 'object' and graph_['objects'][obj2_id]['type'] != 'object') or \
                       (graph_['objects'][obj2_id]['type'] == 'object' and graph_['objects'][obj_id]['type'] != 'object'):
                        G.edges[obj_id, obj2_id]['edge_should_sample_for_loss'] = 1
                    else:
                        G.edges[obj_id, obj2_id]['edge_should_sample_for_loss'] = 0
                
                for obj2_id in graph_['objects'][obj_id]['support']:
                    G.add_edge(obj_id, obj2_id)
                    edge_fea = node_name_emb_dict["support"]
                    G.edges[obj_id, obj2_id]['fea'] = edge_fea
                    G.edges[obj_id, obj2_id]['fea_gpt'] = 0
                    G.edges[obj_id, obj2_id]['label'] = 1
                    
                    if (graph_['objects'][obj_id]['type'] == 'object' and graph_['objects'][obj2_id]['type'] != 'object') or \
                       (graph_['objects'][obj2_id]['type'] == 'object' and graph_['objects'][obj_id]['type'] != 'object'):
                        G.edges[obj_id, obj2_id]['edge_should_sample_for_loss'] = 1
                    else:
                        G.edges[obj_id, obj2_id]['edge_should_sample_for_loss'] = 0
                
                for obj2_id in graph_['objects'][obj_id]['close']:
                    G.add_edge(obj_id, obj2_id)
                    edge_fea = node_name_emb_dict["close by"]
                    G.edges[obj_id, obj2_id]['fea'] = edge_fea
                    G.edges[obj_id, obj2_id]['fea_gpt'] = 0
                    G.edges[obj_id, obj2_id]['label'] = 1
                    
                    if (graph_['objects'][obj_id]['type'] == 'object' and graph_['objects'][obj2_id]['type'] != 'object') or \
                       (graph_['objects'][obj2_id]['type'] == 'object' and graph_['objects'][obj_id]['type'] != 'object'):
                        G.edges[obj_id, obj2_id]['edge_should_sample_for_loss'] = 1
                    else:
                        G.edges[obj_id, obj2_id]['edge_should_sample_for_loss'] = 0
            
              
            # 随机采样负样本
            # 随机采10个
            negative_samples = random.sample(not_obj_id, min(6, len(not_obj_id)))
            for neg_obj in negative_samples:
                for is_obj_id in objs_id:
                    if not G.has_edge(is_obj_id, neg_obj):
                        edge_fea = node_name_emb_dict["none relevent"]
                        G.add_edge(is_obj_id, neg_obj)
                        G.edges[is_obj_id, neg_obj]['fea'] = edge_fea
                        G.edges[is_obj_id, neg_obj]['fea_gpt'] = 0
                        G.edges[is_obj_id, neg_obj]['label'] = 0
                        G.edges[is_obj_id, neg_obj]['edge_should_sample_for_loss'] = 1

                        
            # store nodes
            node_list = []
            node_nums = {}
            node_num = 0
            
            for node_id, node_fea in G.nodes(data=True):
                node_nums[node_id] = {}
                node_nums[node_id]['mapping_id'] = node_num
                node_nums[node_id]['name'] = node_fea['name']
                node_nums[node_id]['id'] = node_fea['id']
                # if node_id in graph_['rooms'].keys():
                #     node_nums[node_id]["type"] = 'room'
                # elif node_id in graph_['furnitures'].keys():
                #     node_nums[node_id]["type"] = 'furniture'
                # elif node_id in graph_['objects'].keys():
                #     node_nums[node_id]["type"] = 'object'
                # elif node_id in graph_['structures'].keys():
                #     node_nums[node_id]["type"] = 'structure'
                node_num += 1
                # print(node_id)
                # print(node_fea)
                fea_ = []

                # if 'fea' in node_fea.keys():
                fea_.extend(node_fea['fea0'])
                fea_.extend(node_fea['fea1'])
                fea_.extend(node_fea['fea2'])
                # else:
                #     print(node_id)
                #     print(node_fea.keys())
                #     print(graph_['objects'].keys())
                # fea_.extend(node_name_emb_dict[node_fea['nodeType']]["semantic_emb"])
                node_list.append(np.array(fea_))
            
            data.x = torch.Tensor(np.array(node_list))
            print(data.x.shape[0])
            if data.x.shape[0] == 0:
                print(graph_k)
            
            # store edges
            edge_list = [[],[]]
            edge_fea_list = []
            edge_label_list = []
            edge_should_sample_for_loss_list = []
            
            for src_node, dst_node, edge_fea in G.edges(data=True):
                edge_list[0].append(node_nums[src_node]['mapping_id'])
                edge_list[1].append(node_nums[dst_node]['mapping_id'])
                edge_label_list.append(edge_fea['label'])
                edge_fea_list.append(edge_fea['fea'])
                edge_should_sample_for_loss_list.append(edge_fea['edge_should_sample_for_loss'])
            
            data.y = torch.Tensor(np.array(edge_label_list))
            data.edge_index = torch.LongTensor(np.array(edge_list))
            data.edge_attr = torch.Tensor(np.array(edge_fea_list))
            data.should_sample_for_loss = torch.Tensor(np.array(edge_should_sample_for_loss_list))
            
            samples.append(data)
            node_name_mapping[graph_k] = node_nums 


        data_json = json.dumps(node_name_mapping, indent=4)
        with open('node_name_mapping_house'+self.num_data+'.json', 'w') as f:
            f.write(data_json)
        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])


class SGDataset_3rscan1(InMemoryDataset):
    def __init__(self, 
                 graphs,
                 root = '../data3scan1', 
                 num_data = '',
                 transform=None, 
                 pre_transform=None):
        self.graphs = graphs # a list of graphs
        
        self.num_data = num_data
        
        self.root = root if root else '../data3scan1'
        
        super(SGDataset_3rscan1, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
            
    @property
    def raw_file_names(self):
        return ['scene_graphs_3scan1'+self.num_data+'.pt']
    
    @property
    def processed_file_names(self):
        return ['scene_graphs_3scan1'+self.num_data+'.pt']

    
    def process(self):
        
        samples = []
        node_name_mapping = {}
        
        for graph_k in tqdm(self.graphs.keys()):
            data = Data()
            node_name_mapping[graph_k] = {}
            G = nx.Graph()
            graph_ = self.graphs[graph_k]
            
            # new nodes
            all_objs = set()
            # k_ 为scene id
            not_obj_id = []
            objs_id = []
            for obj_id in graph_['objects']:
                G.add_node(obj_id)
                all_objs.add(obj_id)
                G.nodes[obj_id]['nodeType'] = graph_['objects'][obj_id]['type']
                if G.nodes[obj_id]['nodeType'] != 'object':
                    not_obj_id.append(obj_id)
                else:
                    objs_id.append(obj_id)
                G.nodes[obj_id]['fea0'] = node_name_emb_dict[graph_['objects'][obj_id]["label"]]
                
                if G.nodes[obj_id]['nodeType'] == 'object':
                    G.nodes[obj_id]['fea1'] = node_name_emb_dict['movable object']
                elif G.nodes[obj_id]['nodeType'] == 'furniture':
                    G.nodes[obj_id]['fea1'] = node_name_emb_dict['static furniture']
                else:
                    G.nodes[obj_id]['fea1'] = node_name_emb_dict['room structure']
                G.nodes[obj_id]['name'] = graph_['objects'][obj_id]["label"]
                G.nodes[obj_id]['id'] = obj_id
                G.nodes[obj_id]['fea2'] = node_name_emb_dict[describe_txt[graph_['objects'][obj_id]["label"]][2]]
                    
            
            for obj_id in graph_['objects']:
                for obj2_id in graph_['objects'][obj_id]['attach']:
                    G.add_edge(obj_id, obj2_id)
                    edge_fea = node_name_emb_dict["attach"]
                    G.edges[obj_id, obj2_id]['fea'] = edge_fea
                    G.edges[obj_id, obj2_id]['fea_gpt'] = 0
                    G.edges[obj_id, obj2_id]['label'] = 1
                    
                    if (graph_['objects'][obj_id]['type'] == 'object' and graph_['objects'][obj2_id]['type'] != 'object') or \
                       (graph_['objects'][obj2_id]['type'] == 'object' and graph_['objects'][obj_id]['type'] != 'object'):
                        G.edges[obj_id, obj2_id]['edge_should_sample_for_loss'] = 1
                    else:
                        G.edges[obj_id, obj2_id]['edge_should_sample_for_loss'] = 0
                
                for obj2_id in graph_['objects'][obj_id]['support']:
                    G.add_edge(obj_id, obj2_id)
                    edge_fea = node_name_emb_dict["support"]
                    G.edges[obj_id, obj2_id]['fea'] = edge_fea
                    G.edges[obj_id, obj2_id]['fea_gpt'] = 0
                    G.edges[obj_id, obj2_id]['label'] = 1
                    
                    if (graph_['objects'][obj_id]['type'] == 'object' and graph_['objects'][obj2_id]['type'] != 'object') or \
                       (graph_['objects'][obj2_id]['type'] == 'object' and graph_['objects'][obj_id]['type'] != 'object'):
                        G.edges[obj_id, obj2_id]['edge_should_sample_for_loss'] = 1
                    else:
                        G.edges[obj_id, obj2_id]['edge_should_sample_for_loss'] = 0
                
                for obj2_id in graph_['objects'][obj_id]['close']:
                    G.add_edge(obj_id, obj2_id)
                    edge_fea = node_name_emb_dict["close by"]
                    G.edges[obj_id, obj2_id]['fea'] = edge_fea
                    G.edges[obj_id, obj2_id]['fea_gpt'] = 0
                    G.edges[obj_id, obj2_id]['label'] = 1
                    
                    if (graph_['objects'][obj_id]['type'] == 'object' and graph_['objects'][obj2_id]['type'] != 'object') or \
                       (graph_['objects'][obj2_id]['type'] == 'object' and graph_['objects'][obj_id]['type'] != 'object'):
                        G.edges[obj_id, obj2_id]['edge_should_sample_for_loss'] = 1
                    else:
                        G.edges[obj_id, obj2_id]['edge_should_sample_for_loss'] = 0
            
              
            # 随机采样负样本
            # 随机采10个
            negative_samples = random.sample(not_obj_id, min(6, len(not_obj_id)))
            for neg_obj in negative_samples:
                for is_obj_id in objs_id:
                    if not G.has_edge(is_obj_id, neg_obj):
                        edge_fea = node_name_emb_dict["none relevent"]
                        G.add_edge(is_obj_id, neg_obj)
                        G.edges[is_obj_id, neg_obj]['fea'] = edge_fea
                        G.edges[is_obj_id, neg_obj]['fea_gpt'] = 0
                        G.edges[is_obj_id, neg_obj]['label'] = 0
                        G.edges[is_obj_id, neg_obj]['edge_should_sample_for_loss'] = 1

                        
            # store nodes
            node_list = []
            node_nums = {}
            node_num = 0
            
            for node_id, node_fea in G.nodes(data=True):
                node_nums[node_id] = {}
                node_nums[node_id]['mapping_id'] = node_num
                node_nums[node_id]['name'] = node_fea['name']
                node_nums[node_id]['id'] = node_fea['id']
                # if node_id in graph_['rooms'].keys():
                #     node_nums[node_id]["type"] = 'room'
                # elif node_id in graph_['furnitures'].keys():
                #     node_nums[node_id]["type"] = 'furniture'
                # elif node_id in graph_['objects'].keys():
                #     node_nums[node_id]["type"] = 'object'
                # elif node_id in graph_['structures'].keys():
                #     node_nums[node_id]["type"] = 'structure'
                node_num += 1
                # print(node_id)
                # print(node_fea)
                fea_ = []

                # if 'fea' in node_fea.keys():
                fea_.extend(node_fea['fea0'])
                fea_.extend(node_fea['fea1'])
                # fea_.extend(node_fea['fea2'])
                # else:
                #     print(node_id)
                #     print(node_fea.keys())
                #     print(graph_['objects'].keys())
                # fea_.extend(node_name_emb_dict[node_fea['nodeType']]["semantic_emb"])
                node_list.append(np.array(fea_))
            
            data.x = torch.Tensor(np.array(node_list))
            print(data.x.shape[0])
            if data.x.shape[0] == 0:
                print(graph_k)
            
            # store edges
            edge_list = [[],[]]
            edge_fea_list = []
            edge_label_list = []
            edge_should_sample_for_loss_list = []
            
            for src_node, dst_node, edge_fea in G.edges(data=True):
                edge_list[0].append(node_nums[src_node]['mapping_id'])
                edge_list[1].append(node_nums[dst_node]['mapping_id'])
                edge_label_list.append(edge_fea['label'])
                edge_fea_list.append(edge_fea['fea'])
                edge_should_sample_for_loss_list.append(edge_fea['edge_should_sample_for_loss'])
            
            data.y = torch.Tensor(np.array(edge_label_list))
            data.edge_index = torch.LongTensor(np.array(edge_list))
            data.edge_attr = torch.Tensor(np.array(edge_fea_list))
            data.should_sample_for_loss = torch.Tensor(np.array(edge_should_sample_for_loss_list))
            
            samples.append(data)
            node_name_mapping[graph_k] = node_nums 


        data_json = json.dumps(node_name_mapping, indent=4)
        with open('node_name_mapping_house'+self.num_data+'.json', 'w') as f:
            f.write(data_json)
        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])