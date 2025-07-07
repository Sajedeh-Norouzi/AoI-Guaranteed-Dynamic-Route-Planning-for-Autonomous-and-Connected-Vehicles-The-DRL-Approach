# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 18:14:31 2022

@author: Maryam & Sajedeh
"""

import numpy as np
import networkx as nx
np.random.seed (1376)
class Env:
    
    def __init__(self, n_vehicle, n_subchannel, n_paths):
        
        self.n_vehicle = n_vehicle
        self.x_max = 3 * 250 # m
        self.y_max = 3 * 430 # m
        #self.speed = 10/10e3 # m/ms in this case it added just a bout 0.001 and it didnt make change in locations
        self.speed = 10 #10 m / s
        self.max_AoI = 10 # ms
        self.max_capacity = 4
        self.deltaM = 2 #Capacity estimation error 
        self.n_subchannel =  n_subchannel
        self.sigma_noise = 10e-15
        self.time_slot = 1  # ms
        self.n_vehicle_per_sub = int(0.5* self.n_vehicle/self.n_subchannel)  # NOMA 
        self.n_node = 16
        self.n_paths = n_paths
        
    def road(self):
        
        road = {'1':[0, 0], '2':[250, 0], '3': [500, 0], '4':[750, 0], \
                '5':[0, 430], '6':[250, 430], '7': [500, 430], '8':[750, 430], \
                '9':[0, 860], '10':[250, 860], '11': [500, 860], '12':[750, 860], \
                '13':[0, 1290], '14':[250, 1290], '15': [500, 1290], '16':[750, 1290]}
        
        return road
    
    def selecting_options(self):
        
        selecting_options = {'1':[2, 5], '2':[1, 6, 3], '3': [2, 7, 4], '4':[3, 8], \
                            '5':[1, 6, 9], '6':[2, 5, 7, 10], '7': [3, 6, 8, 11], '8':[4, 7, 12], \
                            '9':[5, 10, 13], '10':[6, 9, 11, 14], '11': [7, 10, 12, 15], '12':[8, 11, 16], \
                            '13':[9, 14], '14':[10, 13, 15], '15': [11, 14, 16], '16':[12, 15]}
                        
        return selecting_options
    
    def selecting_options_links(self):
        
        selecting_options_links = {'1':[0,3], '2':[0,4,1], '3':[1,5,2], '4':[2,6], \
                                '5':[3,7,10], '6':[4,7,8,11], '7': [5, 8, 9, 12], '8':[6,9,8], \
                                '9':[10,14,17], '10':[11,14,18, 15], '11': [12, 15, 16, 19], '12': [13, 16, 20], \
                                '13':[17, 21], '14':[18, 21, 22], '15': [19, 22, 23], '16':[20, 23] }        
                        
        return selecting_options_links    
    
    def capacity_check(self, current_points, capacity, i ):
        
        #current_points = self.n_vehicle *1
        #selecting_options defined matrix
        # capacity = self.n_path *1
        
        selecting_options = {'1':[2, 5], '2':[1, 6, 3], '3': [2, 7, 4], '4':[3, 8], \
                            '5':[1, 6, 9], '6':[2, 5, 7, 10], '7': [3, 6, 8, 11], '8':[4, 7, 12], \
                            '9':[5, 10, 13], '10':[6, 9, 11, 14], '11': [7, 10, 12, 15], '12':[8, 11, 16], \
                            '13':[9, 14], '14':[10, 13, 15], '15': [11, 14, 16], '16':[12, 15]}

        selecting_options_ = {'1':[2, 5], '2':[1, 6, 3], '3': [2, 7, 4], '4':[3, 8], \
                            '5':[1, 6, 9], '6':[2, 5, 7, 10], '7': [3, 6, 8, 11], '8':[4, 7, 12], \
                            '9':[5, 10, 13], '10':[6, 9, 11, 14], '11': [7, 10, 12, 15], '12':[8, 11, 16], \
                            '13':[9, 14], '14':[10, 13, 15], '15': [11, 14, 16], '16':[12, 15]}
                        
        selecting_options_links = {'1':[0,3], '2':[0,4,1], '3':[1,5,2], '4':[2,6], \
                                '5':[3,7,10], '6':[4,7,8,11], '7': [5, 8, 9, 12], '8':[6,9,13], \
                                '9':[10,14,17], '10':[11,14,15,18], '11': [12, 15, 16, 19], '12': [13, 16, 20], \
                                '13':[17, 21], '14':[18, 21, 22], '15': [19, 22, 23], '16':[20, 23] }
        
        if current_points[i] !=0:
            c =[]
            capacity_index=[]
            for n in range (len(selecting_options_links[str(current_points[i])])):
                if  capacity[selecting_options_links[str(current_points[i])][n]] >= self.max_capacity:
                    c.append(selecting_options[str(current_points[i])][n])
                capacity_index.append(capacity[selecting_options_links[str(current_points[i])][n]])
            selecting_options [str(current_points[i])] =  [x for x in selecting_options[str(current_points[i])] if x not in c]
            
            if selecting_options [str(current_points[i])] == []:
                selecting_options [str(current_points[i])].append(selecting_options_ [str(current_points[i])][np.argmin(capacity_index)])
                
                

        return selecting_options
          


    
    def initial_locs(self, road):
        
        initial_loc = []
        destination_loc = []
        initial_loc_co = []
        destination_loc_co = []
            
        for i in range(self.n_vehicle):
            origin_nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            destination_nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            i_o = np.random.choice(origin_nodes)
            initial_loc.append(i_o) #random origin
            destination_nodes.remove(i_o)
            i_d = np.random.choice(destination_nodes) #random dest
            destination_loc.append(i_d) 
            destination_loc_co.append(road[str(i_d)]) #destination coordination 
            initial_loc_co.append(road[str(i_o)])
            
        #__________________array____________________ 
        initial_loc = np.array(initial_loc)
        destination_loc = np.array(destination_loc)
        initial_loc_co = np.array(initial_loc_co)
        destination_loc_co = np.array(destination_loc_co)    
             
        return initial_loc, destination_loc, initial_loc_co, destination_loc_co # list of vehicles and nodes

    def initial_select_point(self, initial_loc, destination_loc):
        ini_selected_points = []
        first_selected_point = np.zeros(self.n_vehicle)
        #first_selected_point_co = [] 
        G = nx.Graph()
        e = [('1','2', 250), ('2','3', 250), ('3','4', 250),\
             ('5','6', 250), ('6','7', 250), ('7','8', 250),\
             ('9','10', 250), ('10','11', 250), ('11','12', 250),\
             ('13','14', 250), ('14','15', 250), ('15','16', 250),\
             ('1','5', 430),('5','9', 430), ('9','13', 430),\
             ('2','6', 430),('6','10', 430), ('10','14', 430),\
             ('3','7', 430),('7','11', 430), ('11','15', 430),\
             ('4','8', 430),('8','12', 430), ('12','16', 430)]
            
        G.add_weighted_edges_from(e)
        for i in range(self.n_vehicle):
            ini_selected_points.append(nx.dijkstra_path(G, str(initial_loc[i]),\
                                                        str(destination_loc[i])))
            first_selected_point[i] = ini_selected_points[i][1]  
            # first_selected_point_co.append (road [str(int(first_selected_point[i]))])
        #__________ Array form ____________
        first_selected_point = np.array(first_selected_point)
        
        
        # return ini_selected_points, first_selected_point, first_selected_point_co
        # return ini_selected_points, first_selected_point
        return first_selected_point
    
    def loc_update(self, road, loc_vehicle_f, selected_point, destination_loc_co, flag_vehicle):  
        #selected point is indicated by nodes and selected_point_co is coordination 
        loc_vehicle = loc_vehicle_f.copy()
        #selected_point = np.array(selected_point)
        selected_point_co = []
        # location vehicle is a state
        # selected_point is an action and state
        for i in range(self.n_vehicle):
            selected_point_co.append(road[str(int(selected_point[i]))])
            
        for i in range(self.n_vehicle):
            #for j in range(self.n_node):
                # if loc_vehicle[i][0] == road[str(j+1)][0] and loc_vehicle[i][1] == road[str(j+1)][1] 
            if flag_vehicle[i]==0 and (not (np.array_equal(destination_loc_co[i], loc_vehicle[i]))):
                
                if selected_point_co[i][0]==loc_vehicle[i][0] and selected_point_co[i][1]>loc_vehicle[i][1]: #up 
                    my_random_float = np.random.random()
                    if my_random_float >= 0.5:
                        my_rand_int = self.speed
                    else:
                        my_rand_int = self.speed/2
                    if selected_point_co[i][1]-loc_vehicle[i][1]<=self.speed/2:
                        my_rand_int = self.speed/2  
                        
                    loc_vehicle[i][1] = loc_vehicle[i][1] + my_rand_int*self.time_slot
                
                if selected_point_co[i][0]==loc_vehicle[i][0] and selected_point_co[i][1]<loc_vehicle[i][1]:  #down
                    my_random_float = np.random.random()
                    if my_random_float >= 0.5:
                        my_rand_int = self.speed
                    else:
                        my_rand_int = self.speed/2   
                    if loc_vehicle[i][1]-selected_point_co[i][1]<=self.speed/2:
                        my_rand_int = self.speed/2    
                        
                    loc_vehicle[i][1] = loc_vehicle[i][1] - my_rand_int*self.time_slot
                
                if selected_point_co[i][1]==loc_vehicle[i][1] and selected_point_co[i][0]>loc_vehicle[i][0]:   ##right 
                    my_random_float = np.random.random()
                    if my_random_float >= 0.5:
                        my_rand_int = self.speed
                    else:
                        my_rand_int = self.speed/2
                    if selected_point_co[i][0]-loc_vehicle[i][0]<=self.speed/2:
                        my_rand_int = self.speed/2
                        
                    loc_vehicle[i][0] = loc_vehicle[i][0] + my_rand_int*self.time_slot
                    
                if selected_point_co[i][1]==loc_vehicle[i][1] and selected_point_co[i][0]<loc_vehicle[i][0]:   #left 
                    my_random_float = np.random.random()
                    if my_random_float >= 0.5:
                        my_rand_int = self.speed
                    else:
                        my_rand_int = self.speed/2
                    if loc_vehicle[i][0]-selected_point_co[i][0]<=self.speed/2:
                        my_rand_int = self.speed/2
                        
                    loc_vehicle[i][0] = loc_vehicle[i][0] - my_rand_int*self.time_slot
            
            if np.array_equal(destination_loc_co[i], loc_vehicle[i]):
                flag_vehicle[i] = 1
                    
        return loc_vehicle, flag_vehicle
    
    def flow(self, loc_vehicle, road, age):
        
        flow = np.zeros(self.n_paths) 
        AoI = np.zeros(self.n_paths)
        AoI_M = np.zeros(self.n_paths)
        for i in range(self.n_vehicle):
        
            if road[str(1)][0]<loc_vehicle[i][0] and loc_vehicle[i][0]<road[str(2)][0]:
                if loc_vehicle[i][1]==0:
                  flow[0] += 1
                  AoI[0] += age[i]
                if loc_vehicle[i][1]==430:
                  flow[7] += 1
                  AoI[7] += age[i]
                if loc_vehicle[i][1]==860:
                  flow[14] += 1 
                  AoI[14] += age[i]
                if loc_vehicle[i][1]==1290:
                  flow[21] += 1  
                  AoI[21] += age[i]
            #-------------------------------------------------------------------
            if road[str(2)][0]<loc_vehicle[i][0] and loc_vehicle[i][0]<road[str(3)][0]:
                if loc_vehicle[i][1]==0:
                  flow[1] += 1
                  AoI[1] += age[i]
                if loc_vehicle[i][1]==430:
                  flow[8] += 1
                  AoI[8] += age[i]
                if loc_vehicle[i][1]==860:
                  flow[15] += 1  
                  AoI[15] += age[i]
                if loc_vehicle[i][1]==1290:
                  flow[22] += 1 
                  AoI[22] += age[i]
            #-------------------------------------------------------------------
            if road[str(3)][0]<loc_vehicle[i][0] and loc_vehicle[i][0]<road[str(4)][0]:
                if loc_vehicle[i][1]==0:
                  flow[2] += 1
                  AoI[2] += age[i]
                if loc_vehicle[i][1]==430:
                  flow[9] += 1
                  AoI[9] += age[i]
                if loc_vehicle[i][1]==860:
                  flow[16] += 1  
                  AoI[16] += age[i]
                if loc_vehicle[i][1]==1290:
                  flow[23] += 1  
                  AoI[23] += age[i]
            #-------------------------------------------------------------------
            if road[str(1)][1]<loc_vehicle[i][1] and loc_vehicle[i][1]<road[str(5)][1]:
                if loc_vehicle[i][0]==0:
                  flow[3] += 1
                  AoI[3] += age[i]
                if loc_vehicle[i][0]==250:
                  flow[4] += 1
                  AoI[4] += age[i]
                if loc_vehicle[i][0]==500:
                  flow[5] += 1  
                  AoI[5] += age[i]
                if loc_vehicle[i][0]==750:
                  flow[6] += 1  
                  AoI[6] += age[i]
            #-------------------------------------------------------------------
            if road[str(5)][1]<loc_vehicle[i][1] and loc_vehicle[i][1]<road[str(9)][1]:
                if loc_vehicle[i][0]==0:
                  flow[10] += 1
                  AoI[10] += age[i]
                if loc_vehicle[i][0]==250:
                  flow[11] += 1
                  AoI[11] += age[i]
                if loc_vehicle[i][0]==500:
                  flow[12] += 1  
                  AoI[12] += age[i]
                if loc_vehicle[i][0]==750:
                  flow[13] += 1
                  AoI[13] += age[i]
            #-------------------------------------------------------------------
            if road[str(9)][1]<loc_vehicle[i][1] and loc_vehicle[i][1]<road[str(13)][1]:
                if loc_vehicle[i][0]==0:
                  flow[17] += 1
                  AoI[17] += age[i]
                if loc_vehicle[i][0]==250:
                  flow[18] += 1
                  AoI[18] += age[i]
                if loc_vehicle[i][0]==500:
                  flow[19] += 1 
                  AoI[19] += age[i]
                if loc_vehicle[i][0]==750:
                  flow[20] += 1   
                  AoI[20] += age[i]
            
        for r in range(self.n_paths):
            if (AoI[r] !=0) and (flow[r] !=0):
                AoI_M[r] = AoI[r] /flow[r]
                  
        return flow, AoI_M
    
    
    def estimated_capacity(self, flow, AoI_M):
        
        estimated_capacity = np.zeros(self.n_paths)
        travel_times = np.zeros(self.n_paths)
        lengthroad = np.zeros(self.n_paths)
        free_flow_time = np.zeros(self.n_paths)
        length250 = [0,1,2,7,8,9,14,15,16,21,22,23]
        length430 = [3,4,5,6,10,11,12,13,17,18,19,20] 
            
        for r in range(self.n_paths): 
            estimated_capacity[r] = self.max_capacity - flow[r] - (int(AoI_M[r]/self.max_AoI)* self.deltaM) 
            # the estimated capacity 
            if r in length250:
                lengthroad[r] = 250
            elif r in length430 :
                lengthroad[r] = 430
                
            free_flow_time[r] = lengthroad[r]/ self.speed  
            if estimated_capacity [r] > 0:
                travel_times[r] = free_flow_time[r] * ( 1 + 0.15*((flow[r]/estimated_capacity[r] )**4))
            else:
                travel_times[r] = 5*free_flow_time[r] * ( 1 + 0.15*((flow[r]/1 )**4))
        return estimated_capacity, travel_times
              
    def AoI(self, ch_assignment, age_f, flag_vehicle):
        
        ch_assignment = sum(ch_assignment.T).T
        age = age_f.copy()
        for i in range(self.n_vehicle):
            # math.isclose(vehicle_rate[i], self.min_rate, rel_tol=1e-5)==True
            if ch_assignment[i]>= 1 or flag_vehicle[i]==1:
                age[i] = 0
            else:
                age[i] += 1
        
        return age
    
    def reach_intersection(self, loc_vehicle, road):
        
        flag_reach = np.zeros(self.n_vehicle)
        reach_to = np.zeros(self.n_vehicle)
        for i in range(self.n_vehicle):
            for j in range(1, self.n_node+1):
                if loc_vehicle[i][0]==road[str(j)][0] and loc_vehicle[i][1]==road[str(j)][1]:
                    flag_reach[i] = 1
                    reach_to[i] = j
                    
        return flag_reach, reach_to           
    
    def start_point(self):
        # Generate the road network and get initial positions (both node indices and coordinates)
        road = self.road()
        initial_locs, destination_loc, initial_locs_co, destination_loc_co = self.initial_locs(road)
    
        # Normalize coordinates
        initial_locs_co[:, 0] = normalize_coords(initial_locs_co[:, 0], 0, self.x_max)
        initial_locs_co[:, 1] = normalize_coords(initial_locs_co[:, 1], 0, self.y_max)
        destination_loc_co[:, 0] = normalize_coords(destination_loc_co[:, 0], 0, self.x_max)
        destination_loc_co[:, 1] = normalize_coords(destination_loc_co[:, 1], 0, self.y_max)
    
        # Select initial route or point
        first_selected_point = self.initial_select_point(initial_locs, destination_loc)
    
        # Initialize other state components
        age = 5 * np.ones(self.n_vehicle)  # Initial AoI
        flow = np.zeros(self.n_paths)      # Path flows
        flag_vehicle = np.zeros(self.n_vehicle)  # Route assignment flags
        tt_veh = np.zeros(self.n_vehicle)        # Estimated travel times
    
        # Combine all into a single state vector
        state_vector = np.concatenate((
            initial_locs_co.flatten(),      # Normalized coordinates 2V [0:2V]
            destination_loc_co.flatten(),   # Normalized destination coordinates 2V [2V:4V]
            first_selected_point.flatten(), # Initial path selection V [4V:5V]
            age, # V [5V:6V]
            flag_vehicle, #V [6V:7V]
            flow, #n_path n_path [7V:7V+self.n_paths]
            tt_veh #V   [7V+self.n_paths:8V+self.n_paths]
        ))
    
        return state_vector



    
    

    def step(self, actions, old_states):
                
        road= self.road()
        #_____ Getting old states _____
        vehicle_loc = old_states[0: 2*self.n_vehicle].reshape(self.n_vehicle, 2)
        destination_loc_co = old_states[2*self.n_vehicle: 4*self.n_vehicle].reshape(self.n_vehicle, 2)
        selected_point = old_states[ 4*self.n_vehicle: 5*self.n_vehicle].reshape(self.n_vehicle)
        age = old_states[5*self.n_vehicle:6*self.n_vehicle].reshape(self.n_vehicle)
        flag_vehicle = old_states[6*self.n_vehicle: 7* self.n_vehicle].reshape(self.n_vehicle) #reaching destination 
        flow = old_states[7*self.n_vehicle: 7*self.n_vehicle + self.n_paths].reshape(self.n_paths)
        tt_veh = old_states[7*self.n_vehicle + self.n_paths: 8*self.n_vehicle + self.n_paths].reshape(self.n_vehicle)
        
        
        # vehicle_loc = old_states[0: 2*self.n_vehicle].reshape(self.n_vehicle, 2) # initial_locs_co
        # destination_loc_co = old_states[2*self.n_vehicle: 4*self.n_vehicle].reshape(self.n_vehicle, 2) # destination_loc_co
        # selected_point = old_states[ 4*self.n_vehicle: 5*self.n_vehicle].reshape(self.n_vehicle)
        # age = old_states[5*self.n_vehicle:6*self.n_vehicle].reshape(self.n_vehicle)
        # flag_vehicle = old_states[6*self.n_vehicle: 7* self.n_vehicle].reshape(self.n_vehicle) #reaching destination 
        # flow = old_states[7*self.n_vehicle: 7*self.n_vehicle + self.n_paths].reshape(self.n_paths)
        # tt_veh = old_states[7*self.n_vehicle + self.n_paths: 8*self.n_vehicle + self.n_paths].reshape(self.n_vehicle)
        
        
        
        # _____ actions____
        selecting_point = ((actions[0: self.n_vehicle]+1)/2).reshape(self.n_vehicle, 1)
        channel_assignment = ((actions[self.n_vehicle: self.n_vehicle+ (self.n_vehicle * self.n_subchannel)]+1)/2).reshape(self.n_vehicle, self.n_subchannel)
        
        #_____Getting new states and Applying actions____
        flag_reach, reach_to = self.reach_intersection(vehicle_loc, road)
        reach_to = reach_to.astype(int)
        selecting_options = self.selecting_options()
        #selecting_options_links = self.selecting_options_links()
        #_________ Applying actions________
        #_________ Selecting the next points ______****************************************
        selected_point_ = np.zeros(self.n_vehicle)
        #selected_point_link_ = np.zeros(self.n_vehicle)
        
                
                
        for i in range(self.n_vehicle):
            if flag_vehicle[i] !=1:
                if flag_reach[i]==1: 
                    if not np.sum(flow[:])==0:
                        selecting_options_new = self.capacity_check(reach_to, flow, i)
                    else: 
                        selecting_options_new = selecting_options
                        
                    if len(selecting_options_new[str(reach_to[i])]) ==1:
                        action_reach =0
                    if len(selecting_options_new[str(reach_to[i])]) ==0:
                        print ('error')
                    if len(selecting_options_new[str(reach_to[i])]) >1: 
                        if np.isnan(selecting_point[i]):
                            selecting_point[i] = 0
                        action_reach = int(len(selecting_options_new[str(reach_to[i])]) *selecting_point[i])
                        if action_reach >= len(selecting_options_new[str(reach_to[i])]):
                            action_reach = len(selecting_options_new[str(reach_to[i])]) -1
                    selected_point_[i] = selecting_options_new[str(reach_to[i])][action_reach]   
                else:
                    selected_point_[i] = selected_point[i]
            else:
                selected_point_[i] = selected_point[i]
                
                
                
        vehicle_loc_, flag_vehicle_ = self.loc_update(road, vehicle_loc, selected_point_, destination_loc_co, flag_vehicle)    
        
        # Normalize coordinates
        vehicle_loc_[:, 0] = normalize_coords(vehicle_loc_[:, 0], 0, self.x_max)
        vehicle_loc_[:, 1] = normalize_coords(vehicle_loc_[:, 1], 0, self.y_max)
        destination_loc_co[:, 0] = normalize_coords(destination_loc_co[:, 0], 0, self.x_max)
        destination_loc_co[:, 1] = normalize_coords(destination_loc_co[:, 1], 0, self.y_max)
        
        #_________ Channel assignment ______
        allocated_channel = np.zeros([self.n_vehicle, self.n_subchannel])
       
        for i in range(self.n_vehicle):
            if flag_vehicle[i] !=1:
                for j in range(self.n_subchannel): 
                    if np.sum(allocated_channel[i, :]) <1:
                        if np.sum(allocated_channel[:, j]) < self.n_vehicle_per_sub:
                            if channel_assignment[i, j] >= 0.5:
                                allocated_channel[i, j] = 1
                    

        age_ = self.AoI(allocated_channel, age, flag_vehicle) #age each vehicle 
        flow_, AoI_= self.flow(vehicle_loc_, road, age_) 
        estimated_capacity_, travel_time_ = self.estimated_capacity(flow_, AoI_)  
        
        tt_veh_ = np.zeros(self.n_vehicle)
        for i in range(self.n_vehicle):
            if flag_vehicle[i]==0 :
                tt_veh_[i] = tt_veh[i] + 1
            else:
                tt_veh_[i] = tt_veh[i]
                
        new_state_vector = np.concatenate((
            vehicle_loc_.flatten(),      # Normalized coordinates 2V [2V:4V]
            destination_loc_co.flatten(),   # Normalized destination coordinates 2V [4V:6V]
            selected_point_.flatten(), # Initial path selection V [6V:7V]
            age_, # V [7V:8V]
            flag_vehicle_, #V [8V:9V]
            flow_, #n_path n_path [9V:9V+self.n_paths]
            tt_veh_ #V   [9V+self.n_paths:10V+self.n_paths]
        ))
                

        
        #reward = -(np.mean(travel_time_[:])/np.max(travel_time_[:]))  # base line 
        reward = -(np.mean(travel_time_[:])/np.max(travel_time_[:]))-(np.mean(age_[:])/np.max(age_[:]))  #agdrp

        # return new_states, reward
        return new_state_vector, reward, np.sum(travel_time_[:]), np.sum(tt_veh_[:])/ self.n_vehicle, np.sum(age_[:]), np.sum(flag_vehicle_)
     
        
def normalize_coords(coords, min_val, max_val):
    return (coords - min_val) / (max_val - min_val)




