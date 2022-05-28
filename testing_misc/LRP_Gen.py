import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import DistanceMetric
from math import radians
import geopandas as gpd
from shapely.geometry import Point, LineString
import networkx as nx
from gurobipy import *
import osmnx as ox
ox.config(use_cache=True, log_console=True)


def DC_cfg(DC_N, fixed_cost_DC=[], capacity_DC=[], varCost_DC =[], lat_DC =[], lon_DC=[], osmid =[]):    

    # DC Configurator
    
    # List Creator for DC Names / Index of DF
    def createList_DC(r1, r2):
        return list(['DC%d'%x for x in range(r1, r2+1)])

    #create an index
    DC_ID = createList_DC(1, DC_N)

    # Fixed Cost for opening DC
    fixed_cost_DC = fixed_cost_DC * DC_N

    #Maximum Throughput at DC
    capacity_DC = capacity_DC * DC_N

    #Variable Warehousing Cost (Picking)
    varCost_DC = varCost_DC

    #Position of the DCs
    lat_DC1 = lat_DC
    lon_DC1 = lon_DC

    # Osmid of the Dcs
    osmid1 = osmid

    dc_tuples = list(zip(DC_ID, fixed_cost_DC, capacity_DC, varCost_DC, lat_DC1, lon_DC1, osmid1))

    set_of_all_DC = pd.DataFrame(dc_tuples, columns = ["DC_ID", "fixed_cost_DC", "capacity_DC", "varCost_DC",       "lat", "lon", "osmid"])
    set_of_all_DC.set_index("DC_ID", inplace = True)
    
     # Define Index I for model building  
    DC_cfg.I = set_of_all_DC.index.values 
    
    # Define DC_S for dist/time matrix later
    DC_cfg.DC_S = set_of_all_DC["osmid"].values.tolist()
    
    DC_cfg.set_of_all_DC = set_of_all_DC
    return set_of_all_DC



def C_cfg(Customer_N, demand_per_customer):
    
    # Customer Configurator

    # Function to create a Customer Index List
    def createList_C(r1, r2):
        return list(['C%d'%x for x in range(r1, r2+1)])

    # Get all nodes in Würzburg from OSMNX which will be used to sample a certain amount of customers
    G = ox.graph_from_place("Wuerzburg, Germany", network_type = "drive")
    Gs = ox.utils_graph.get_largest_component(G, strongly = True)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(Gs)

    # Create nodes_df which is the basis for our customer df
    nodes_df = gdf_nodes[["y", "x"]].copy()
    nodes_df.columns = ["lat", "lon"]

    #Sample from nodes_df
    sample_nodes_df = nodes_df.sample(n = Customer_N, random_state= 3)

    #DF Manipulation
    C_ID = createList_C(1,Customer_N)
    osmid = list(sample_nodes_df.index.values)
    sample_nodes_df = sample_nodes_df.rename(index=dict(zip(osmid,C_ID)))
    sample_nodes_df.index.name = "C_ID"
    sample_nodes_df["osmid"] = osmid

    #Create final demand column for customer DF
    mylist = [demand_per_customer] * Customer_N

    set_of_all_customers = sample_nodes_df.copy()
    
    # Add Demand column from the newly created list
    set_of_all_customers['Demand_C'] = mylist
    
    # Sort the df by osmid, this needs to be done for renaming purposes in the dist_matrix
    set_of_all_customers = set_of_all_customers.sort_values(by=["osmid"])
    
    # New index C1-C2 etc.
    set_of_all_customers.reset_index(drop=True, inplace=True)
    set_of_all_customers.index = C_ID
    set_of_all_customers.index.names = ["C_ID"]
    
    # Create Nodes_S for Dist/Time matrix and plotting later
    C_cfg.Nodes_S = set_of_all_customers["osmid"].tolist()
    
    # Create index J for model building
    C_cfg.J = set_of_all_customers.index.values

    C_cfg.set_of_all_customers = set_of_all_customers
    return set_of_all_customers


def V_cfg(Vehicle_N, capacity_V = [], fixed_cost_V = []):

    # Vehicle Configurator

    # Vehicle Index/Name Creator
    def createList_V(r1, r2):
        return list(['V_%d'%x for x in range(r1, r2+1)])

    # create index
    V_ID = createList_V(1, Vehicle_N)

    # Vehicle Capacity Qk
    capacity_V = capacity_V * Vehicle_N

    # fixed cost of using Vehicle Fk
    fixed_cost_V = fixed_cost_V * Vehicle_N

    v_tuples = list(zip(V_ID, capacity_V, fixed_cost_V))

    set_of_all_vehicles = pd.DataFrame(v_tuples, columns = ["V_ID", "capacity_V", "fixed_cost_V"])
    set_of_all_vehicles.set_index("V_ID", inplace=True)
    V_cfg.K = set_of_all_vehicles.index.values
    
    V_cfg.set_of_all_vehicles = set_of_all_vehicles
    return set_of_all_vehicles



def Dist_m():
    
    # Create full Distance Matrix from which we can sample the Customers/PuPs
    G = ox.graph_from_place("Wuerzburg, Germany", network_type = "drive")
    Gs = ox.utils_graph.get_largest_component(G, strongly = True)
    mat_generator = nx.all_pairs_dijkstra_path_length(Gs, weight = "length")
    mat_dict = dict(mat_generator)
    mat = pd.DataFrame(mat_dict).round(1)
    mat = mat.rename_axis("osmid").sort_values(by = ["osmid"])
    
    # Join Distances for the sampled nodes from our Initial Distance Matrix for all nodes
    Dist_m.Nodes_S1 = DC_cfg.DC_S + C_cfg.Nodes_S 
    dist_matrix = pd.DataFrame(index=Dist_m.Nodes_S1)
    
    # Rename the index for the join with the distance matrix
    dist_matrix = dist_matrix.rename_axis("osmid")
    
    # Join new empty dist_matrix with filled matrix for every chosen node
    dist_matrix = dist_matrix.merge(mat, left_index=True, right_index=True)
    dist_matrix = dist_matrix[dist_matrix.columns.intersection(Dist_m.Nodes_S1)]
    
    #Rename Columns to fit the order of Nodes_S1 = DCs  then Customers. Match Index and columns positionwise
    dist_matrix = dist_matrix.reindex(columns=Dist_m.Nodes_S1)

    # Replace column names with I and J variables = DC1, C1 etc.
    nn = np.concatenate((DC_cfg.I,C_cfg.J), axis=None)
    new_colnames = nn.tolist()
    dist_matrix.columns = new_colnames
    dist_matrix. index = new_colnames
    Dist_m.dist_matrix = dist_matrix
    return dist_matrix


def Time_m():
    
    # Add a Time matrix so that we can incorporate traveling times into our routes and cost calculation
    F = ox.graph_from_place("Wuerzburg, Germany", network_type = "drive")
    Fs = ox.utils_graph.get_largest_component(F, strongly = True)
    Fs = ox.add_edge_speeds(Fs)
    Fs = ox.add_edge_travel_times(Fs)
    mat_generator1 = nx.all_pairs_dijkstra_path_length(Fs, weight = "travel_time")
    mat_dict1 = dict(mat_generator1)
    mat1 = pd.DataFrame(mat_dict1).round(1)
    mat1 = mat1.rename_axis("osmid").sort_values(by = ["osmid"])
    
    # Join Distances for the sampled nodes from our Initial Distance Matrix for all nodes 
    time_matrix = pd.DataFrame(index=Dist_m.Nodes_S1)
    
    # Rename the index for the join with the distance matrix
    time_matrix = time_matrix.rename_axis("osmid")
    
    # Join new empty time_matrix with filled matrix for every chosen node
    time_matrix = time_matrix.merge(mat1, left_index=True, right_index=True)
    time_matrix = time_matrix[time_matrix.columns.intersection(Dist_m.Nodes_S1)]
    
    #Rename Columns to fit the order of Nodes_S1 = DCs  then Customers. Match Index and columns positionwise
    time_matrix = time_matrix.reindex(columns=Dist_m.Nodes_S1)

    # Replace column names with I and J variables = DC1, C1 etc.
    nn = np.concatenate((DC_cfg.I, C_cfg.J), axis=None)
    new_colnames = nn.tolist()
    time_matrix.columns = new_colnames
    time_matrix.index = new_colnames
    Time_m.time_matrix = time_matrix
    return time_matrix

def LRP(Gap):
        
    # Define Gurobi Model
    m = Model()
    
    # Define Decision Variables
    x = m.addVars([*DC_cfg.I,*C_cfg.J],[*DC_cfg.I,*C_cfg.J],V_cfg.K, name = "x", vtype=GRB.BINARY)
    y = m.addVars(DC_cfg.I, name = "y", vtype = GRB.BINARY)
    z = m.addVars(DC_cfg.I, C_cfg.J, name = "z", vtype = GRB.BINARY)

    U = [(l,k) for l in C_cfg.J for k in V_cfg.K]
    u = m.addVars(U, name = "u", vtype= GRB.CONTINUOUS)
    
    # Set Math. Notation
    fixedCost_depot = quicksum(y[i] * DC_cfg.set_of_all_DC.loc[i].fixed_cost_DC for i in DC_cfg.I)

    variableCosts_transp = quicksum([(Dist_m.dist_matrix.loc[i,j] + Time_m.time_matrix.loc[i,j]) * 
                                     x[i,j,k] for i in [*DC_cfg.I,*C_cfg.J]
                                     for j in [*DC_cfg.I,*C_cfg.J] for k in V_cfg.K if i!=j])

    variableCosts_DC = quicksum(DC_cfg.set_of_all_DC.loc[i].varCost_DC *
                                z[i,j] * C_cfg.set_of_all_customers.loc[j].Demand_C
                                for j in C_cfg.J for i in DC_cfg.I)

    fixedCost_vehicle = quicksum(x[i,j,k] * V_cfg.set_of_all_vehicles.loc[k].fixed_cost_V
                                 for i in [*DC_cfg.I,*C_cfg.J] for j in [*DC_cfg.I,*C_cfg.J]
                                 for k in V_cfg.K if i!=j)

    obj = fixedCost_depot + variableCosts_transp + variableCosts_DC + fixedCost_vehicle
    
    m.setObjective(obj, GRB.MINIMIZE)
    
    # Set Constraints WIP
    for j in C_cfg.J:
        m.addConstr(quicksum(x[i,j,k] for i in [*DC_cfg.I,*C_cfg.J] for k in V_cfg.K if i!=j) == 1)
    
    for k in V_cfg.K:
        m.addConstr(quicksum(C_cfg.set_of_all_customers.loc[j].Demand_C * x[i,j,k]
                             for i in [*DC_cfg.I,*C_cfg.J] 
                             for j in C_cfg.J if i!=j)
                    <= V_cfg.set_of_all_vehicles.loc[k].capacity_V)

    for l in C_cfg.J:
        for j in C_cfg.J:
            if l!=j:
                for k in V_cfg.K:
                    m.addConstr(u[l,k] - u[j,k] + (len(C_cfg.set_of_all_customers)
                                                   * x[l,j,k]) <= len(C_cfg.set_of_all_customers) -1)

    for i in [*DC_cfg.I,*C_cfg.J]:
        for k in V_cfg.K:
            m.addConstr(quicksum(x[i,j,k] for j in [*DC_cfg.I,*C_cfg.J] if i!=j)
                        - quicksum(x[j,i,k] for j in [*DC_cfg.I,*C_cfg.J] if i!=j) == 0)


    for k in V_cfg.K:
        m.addConstr(quicksum(x[i,j,k] for i in DC_cfg.I for j in C_cfg.J) <= 1)

    for i in DC_cfg.I:
        m.addConstr(quicksum(z[i,j] * C_cfg.set_of_all_customers.loc[j].Demand_C
                             for j in C_cfg.J) - (DC_cfg.set_of_all_DC.loc[i].capacity_DC * y[i]) <= 0)


    for i in DC_cfg.I:
        for j in C_cfg.J:
            for k in V_cfg.K:
                m.addConstr(quicksum(x[i,u,k] + x[u,j,k] for u in [*DC_cfg.I,*C_cfg.J]) - z[i,j] <= 1)
    
    # Set Gap
    m.Params.MIPGap = Gap
    m.update()
    m.Params.LogFile = "gurobi_logee.log"
    m.optimize()
