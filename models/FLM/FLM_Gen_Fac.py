import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
ox.config(use_cache=True, log_console=True)
from gurobipy import *
import warnings
warnings.filterwarnings('ignore')

class FLM:
    
    
    
    def customer_gen(self, customers_n, demand_per_customer, city, seed):
        # Create Distance Matrix containing all nodes of Würzburg
        self.place = {"city": city, "country": "Germany"}
        G = ox.graph_from_place(self.place, network_type = "drive")
        Gs = ox.utils_graph.get_largest_component(G, strongly = True)
        self.Gs = Gs
        mat_gen = nx.all_pairs_dijkstra_path_length(self.Gs, weight = "length")
        mat_dict3 = dict(mat_gen)
        self.mat3 = pd.DataFrame(mat_dict3).round(1)
        self.mat3 = self.mat3.rename_axis("osmid").sort_values(by = ["osmid"])
        
        # DF with all nodes and coordinates
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(Gs)
        
        # Transform to get customers
        nodes_df = gdf_nodes[["y", "x"]].copy()
        nodes_df.columns = ["lat", "lon"]
        
        # Sample Customers
        self.sample_nodes_df = nodes_df.sample(n = customers_n, random_state= seed)
        
        # Create final Customer DF
        self.customers_n = customers_n
        demand_list = [demand_per_customer] * self.customers_n

        set_of_all_customers = self.sample_nodes_df.copy()
        set_of_all_customers['Demand_C'] = demand_list
        self.I = set_of_all_customers.index.values
        self.set_of_all_customers = set_of_all_customers

        # Save all the sample nodes osmid for plotting later
        Nodes_S = self.sample_nodes_df.index.values
        self.Nodes_S = Nodes_S.tolist()
        return self.set_of_all_customers
    
    
    
    def fac_gen(self, capacity_per_fac):
        # Create Pickup-Point Dataframe
        set_of_all_pup = self.sample_nodes_df.copy()
        cap_list = [capacity_per_fac] * self.customers_n
        set_of_all_pup["Capacity_fac"] = cap_list
        self.J = set_of_all_pup.index.values
        self.set_of_all_pup = set_of_all_pup
        return self.set_of_all_pup
    
    
        
    def dist_gen(self):   
        # Join Distances for the sampled nodes from our Initial Distance Matrix for all nodes 
        dist_matrix3 = pd.DataFrame(index=self.Nodes_S)
        dist_matrix3 = dist_matrix3.rename_axis("osmid")
        dist_matrix3 = dist_matrix3.merge(self.mat3, left_index=True, right_index=True)
        dist_matrix3 = dist_matrix3[dist_matrix3.columns.intersection(self.Nodes_S)]
        dist_matrix3 = dist_matrix3.sort_values(by = ["osmid"])
        self.dist_matrix3 = dist_matrix3
        return self.dist_matrix3
    
        
        
    def solver(self, Fac_N, Gap):

        self.model = Model()
        
        def printSolve(scenStr):
            sLen = len(scenStr)
            print("\n" + "*"*sLen + "\n" + scenStr + "\n" + "*"*sLen + "\n")
            
        printSolve("Model is being built, this may take a moment")
        
        # Decision Variables
        self.x = self.model.addVars(self.J, name = "x", vtype = GRB.BINARY)
        self.y = self.model.addVars(self.I, self.J, name = "y", vtype = GRB.BINARY)
        
        self.model.setObjective(quicksum(self.dist_matrix3.loc[i,j] * self.y[i,j] for i in self.I for j in self.J), GRB.MINIMIZE)
        self.model.update()
        # How many Pickup Points do we want
        P = Fac_N
        self.P = P
        
        # Constraints
        for i in self.I:
            self.model.addConstr(quicksum(self.y[i,j] for j in self.J) == 1)
            
        self.model.addConstr(quicksum(self.x[j] for j in self.J) == self.P)
        
        self.model.addConstrs(quicksum(self.set_of_all_customers.loc[i].Demand_C * self.y[i,j] for i in self.I) <= self.set_of_all_pup.loc[j].Capacity_fac * self.x[j] for j in self.J)
        self.model.update()

        self.model.addConstrs(self.y[i,j] <= self.x[j] for j in self.J for i in self.I)
        self.model.update()
                                 
        self.model.Params.MIPGap = Gap                         
        self.model.update()
        self.model.optimize()
        
        
        
    # Facility Printer
    def Fac_Print(self):

        print("The following Facilities are established:")
        for j in self.J:
            if self.x[j].X >= 0.1:
                print("-{}; Coordinates: {}, {}".format(self.set_of_all_pup.loc[j].name,
                                                        self.set_of_all_pup.loc[j].lat,
                                                        self.set_of_all_pup.loc[j].lon))
                
                
                
    # Create Allocation Dataframe which is the basis for most of the Exploratory Analysis            
    def Fac_Alloc(self):
        pup_alloc = []
        for i in self.I:
            for j in self.J:
                if self.x[j].X >= 0.1:
                    if self.y[i,j].X >= 0.1:
                        pup_alloc.append(self.set_of_all_pup.loc[j].name)
        pup_lat = []
        for i in self.I:
            for j in self.J:
                if self.x[j].X >= 0.1:
                    if self.y[i,j].X >= 0.1:
                        pup_lat.append(self.set_of_all_pup.loc[j].lat)

        pup_lon = []
        for i in self.I:
            for j in self.J:
                if self.x[j].X >= 0.1:
                    if self.y[i,j].X >= 0.1:
                        pup_lon.append(self.set_of_all_pup.loc[j].lon)                


        customer_alloc = []
        for i in self.I:
            for j in self.J:
                if self.y[i,j].X >= 0.1:
                    if self.x[j].X >= 0.1:
                        customer_alloc.append(self.set_of_all_customers.loc[i].name)
                        
        customer_lat = []
        for i in self.I:
            for j in self.J:
                if self.y[i,j].X >= 0.1:
                    if self.x[j].X >= 0.1:
                        customer_lat.append(self.set_of_all_customers.loc[i].lat)

        customer_lon = []
        for i in self.I:
            for j in self.J:
                if self.y[i,j].X >= 0.1:
                    if self.x[j].X >= 0.1:
                        customer_lon.append(self.set_of_all_customers.loc[i].lon)

        # Create DF with the lists                
        Alloc_DF = pd.DataFrame(list(zip(pup_alloc,pup_lat, 
                                         pup_lon, customer_alloc, customer_lat, customer_lon)),
                       columns =['Fac_ID',"Fac_lat", "Fac_lon", 'C_ID', "C_LAT", "C_LON"])

        # Shortest Path Function from osmnx/nx
        # G = ox.graph_from_place(self.place, network_type = "drive")
        # self.Fs = ox.utils_graph.get_largest_component(G, strongly = True)
        self.Fs = self.Gs
        self.Fs = ox.add_edge_speeds(self.Fs)
        self.Fs = ox.add_edge_travel_times(self.Fs)

        def shortest_path(Fac_ID, C_ID):
            length = nx.shortest_path_length(self.Fs, source=Fac_ID, target=C_ID, weight= "length")                                 
            return length

        # Create new Distance column with the OSMNX Distances
        Alloc_DF["Distance"] = Alloc_DF.apply(lambda x: shortest_path(x["Fac_ID"],x["C_ID"]), axis=1).round(1)

        # Shortest Path Time Function
        
        def shortest_path_time(Fac_ID, C_ID):
            time = nx.shortest_path_length(self.Fs, source=Fac_ID, target=C_ID, weight= "travel_time")                                 
            return time

        # Add New Time Column
        Alloc_DF["Travel_Time"] = Alloc_DF.apply(lambda x: shortest_path_time(x["Fac_ID"], x["C_ID"]),axis=1)

        Alloc_DF = Alloc_DF.sort_values("Fac_ID")
        self.Alloc_DF = Alloc_DF
        
        # Add Column that has Walking Time from Customer to Fac, assuming a walking speed of 5km/h
        Alloc_DF["Walking_Time"] = (Alloc_DF["Distance"]/1.38888).round(1)
        
        
        # needed later for plotting/df creation
        Location_DF = Alloc_DF.groupby("Fac_ID")["Fac_lat", "Fac_lon"].max()
        self.PuP_Index = Location_DF.index.values
        
        return self.Alloc_DF
    
    
    
    # Function that counts how many Customers are allocated to each Fac
    def Alloc_Count(self):
        
        Alloc_Count = self.Alloc_DF["Fac_ID"].value_counts().rename_axis("Fac_ID").to_frame("Customer_Count")
        return Alloc_Count
    
    
    
    # Print out basic allocation stats regarding distance/time
    def Alloc_Stats(self):
        Alloc_Stats = pd.DataFrame(index=self.PuP_Index)
        Alloc_Stats = Alloc_Stats.rename_axis("Fac_ID")
        Alloc_Stats["Max_Distance"] = self.Alloc_DF.groupby("Fac_ID")["Distance"].max()
        Alloc_Stats["Mean_Distance"] = self.Alloc_DF.groupby("Fac_ID")["Distance"].mean()
        Alloc_Stats["Median_Distance"] = self.Alloc_DF.groupby("Fac_ID")["Distance"].median()
        Alloc_Stats["Max_Time"] = self.Alloc_DF.groupby("Fac_ID")["Travel_Time"].max()
        Alloc_Stats["Mean_Time"] = self.Alloc_DF.groupby("Fac_ID")["Travel_Time"].mean()
        Alloc_Stats["Median_Time"] = self.Alloc_DF.groupby("Fac_ID")["Travel_Time"].median()
        Alloc_Stats = Alloc_Stats.round(1)
        self.Alloc_Stats = Alloc_Stats
        return self.Alloc_Stats
    
    
    
    
    def Alloc_Within(self, Dist1, Dist2, Time1, Time2, Walk_Time1, Walk_Time2):
    # Create Columns which indicate whether allocated Customers are within certain distance/time of assigned Fac
        Alloc_DF_Copy = self.Alloc_DF.copy()
        Alloc_DF_Copy["Within_Dist1"] = np.where(Alloc_DF_Copy["Distance"] <= Dist1, 1, 0)
        Alloc_DF_Copy["Within_Dist2"] = np.where(Alloc_DF_Copy["Distance"] <= Dist2, 1, 0)
        Alloc_DF_Copy["Within_Time1"] = np.where(Alloc_DF_Copy["Travel_Time"] <= Time1, 1, 0)
        Alloc_DF_Copy["Within_Time2"] = np.where(Alloc_DF_Copy["Travel_Time"] <= Time2, 1, 0)
        Alloc_DF_Copy["Within_Walk_Time1"] = np.where(Alloc_DF_Copy["Walking_Time"] <= Walk_Time1, 1, 0)
        Alloc_DF_Copy["Within_Walk_Time2"] = np.where(Alloc_DF_Copy["Walking_Time"] <= Walk_Time2, 1, 0)
        Alloc_DF_Copy = Alloc_DF_Copy.sort_values("Fac_ID")
        
# Create Dataframe which shows percentage of customers within certain distance/time of their assigned Fac
# Group by the Fac id, Sum the helper column "Within" for each specific Fac _ID and divide by Total Customers allocated
# to the Fac to get the percentage of customers within a certain distance/time
        Check_DF = pd.DataFrame(index=self.PuP_Index)
        Check_DF = Check_DF.rename_axis("Fac_ID")
        Check_DF["Within_Time1"] = Alloc_DF_Copy.groupby("Fac_ID")["Within_Time1"].sum()/Alloc_DF_Copy["Fac_ID"].value_counts()*100
        Check_DF["Within_Time2"] = Alloc_DF_Copy.groupby("Fac_ID")["Within_Time2"].sum()/Alloc_DF_Copy["Fac_ID"].value_counts()*100
        Check_DF["Within_Dist1"] = Alloc_DF_Copy.groupby("Fac_ID")["Within_Dist1"].sum()/Alloc_DF_Copy["Fac_ID"].value_counts()*100
        Check_DF["Within_Dist2"] = Alloc_DF_Copy.groupby("Fac_ID")["Within_Dist2"].sum()/Alloc_DF_Copy["Fac_ID"].value_counts()*100
        Check_DF["Within_Walk_Time1"] = Alloc_DF_Copy.groupby("Fac_ID")["Within_Walk_Time1"].sum()/Alloc_DF_Copy["Fac_ID"].value_counts()*100
        Check_DF["Within_Walk_Time2"] = Alloc_DF_Copy.groupby("Fac_ID")["Within_Walk_Time2"].sum()/Alloc_DF_Copy["Fac_ID"].value_counts()*100
        Check_DF = Check_DF.round(1)
        self.Check_DF = Check_DF
        return self.Check_DF
    
    
    
    
    # Basic Plot -> shows where opened Facs / customers are
    def Basic_Alloc_Plot(self):
        
        #place = {"city": city, "country": "Germany"}
        # G = ox.graph_from_place(self.place, network_type = "drive")

        nodes, edges = ox.graph_to_gdfs(self.Gs, nodes=True, edges=True)


        ns = []
        for node in self.Gs.nodes():
            if node in self.PuP_Index:
                ns.append(900)
            elif node in self.Nodes_S:
                ns.append(40)
            else:
                ns.append(0)

        nc = []
        for node in self.Gs.nodes():
            if node in self.PuP_Index: 
                nc.append("red")
            else:
                nc.append("white")

        fig, ax = ox.plot_graph(self.Gs, node_size = ns, edge_linewidth = 0.5, node_color = nc, 
                                figsize = (22,22), bgcolor = "black")
        fig.savefig("result_graph/alb_test123.svg", format = "svg")
        
        
        
        
    # Color Coded Plot. Color every Fac and allocated Customers differently     
    def Colored_Plot(self):
        
            alloc_list = self.Alloc_DF.groupby("Fac_ID")["C_ID"].apply(list)
            flat_list = [item for sublist in alloc_list for item in sublist]

            
            #place = {"city": city, "country": "Germany"}
            # G = ox.graph_from_place(self.place, network_type = "drive")

            nodes, edges = ox.graph_to_gdfs(self.Gs, nodes=True, edges=True)

            ns = []
            for node in self.Gs.nodes():
                if node in self.PuP_Index:
                    ns.append(900)
                elif node in self.Nodes_S:
                    ns.append(40)
                else:
                    ns.append(0)

            P = self.P

            nc = []

            for node in self.Gs.nodes():
                if P == 1 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 1 and node not in flat_list:
                    nc.append("white")

            # 2 Facs        
            for node in self.Gs.nodes():
                if P == 2 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 2 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 2 and node not in flat_list:
                    nc.append("white")

            # 3 Facs        
            for node in self.Gs.nodes():
                if P == 3 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P== 3 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P==3 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 3 and node not in flat_list:
                    nc.append("white")

            # 4 Facs

            for node in self.Gs.nodes():
                if P == 4 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 4 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 4 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 4 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 4 and node not in flat_list:
                    nc.append("white")

            # 5 Facs

            for node in self.Gs.nodes():
                if P == 5 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 5 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 5 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 5 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 5 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 5 and node not in flat_list:
                    nc.append("white")

            # 6 Facs
            for node in self.Gs.nodes():
                if P == 6 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 6 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 6 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 6 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 6 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 6 and node in alloc_list.iloc[P-6]:
                    nc.append("magenta")
                elif P == 6 and node not in flat_list:
                    nc.append("white")

            # 7 Facs
            for node in self.Gs.nodes():
                if P == 7 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 7 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 7 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 7 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 7 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 7 and node in alloc_list.iloc[P-6]:
                    nc.append("magenta")
                elif P == 7 and node in alloc_list.iloc[P-7]:
                    nc.append("#FB3005")    
                elif P == 7 and node not in flat_list:
                    nc.append("white")

            # 8 Facs
            for node in self.Gs.nodes():
                if P == 8 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 8 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 8 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 8 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 8 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 8 and node in alloc_list.iloc[P-6]:
                    nc.append("magenta")
                elif P == 8 and node in alloc_list.iloc[P-7]:
                    nc.append("#FB3005")
                elif P == 8 and node in alloc_list.iloc[P-8]:
                    nc.append("#784212")
                elif P == 8 and node not in flat_list:
                    nc.append("white")

            # 9 Facs
            for node in self.Gs.nodes():
                if P == 9 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 9 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 9 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 9 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 9 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 9 and node in alloc_list.iloc[P-6]:
                    nc.append("magenta")
                elif P == 9 and node in alloc_list.iloc[P-7]:
                    nc.append("#FB3005")
                elif P == 9 and node in alloc_list.iloc[P-8]:
                    nc.append("#794212")
                elif P == 9 and node in alloc_list.iloc[P-9]:
                    nc.append("#05D6FB")
                elif P == 9 and node not in flat_list:
                    nc.append("white")

            # 10 Facs
            for node in self.Gs.nodes():
                if P == 10 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 10 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 10 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 10 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 10 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 10 and node in alloc_list.iloc[P-6]:
                    nc.append("magenta")
                elif P == 10 and node in alloc_list.iloc[P-7]:
                    nc.append("#FB3005")
                elif P == 10 and node in alloc_list.iloc[P-8]:
                    nc.append("#794212")
                elif P == 10 and node in alloc_list.iloc[P-9]:
                    nc.append("#05D6FB")
                elif P == 10 and node in alloc_list.iloc[P-10]:
                    nc.append("#05FBB0")
                elif P == 10 and node not in flat_list:
                    nc.append("white")

            # 11 Facs
            for node in self.Gs.nodes():
                if P == 11 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 11 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 11 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 11 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 11 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 11 and node in alloc_list.iloc[P-6]:
                    nc.append("magenta")
                elif P == 11 and node in alloc_list.iloc[P-7]:
                    nc.append("#FB3005")
                elif P == 11 and node in alloc_list.iloc[P-8]:
                    nc.append("#794212")
                elif P == 11 and node in alloc_list.iloc[P-9]:
                    nc.append("#05D6FB")
                elif P == 11 and node in alloc_list.iloc[P-10]:
                    nc.append("#05FBB0")
                elif P == 11 and node in alloc_list.iloc[P-11]:
                    nc.append("#FBB405")
                elif P == 11 and node not in flat_list:
                    nc.append("white")

            # 12 Facs
            for node in self.Gs.nodes():
                if P == 12 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 12 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 12 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 12 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 12 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 12 and node in alloc_list.iloc[P-6]:
                    nc.append("magenta")
                elif P == 12 and node in alloc_list.iloc[P-7]:
                    nc.append("#FB3005")
                elif P == 12 and node in alloc_list.iloc[P-8]:
                    nc.append("#794212")
                elif P == 12 and node in alloc_list.iloc[P-9]:
                    nc.append("#05D6FB")
                elif P == 12 and node in alloc_list.iloc[P-10]:
                    nc.append("#05FBB0")
                elif P == 12 and node in alloc_list.iloc[P-11]:
                    nc.append("#FBB405")
                elif P == 12 and node in alloc_list.iloc[P-12]:
                    nc.append("#5705FB")
                elif P == 12 and node not in flat_list:
                    nc.append("white")

            # 13 Facs
            for node in self.Gs.nodes():
                if P == 13 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 13 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 13 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 13 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 13 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 13 and node in alloc_list.iloc[P-6]:
                    nc.append("magenta")
                elif P == 13 and node in alloc_list.iloc[P-7]:
                    nc.append("#FB3005")
                elif P == 13 and node in alloc_list.iloc[P-8]:
                    nc.append("#794212")
                elif P == 13 and node in alloc_list.iloc[P-9]:
                    nc.append("#05D6FB")
                elif P == 13 and node in alloc_list.iloc[P-10]:
                    nc.append("#05FBB0")
                elif P == 13 and node in alloc_list.iloc[P-11]:
                    nc.append("#FBB405")
                elif P == 13 and node in alloc_list.iloc[P-12]:
                    nc.append("#5705FB")
                elif P == 13 and node in alloc_list.iloc[P-13]:
                    nc.append("#A5FB05")
                elif P == 13 and node not in flat_list:
                    nc.append("white")

            # 14 Facs
            for node in self.Gs.nodes():
                if P == 14 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 14 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 14 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 14 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 14 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 14 and node in alloc_list.iloc[P-6]:
                    nc.append("magenta")
                elif P == 14 and node in alloc_list.iloc[P-7]:
                    nc.append("#FB3005")
                elif P == 14 and node in alloc_list.iloc[P-8]:
                    nc.append("#794212")
                elif P == 14 and node in alloc_list.iloc[P-9]:
                    nc.append("#05D6FB")
                elif P == 14 and node in alloc_list.iloc[P-10]:
                    nc.append("#05FBB0")
                elif P == 14 and node in alloc_list.iloc[P-11]:
                    nc.append("#FBB405")
                elif P == 14 and node in alloc_list.iloc[P-12]:
                    nc.append("#5705FB")
                elif P == 14 and node in alloc_list.iloc[P-13]:
                    nc.append("#A5FB05")
                elif P == 14 and node in alloc_list.iloc[P-14]:
                    nc.append("#FB0505")
                elif P == 14 and node not in flat_list:
                    nc.append("white")

            # 15 Facs
            for node in self.Gs.nodes():
                if P == 15 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 15 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 15 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 15 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 15 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 15 and node in alloc_list.iloc[P-6]:
                    nc.append("magenta")
                elif P == 15 and node in alloc_list.iloc[P-7]:
                    nc.append("#FB3005")
                elif P == 15 and node in alloc_list.iloc[P-8]:
                    nc.append("#794212")
                elif P == 15 and node in alloc_list.iloc[P-9]:
                    nc.append("#05D6FB")
                elif P == 15 and node in alloc_list.iloc[P-10]:
                    nc.append("#05FBB0")
                elif P == 15 and node in alloc_list.iloc[P-11]:
                    nc.append("#FBB405")
                elif P == 15 and node in alloc_list.iloc[P-12]:
                    nc.append("#5705FB")
                elif P == 15 and node in alloc_list.iloc[P-13]:
                    nc.append("#A5FB05")
                elif P == 15 and node in alloc_list.iloc[P-14]:
                    nc.append("#FB0505")
                elif P == 15 and node in alloc_list.iloc[P-15]:
                    nc.append("#784212")
                elif P == 15 and node not in flat_list:
                    nc.append("white")

            # 16 Facs
            for node in self.Gs.nodes():
                if P == 16 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 16 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 16 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 16 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 16 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 16 and node in alloc_list.iloc[P-6]:
                    nc.append("magenta")
                elif P == 16 and node in alloc_list.iloc[P-7]:
                    nc.append("#FB3005")
                elif P == 16 and node in alloc_list.iloc[P-8]:
                    nc.append("#794212")
                elif P == 16 and node in alloc_list.iloc[P-9]:
                    nc.append("#05D6FB")
                elif P == 16 and node in alloc_list.iloc[P-10]:
                    nc.append("#05FBB0")
                elif P == 16 and node in alloc_list.iloc[P-11]:
                    nc.append("#FBB405")
                elif P == 16 and node in alloc_list.iloc[P-12]:
                    nc.append("#5705FB")
                elif P == 16 and node in alloc_list.iloc[P-13]:
                    nc.append("#A5FB05")
                elif P == 16 and node in alloc_list.iloc[P-14]:
                    nc.append("#FB0505")
                elif P == 16 and node in alloc_list.iloc[P-15]:
                    nc.append("#784212")
                elif P == 16 and node in alloc_list.iloc[P-16]:
                    nc.append("#5A91B9")
                elif P == 16 and node not in flat_list:
                    nc.append("white")

            # 17 Facs
            for node in self.Gs.nodes():
                if P == 17 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 17 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 17 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 17 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 17 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 17 and node in alloc_list.iloc[P-6]:
                    nc.append("magenta")
                elif P == 17 and node in alloc_list.iloc[P-7]:
                    nc.append("#FB3005")
                elif P == 17 and node in alloc_list.iloc[P-8]:
                    nc.append("#794212")
                elif P == 17 and node in alloc_list.iloc[P-9]:
                    nc.append("#05D6FB")
                elif P == 17 and node in alloc_list.iloc[P-10]:
                    nc.append("#05FBB0")
                elif P == 17 and node in alloc_list.iloc[P-11]:
                    nc.append("#FBB405")
                elif P == 17 and node in alloc_list.iloc[P-12]:
                    nc.append("#5705FB")
                elif P == 17 and node in alloc_list.iloc[P-13]:
                    nc.append("#A5FB05")
                elif P == 17 and node in alloc_list.iloc[P-14]:
                    nc.append("#FB0505")
                elif P == 17 and node in alloc_list.iloc[P-15]:
                    nc.append("#784212")
                elif P == 17 and node in alloc_list.iloc[P-16]:
                    nc.append("#5A91B9")
                elif P == 17 and node in alloc_list.iloc[P-17]:
                    nc.append("#747255")
                elif P == 17 and node not in flat_list:
                    nc.append("white")

            # 18 Facs
            for node in self.Gs.nodes():
                if P == 18 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 18 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 18 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 18 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 18 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 18 and node in alloc_list.iloc[P-6]:
                    nc.append("magenta")
                elif P == 18 and node in alloc_list.iloc[P-7]:
                    nc.append("#FB3005")
                elif P == 18 and node in alloc_list.iloc[P-8]:
                    nc.append("#794212")
                elif P == 18 and node in alloc_list.iloc[P-9]:
                    nc.append("#05D6FB")
                elif P == 18 and node in alloc_list.iloc[P-10]:
                    nc.append("#05FBB0")
                elif P == 18 and node in alloc_list.iloc[P-11]:
                    nc.append("#FBB405")
                elif P == 18 and node in alloc_list.iloc[P-12]:
                    nc.append("#5705FB")
                elif P == 18 and node in alloc_list.iloc[P-13]:
                    nc.append("#A5FB05")
                elif P == 18 and node in alloc_list.iloc[P-14]:
                    nc.append("#FB0505")
                elif P == 18 and node in alloc_list.iloc[P-15]:
                    nc.append("#784212")
                elif P == 18 and node in alloc_list.iloc[P-16]:
                    nc.append("#5A91B9")
                elif P == 18 and node in alloc_list.iloc[P-17]:
                    nc.append("#747255")
                elif P == 18 and node in alloc_list.iloc[P-18]:
                    nc.append("#834D7A")
                elif P == 18 and node not in flat_list:
                    nc.append("white")

            # 19 Facs
            for node in self.Gs.nodes():
                if P == 19 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 19 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 19 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 19 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 19 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 19 and node in alloc_list.iloc[P-6]:
                    nc.append("magenta")
                elif P == 19 and node in alloc_list.iloc[P-7]:
                    nc.append("#FB3005")
                elif P == 19 and node in alloc_list.iloc[P-8]:
                    nc.append("#794212")
                elif P == 19 and node in alloc_list.iloc[P-9]:
                    nc.append("#05D6FB")
                elif P == 19 and node in alloc_list.iloc[P-10]:
                    nc.append("#05FBB0")
                elif P == 19 and node in alloc_list.iloc[P-11]:
                    nc.append("#FBB405")
                elif P == 19 and node in alloc_list.iloc[P-12]:
                    nc.append("#5705FB")
                elif P == 19 and node in alloc_list.iloc[P-13]:
                    nc.append("#A5FB05")
                elif P == 19 and node in alloc_list.iloc[P-14]:
                    nc.append("#FB0505")
                elif P == 19 and node in alloc_list.iloc[P-15]:
                    nc.append("#784212")
                elif P == 19 and node in alloc_list.iloc[P-16]:
                    nc.append("#5A91B9")
                elif P == 19 and node in alloc_list.iloc[P-17]:
                    nc.append("#747255")
                elif P == 19 and node in alloc_list.iloc[P-18]:
                    nc.append("#834D7A")
                elif P == 19 and node in alloc_list.iloc[P-19]:
                    nc.append("#E47873")
                elif P == 19 and node not in flat_list:
                    nc.append("white")

            # 20 Facs
            for node in self.Gs.nodes():
                if P == 20 and node in alloc_list.iloc[P-1]:
                    nc.append("orange")
                elif P == 20 and node in alloc_list.iloc[P-2]:
                    nc.append("green")
                elif P == 20 and node in alloc_list.iloc[P-3]:
                    nc.append("purple")
                elif P == 20 and node in alloc_list.iloc[P-4]:
                    nc.append("red")
                elif P == 20 and node in alloc_list.iloc[P-5]:
                    nc.append("yellow")
                elif P == 20 and node in alloc_list.iloc[P-6]:
                    nc.append("magenta")
                elif P == 20 and node in alloc_list.iloc[P-7]:
                    nc.append("#FB3005")
                elif P == 20 and node in alloc_list.iloc[P-8]:
                    nc.append("#794212")
                elif P == 20 and node in alloc_list.iloc[P-9]:
                    nc.append("#05D6FB")
                elif P == 20 and node in alloc_list.iloc[P-10]:
                    nc.append("#05FBB0")
                elif P == 20 and node in alloc_list.iloc[P-11]:
                    nc.append("#FBB405")
                elif P == 20 and node in alloc_list.iloc[P-12]:
                    nc.append("#5705FB")
                elif P == 20 and node in alloc_list.iloc[P-13]:
                    nc.append("#A5FB05")
                elif P == 20 and node in alloc_list.iloc[P-14]:
                    nc.append("#FB0505")
                elif P == 20 and node in alloc_list.iloc[P-15]:
                    nc.append("#784212")
                elif P == 20 and node in alloc_list.iloc[P-16]:
                    nc.append("#5A91B9")
                elif P == 20 and node in alloc_list.iloc[P-17]:
                    nc.append("#747255")
                elif P == 20 and node in alloc_list.iloc[P-18]:
                    nc.append("#834D7A")
                elif P == 20 and node in alloc_list.iloc[P-19]:
                    nc.append("#E47873")
                elif P == 20 and node in alloc_list.iloc[P-20]:
                    nc.append("#789B70")
                elif P == 20 and node not in flat_list:
                    nc.append("white")

            fig, ax = ox.plot_graph(self.Gs, node_size = ns, edge_linewidth = 0.5,
                                    node_color = nc, figsize = (22,22), bgcolor = "black")
            fig.savefig("result_graph/alb_color_test123.svg", format = "svg")