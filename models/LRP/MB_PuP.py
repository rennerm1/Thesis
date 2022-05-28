import numpy as np
import pandas as pd
import networkx as nx
from gurobipy import *
import osmnx as ox
ox.config(use_cache=True, log_console=True)
import warnings
warnings.filterwarnings('ignore')

class LRP:
    
    
    
    def DC_cfg(self, DC_N, customer_N, fixed_cost_factor, avg_basket_size, capacity_DC=[], varCost_DC =[], lat_DC =[], lon_DC=[], osmid =[]):    

        # DC Configurator
        
        # List Creator for DC Names / Index of DF
        def createList_DC(r1, r2):
            return list(['DC%d'%x for x in range(r1, r2+1)])

        #create an index
        DC_ID = createList_DC(1, DC_N)

        # Fixed Cost for opening DC
        fixed_cost_DC = [(customer_N * avg_basket_size) * fixed_cost_factor] * DC_N

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

        set_of_all_DC = pd.DataFrame(dc_tuples, columns = ["DC_ID", "fixed_cost_DC", "capacity_DC", "varCost_DC", "lat", "lon", "osmid"])
        set_of_all_DC.set_index("DC_ID", inplace = True)

         # Define Index I for model building  
        self.I = set_of_all_DC.index.values 

        # Define DC_S for dist/time matrix later
        self.DC_S = set_of_all_DC["osmid"].values.tolist()

        self.set_of_all_DC = set_of_all_DC
        return set_of_all_DC
    
    
    def C_cfg(self, scenario, city, mean_demand, max_demand, min_demand, seed_demand, customers, pup_n, seed_customers, seed_pup, demand_p, seed_demand_p):
    
        # Customer Configurator

        # Function to create a Customer Index List
        def createList_C(r1, r2):
            return list(['C%d'%x for x in range(r1, r2+1)])

        self.customers = customers
        self.pups = pup_n
        self.city = city
        if self.city == "Albstadt" and scenario == "low":
            nodes_df = pd.read_hdf("alb_low.h5", "alb_low")
            sample_nodes_df = nodes_df

            #DF Manipulation
            C_ID = createList_C(1,len(nodes_df))
            osmid = list(sample_nodes_df.index.values)
            sample_nodes_df = sample_nodes_df.rename(index=dict(zip(osmid,C_ID)))
            sample_nodes_df.index.name = "C_ID"
            sample_nodes_df["osmid"] = osmid

            # Define Demand Scenarios
            if demand_p == "very_low":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(125,275,375, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "low":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(410,560,710, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "mid":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(1025, 1175, 1325, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "high":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(1600,1750,1900, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            else:
                None

            set_of_all_customers = sample_nodes_df.copy()

            # Add Demand column from the newly created list
            set_of_all_customers['Demand_C'] = tri_demand_p

            # Sort the df by osmid, this needs to be done for renaming purposes in the dist_matrix
            set_of_all_customers = set_of_all_customers.sort_values(by=["osmid"])

            # New index C1-C2 etc.
            set_of_all_customers.reset_index(drop=True, inplace=True)
            set_of_all_customers.index = C_ID
            set_of_all_customers.index.names = ["C_ID"]

            # Create Nodes_S for Dist/Time matrix and plotting later
            self.Nodes_S = set_of_all_customers["osmid"].tolist()

            # Create index J for model building
            self.J = set_of_all_customers.index.values

            self.set_of_all_customers = set_of_all_customers
            
            
            
        # Repeat this process for every scenario and Test market    
        elif self.city == "Albstadt" and scenario == "mid":
            # Load PULs for specific scenario
            nodes_df = pd.read_hdf("alb_mid.h5", "alb_mid")
            sample_nodes_df = nodes_df
            #DF Manipulation
            C_ID = createList_C(1,len(nodes_df))
            osmid = list(sample_nodes_df.index.values)
            sample_nodes_df = sample_nodes_df.rename(index=dict(zip(osmid,C_ID)))
            sample_nodes_df.index.name = "C_ID"
            sample_nodes_df["osmid"] = osmid

            # Define Demand Scenarios
            if demand_p == "very_low":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(5,140,270, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "low":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(135,285,435, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "mid":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(440, 590, 740, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "high":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(735,885,1035, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            else:
                None


            set_of_all_customers = sample_nodes_df.copy()

            # Add Demand column from the newly created list
            set_of_all_customers['Demand_C'] = tri_demand_p

            # Sort the df by osmid, this needs to be done for renaming purposes in the dist_matrix
            set_of_all_customers = set_of_all_customers.sort_values(by=["osmid"])

            # New index C1-C2 etc.
            set_of_all_customers.reset_index(drop=True, inplace=True)
            set_of_all_customers.index = C_ID
            set_of_all_customers.index.names = ["C_ID"]

            # Create Nodes_S for Dist/Time matrix and plotting later
            self.Nodes_S = set_of_all_customers["osmid"].tolist()

            # Create index J for model building
            self.J = set_of_all_customers.index.values

            self.set_of_all_customers = set_of_all_customers
            
            
        elif self.city == "Albstadt" and scenario == "high":
            # Load PULs for specific scenario
            nodes_df = pd.read_hdf("alb_high.h5", "alb_high")
            sample_nodes_df = nodes_df
            #DF Manipulation
            C_ID = createList_C(1,len(nodes_df))
            osmid = list(sample_nodes_df.index.values)
            sample_nodes_df = sample_nodes_df.rename(index=dict(zip(osmid,C_ID)))
            sample_nodes_df.index.name = "C_ID"
            sample_nodes_df["osmid"] = osmid

            # Define Demand Scenarios
            if demand_p == "very_low":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(5,80,155, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "low":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(10,165,310, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "mid":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(190, 340, 490, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "high":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(355,505,655, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            else:
                None

            set_of_all_customers = sample_nodes_df.copy()

            # Add Demand column from the newly created list
            set_of_all_customers['Demand_C'] = tri_demand_p

            # Sort the df by osmid, this needs to be done for renaming purposes in the dist_matrix
            set_of_all_customers = set_of_all_customers.sort_values(by=["osmid"])

            # New index C1-C2 etc.
            set_of_all_customers.reset_index(drop=True, inplace=True)
            set_of_all_customers.index = C_ID
            set_of_all_customers.index.names = ["C_ID"]

            # Create Nodes_S for Dist/Time matrix and plotting later
            self.Nodes_S = set_of_all_customers["osmid"].tolist()

            # Create index J for model building
            self.J = set_of_all_customers.index.values

            self.set_of_all_customers = set_of_all_customers
        elif self.city  == "Wuerzburg" and scenario == "low":
            # Load PULs for specific scenario
            nodes_df = pd.read_hdf("wue_low.h5", "wue_low")
            sample_nodes_df = nodes_df
            #DF Manipulation
            C_ID = createList_C(1,len(nodes_df))
            osmid = list(sample_nodes_df.index.values)
            sample_nodes_df = sample_nodes_df.rename(index=dict(zip(osmid,C_ID)))
            sample_nodes_df.index.name = "C_ID"
            sample_nodes_df["osmid"] = osmid

            # Define Demand Scenarios
            if demand_p == "very_low":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(35,185,335, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "low":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(230,380,530, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "mid":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(630, 780, 930, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "high":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(1030,1180,1330, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            else:
                None


            set_of_all_customers = sample_nodes_df.copy()

            # Add Demand column from the newly created list
            set_of_all_customers['Demand_C'] = tri_demand_p

            # Sort the df by osmid, this needs to be done for renaming purposes in the dist_matrix
            set_of_all_customers = set_of_all_customers.sort_values(by=["osmid"])

            # New index C1-C2 etc.
            set_of_all_customers.reset_index(drop=True, inplace=True)
            set_of_all_customers.index = C_ID
            set_of_all_customers.index.names = ["C_ID"]

            # Create Nodes_S for Dist/Time matrix and plotting later
            self.Nodes_S = set_of_all_customers["osmid"].tolist()

            # Create index J for model building
            self.J = set_of_all_customers.index.values

            self.set_of_all_customers = set_of_all_customers
        elif self.city  == "Wuerzburg" and scenario == "mid":
            # Load PULs for specific scenario
            nodes_df = pd.read_hdf("wue_mid.h5", "wue_mid")
            sample_nodes_df = nodes_df
            #DF Manipulation
            C_ID = createList_C(1,len(nodes_df))
            osmid = list(sample_nodes_df.index.values)
            sample_nodes_df = sample_nodes_df.rename(index=dict(zip(osmid,C_ID)))
            sample_nodes_df.index.name = "C_ID"
            sample_nodes_df["osmid"] = osmid

            # Define Demand Scenarios
            if demand_p == "very_low":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(5,110,215, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "low":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(80,230,380, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "mid":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(320, 470, 620, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "high":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(560,710,860, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            else:
                None


            set_of_all_customers = sample_nodes_df.copy()

            # Add Demand column from the newly created list
            set_of_all_customers['Demand_C'] = tri_demand_p

            # Sort the df by osmid, this needs to be done for renaming purposes in the dist_matrix
            set_of_all_customers = set_of_all_customers.sort_values(by=["osmid"])

            # New index C1-C2 etc.
            set_of_all_customers.reset_index(drop=True, inplace=True)
            set_of_all_customers.index = C_ID
            set_of_all_customers.index.names = ["C_ID"]

            # Create Nodes_S for Dist/Time matrix and plotting later
            self.Nodes_S = set_of_all_customers["osmid"].tolist()

            # Create index J for model building
            self.J = set_of_all_customers.index.values

            self.set_of_all_customers = set_of_all_customers
        elif self.city  == "Wuerzburg" and scenario == "high":
            # Load PULs for specific scenario
            nodes_df = pd.read_hdf("wue_high.h5", "wue_high")
            sample_nodes_df = nodes_df
            #DF Manipulation
            C_ID = createList_C(1,len(nodes_df))
            osmid = list(sample_nodes_df.index.values)
            sample_nodes_df = sample_nodes_df.rename(index=dict(zip(osmid,C_ID)))
            sample_nodes_df.index.name = "C_ID"
            sample_nodes_df["osmid"] = osmid

            # Define Demand Scenarios
            if demand_p == "very_low":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(5,45,85, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "low":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(5,90,175, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "mid":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(50, 165, 315, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            elif demand_p == "high":
                tri_demand_p = np.random.default_rng(seed_demand_p).triangular(135,285,435, len(sample_nodes_df))
                tri_demand_p = tri_demand_p.round(0)
                tri_demand_p = tri_demand_p.tolist()
            else:
                None

            set_of_all_customers = sample_nodes_df.copy()

            # Add Demand column from the newly created list
            set_of_all_customers['Demand_C'] = tri_demand_p

            # Sort the df by osmid, this needs to be done for renaming purposes in the dist_matrix
            set_of_all_customers = set_of_all_customers.sort_values(by=["osmid"])

            # New index C1-C2 etc.
            set_of_all_customers.reset_index(drop=True, inplace=True)
            set_of_all_customers.index = C_ID
            set_of_all_customers.index.names = ["C_ID"]

            # Create Nodes_S for Dist/Time matrix and plotting later
            self.Nodes_S = set_of_all_customers["osmid"].tolist()

            # Create index J for model building
            self.J = set_of_all_customers.index.values

            self.set_of_all_customers = set_of_all_customers
            
            
        # Mixed Scenarios include PULs and Regular Customers
        elif self.city == "Wuerzburg" and scenario == "mixed_low":
            # Load Customers and PULs for specific scenario
            nodes_df = pd.read_hdf("nodes_df_wue", "nodes_df")
            pup_df = pd.read_hdf("wue_low.h5", "wue_low")
            pup_index = pup_df.index.values.tolist()
         
            samp_first = nodes_df.sample(n= 190, random_state=3)
            node_list = samp_first.index.values.tolist()
            
            # Drop necessary nodes so that we dont have duplicates for pups and customers
            intersection_set = set.intersection(set(node_list), set(pup_index))
            intersection_list = list(intersection_set)

            samp_first = samp_first.drop(index = intersection_list)              
            # Sample Customers and PuPs                
            sample_nodes_df = samp_first.sample(n = self.customers, random_state = seed_customers)
            pup_sample = pup_df.sample(n = self.pups, random_state = seed_pup)
            
            # Demand for customers
            tri_demand_c = [mean_demand] * len(sample_nodes_df)
            sample_nodes_df["Demand_C"] = tri_demand_c
            
            # Define Demand Scenarios for PuPs
            tri_demand_p = [demand_p] * len(pup_sample)
                
            pup_sample["Demand_C"] = tri_demand_p
            
            
            
            # Join the two together
            self.Nodes_C = sample_nodes_df.index.values.tolist()
            sample_nodes_df = pd.concat([sample_nodes_df, pup_sample])
            self.Nodes_P = pup_sample.index.values.tolist()
            
            # Change index to Customer ID and create osmid column
            C_ID = createList_C(1, len(sample_nodes_df))
            osmid = list(sample_nodes_df.index.values)
            sample_nodes_df = sample_nodes_df.rename(index=dict(zip(osmid,C_ID)))
            sample_nodes_df.index.name = "C_ID"
            sample_nodes_df["osmid"] = osmid
            
            # Create set_of_all_customers final df
            set_of_all_customers = sample_nodes_df.copy()
            set_of_all_customers = set_of_all_customers.sort_values(by=["osmid"])
            set_of_all_customers.reset_index(drop = True, inplace = True)
            set_of_all_customers.index = C_ID
            set_of_all_customers.index.names = ["C_ID"]
            self.Nodes_S = set_of_all_customers["osmid"].tolist()
            self.J = set_of_all_customers.index.values
            self.set_of_all_customers = set_of_all_customers
            
            
        elif self.city == "Wuerzburg" and scenario == "mixed_mid":
            # Load Customers and PULs for specific scenario
            nodes_df = pd.read_hdf("nodes_df_wue", "nodes_df")
            pup_df = pd.read_hdf("wue_mid.h5", "wue_mid")
            pup_index = pup_df.index.values.tolist()
         
            samp_first = nodes_df.sample(n= 190, random_state=3)
            node_list = samp_first.index.values.tolist()
            
            # Drop necessary nodes so that we dont have duplicates for pups and customers
            intersection_set = set.intersection(set(node_list), set(pup_index))
            intersection_list = list(intersection_set)

            samp_first = samp_first.drop(index = intersection_list)              
            # Sample Customers and PuPs                
            sample_nodes_df = samp_first.sample(n = self.customers, random_state = seed_customers)
            pup_sample = pup_df.sample(n = self.pups, random_state = seed_pup)
            
            # Demand for customers
            tri_demand_c = [mean_demand] * len(sample_nodes_df)
            sample_nodes_df["Demand_C"] = tri_demand_c
            
            # Define Demand Scenarios for PuPs
            tri_demand_p = [demand_p] * len(pup_sample)
                
            pup_sample["Demand_C"] = tri_demand_p
            
            
            
            # Join the two together
            self.Nodes_C = sample_nodes_df.index.values.tolist()
            sample_nodes_df = pd.concat([sample_nodes_df, pup_sample])
            self.Nodes_P = pup_sample.index.values.tolist()
            
            # Change index to Customer ID and create osmid column
            C_ID = createList_C(1, len(sample_nodes_df))
            osmid = list(sample_nodes_df.index.values)
            sample_nodes_df = sample_nodes_df.rename(index=dict(zip(osmid,C_ID)))
            sample_nodes_df.index.name = "C_ID"
            sample_nodes_df["osmid"] = osmid
            
            # Create set_of_all_customers final df
            set_of_all_customers = sample_nodes_df.copy()
            set_of_all_customers = set_of_all_customers.sort_values(by=["osmid"])
            set_of_all_customers.reset_index(drop = True, inplace = True)
            set_of_all_customers.index = C_ID
            set_of_all_customers.index.names = ["C_ID"]
            self.Nodes_S = set_of_all_customers["osmid"].tolist()
            self.J = set_of_all_customers.index.values
            self.set_of_all_customers = set_of_all_customers
        
        elif self.city == "Wuerzburg" and scenario == "mixed_high":
            # Load Customers and PULs for specific scenario
            nodes_df = pd.read_hdf("nodes_df_wue", "nodes_df")
            pup_df = pd.read_hdf("wue_high.h5", "wue_high")
            pup_index = pup_df.index.values.tolist()
         
            samp_first = nodes_df.sample(n= 190, random_state=3)
            node_list = samp_first.index.values.tolist()
            
            # Drop necessary nodes so that we dont have duplicates for pups and customers
            intersection_set = set.intersection(set(node_list), set(pup_index))
            intersection_list = list(intersection_set)

            samp_first = samp_first.drop(index = intersection_list)              
            # Sample Customers and PuPs                
            sample_nodes_df = samp_first.sample(n = self.customers, random_state = seed_customers)
            pup_sample = pup_df.sample(n = self.pups, random_state = seed_pup)
            
            # Demand for customers
            tri_demand_c = [mean_demand] * len(sample_nodes_df)
            sample_nodes_df["Demand_C"] = tri_demand_c
            
            # Define Demand Scenarios for PuPs
            tri_demand_p = [demand_p] * len(pup_sample)
                
            pup_sample["Demand_C"] = tri_demand_p
            
            
            
            # Join the two together
            self.Nodes_C = sample_nodes_df.index.values.tolist()
            sample_nodes_df = pd.concat([sample_nodes_df, pup_sample])
            self.Nodes_P = pup_sample.index.values.tolist()
            
            # Change index to Customer ID and create osmid column
            C_ID = createList_C(1, len(sample_nodes_df))
            osmid = list(sample_nodes_df.index.values)
            sample_nodes_df = sample_nodes_df.rename(index=dict(zip(osmid,C_ID)))
            sample_nodes_df.index.name = "C_ID"
            sample_nodes_df["osmid"] = osmid
            
            # Create set_of_all_customers final df
            set_of_all_customers = sample_nodes_df.copy()
            set_of_all_customers = set_of_all_customers.sort_values(by=["osmid"])
            set_of_all_customers.reset_index(drop = True, inplace = True)
            set_of_all_customers.index = C_ID
            set_of_all_customers.index.names = ["C_ID"]
            self.Nodes_S = set_of_all_customers["osmid"].tolist()
            self.J = set_of_all_customers.index.values
            self.set_of_all_customers = set_of_all_customers
            
        elif self.city == "Albstadt" and scenario == "mixed_low":
            # Load Customers and PULs for specific scenario
            nodes_df = pd.read_hdf("nodes_df_alb", "nodes_df")
            pup_df = pd.read_hdf("alb_low.h5", "alb_low")
            pup_index = pup_df.index.values.tolist()
         
            samp_first = nodes_df.sample(n= 191, random_state=3)
            samp_first = samp_first.drop(index = 290847730)
            node_list = samp_first.index.values.tolist()
            
            # Drop necessary nodes so that we dont have duplicates for pups and customers
            intersection_set = set.intersection(set(node_list), set(pup_index))
            intersection_list = list(intersection_set)

            samp_first = samp_first.drop(index = intersection_list)              
            # Sample Customers and PuPs                
            sample_nodes_df = samp_first.sample(n = self.customers, random_state = seed_customers)
            pup_sample = pup_df.sample(n = self.pups, random_state = seed_pup)
            self.Nodes_P = pup_sample.index.values.tolist()
            # Demand for customers
            tri_demand_c = [mean_demand] * len(sample_nodes_df)
            sample_nodes_df["Demand_C"] = tri_demand_c
            
            # Define Demand Scenarios for PuPs
            tri_demand_p = [demand_p] * len(pup_sample)
   
            pup_sample["Demand_C"] = tri_demand_p
            
            self.pup_sample = pup_sample
            
            # Join the two together
            sample_nodes_df = pd.concat([sample_nodes_df, self.pup_sample])

            
            # Change index to Customer ID and create osmid column
            C_ID = createList_C(1, len(sample_nodes_df))
            osmid = list(sample_nodes_df.index.values)
            sample_nodes_df = sample_nodes_df.rename(index=dict(zip(osmid,C_ID)))
            sample_nodes_df.index.name = "C_ID"
            sample_nodes_df["osmid"] = osmid
            
            # Create set_of_all_customers final df
            set_of_all_customers = sample_nodes_df.copy()
            set_of_all_customers = set_of_all_customers.sort_values(by=["osmid"])
            set_of_all_customers.reset_index(drop = True, inplace = True)
            set_of_all_customers.index = C_ID
            set_of_all_customers.index.names = ["C_ID"]
            self.Nodes_S = set_of_all_customers["osmid"].tolist()
            self.J = set_of_all_customers.index.values
            self.set_of_all_customers = set_of_all_customers
            
            
        elif self.city == "Albstadt" and scenario == "mixed_mid":
            # Load Customers and PULs for specific scenario
            nodes_df = pd.read_hdf("nodes_df_alb", "nodes_df")
            pup_df = pd.read_hdf("alb_mid.h5", "alb_mid")
            pup_index = pup_df.index.values.tolist()
         
            samp_first = nodes_df.sample(n= 191, random_state=3)
            samp_first = samp_first.drop(index = 290847730)
            node_list = samp_first.index.values.tolist()
            
            # Drop necessary nodes so that we dont have duplicates for pups and customers
            intersection_set = set.intersection(set(node_list), set(pup_index))
            intersection_list = list(intersection_set)

            samp_first = samp_first.drop(index = intersection_list)              
            # Sample Customers and PuPs                
            sample_nodes_df = samp_first.sample(n = self.customers, random_state = seed_customers)
            pup_sample = pup_df.sample(n = self.pups, random_state = seed_pup)
            self.Nodes_P = pup_sample.index.values.tolist()
            # Demand for customers
            tri_demand_c = [mean_demand] * len(sample_nodes_df)
            sample_nodes_df["Demand_C"] = tri_demand_c
            
            # Define Demand Scenarios for PuPs
            tri_demand_p = [demand_p] * len(pup_sample)
   
            pup_sample["Demand_C"] = tri_demand_p
            
            self.pup_sample = pup_sample
            
            # Join the two together
            sample_nodes_df = pd.concat([sample_nodes_df, self.pup_sample])

            
            # Change index to Customer ID and create osmid column
            C_ID = createList_C(1, len(sample_nodes_df))
            osmid = list(sample_nodes_df.index.values)
            sample_nodes_df = sample_nodes_df.rename(index=dict(zip(osmid,C_ID)))
            sample_nodes_df.index.name = "C_ID"
            sample_nodes_df["osmid"] = osmid
            
            # Create set_of_all_customers final df
            set_of_all_customers = sample_nodes_df.copy()
            set_of_all_customers = set_of_all_customers.sort_values(by=["osmid"])
            set_of_all_customers.reset_index(drop = True, inplace = True)
            set_of_all_customers.index = C_ID
            set_of_all_customers.index.names = ["C_ID"]
            self.Nodes_S = set_of_all_customers["osmid"].tolist()
            self.J = set_of_all_customers.index.values
            self.set_of_all_customers = set_of_all_customers
            
        elif self.city == "Albstadt" and scenario == "mixed_high":
            # Load Customers and PULs for specific scenario
            nodes_df = pd.read_hdf("nodes_df_alb", "nodes_df")
            pup_df = pd.read_hdf("alb_high.h5", "alb_high")
            pup_index = pup_df.index.values.tolist()
         
            samp_first = nodes_df.sample(n= 191, random_state=3)
            samp_first = samp_first.drop(index = 290847730)
            node_list = samp_first.index.values.tolist()
            
            # Drop necessary nodes so that we dont have duplicates for pups and customers
            intersection_set = set.intersection(set(node_list), set(pup_index))
            intersection_list = list(intersection_set)

            samp_first = samp_first.drop(index = intersection_list)              
            # Sample Customers and PuPs                
            sample_nodes_df = samp_first.sample(n = self.customers, random_state = seed_customers)
            pup_sample = pup_df.sample(n = self.pups, random_state = seed_pup)
            self.Nodes_P = pup_sample.index.values.tolist()
            # Demand for customers
            tri_demand_c = [mean_demand] * len(sample_nodes_df)
            sample_nodes_df["Demand_C"] = tri_demand_c
            
            # Define Demand Scenarios for PuPs
            tri_demand_p = [demand_p] * len(pup_sample)
   
            pup_sample["Demand_C"] = tri_demand_p
            
            self.pup_sample = pup_sample
            
            # Join the two together
            sample_nodes_df = pd.concat([sample_nodes_df, self.pup_sample])

            
            # Change index to Customer ID and create osmid column
            C_ID = createList_C(1, len(sample_nodes_df))
            osmid = list(sample_nodes_df.index.values)
            sample_nodes_df = sample_nodes_df.rename(index=dict(zip(osmid,C_ID)))
            sample_nodes_df.index.name = "C_ID"
            sample_nodes_df["osmid"] = osmid
            
            # Create set_of_all_customers final df
            set_of_all_customers = sample_nodes_df.copy()
            set_of_all_customers = set_of_all_customers.sort_values(by=["osmid"])
            set_of_all_customers.reset_index(drop = True, inplace = True)
            set_of_all_customers.index = C_ID
            set_of_all_customers.index.names = ["C_ID"]
            self.Nodes_S = set_of_all_customers["osmid"].tolist()
            self.J = set_of_all_customers.index.values
            self.set_of_all_customers = set_of_all_customers
        else:
            None
        
        
        return set_of_all_customers

    
    
    def V_cfg(self,Vehicle_N, capacity_V = [], fixed_cost_V = []):

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
        self.K = set_of_all_vehicles.index.values

        self.set_of_all_vehicles = set_of_all_vehicles
        return set_of_all_vehicles
    
    
    def Dist_m(self, cost_per_meter):
        
        # Load the Distance Matrix for the choosen market from HERE API Output
        if self.city == "Albstadt":
            mat = pd.read_hdf("Desktop/LPR MATS/dist_matrix_alb_200.h5", "final_df")
        elif self.city == "Wuerzburg":
            mat = pd.read_hdf("Desktop/LPR MATS/dist_matrix_wue_200.h5", "final_df")
        else:
            None
           
        self.Nodes_S1 = self.DC_S + self.Nodes_S 
        dist_matrix = pd.DataFrame(index=self.Nodes_S1)

        # Rename the index for the join with the distance matrix
        dist_matrix = dist_matrix.rename_axis("osmid")

        # Join new empty dist_matrix with filled matrix for every chosen node
        dist_matrix = dist_matrix.merge(mat, left_index=True, right_index=True)
        dist_matrix = dist_matrix[dist_matrix.columns.intersection(self.Nodes_S1)]

        #Rename Columns to fit the order of Nodes_S1 = DCs  then Customers. Match Index and columns positionwise
        dist_matrix = dist_matrix.reindex(columns=self.Nodes_S1)

        # Replace column names with I and J variables = DC1, C1 etc.
        nn = np.concatenate((self.I,self.J), axis=None)
        new_colnames = nn.tolist()
        dist_matrix.columns = new_colnames
        dist_matrix.index = new_colnames
        self.dist_matrix_d = dist_matrix
        self.dist_matrix = dist_matrix * cost_per_meter
        return self.dist_matrix
    
    def Time_m(self, cost_per_second, service_time):
        
        # Load right Time matrix from file (Output from HERE API)
        if self.city == "Albstadt":
            mat1 = pd.read_hdf("Desktop/LPR MATS/time_matrix_alb_200.h5", "final_df")
        elif self.city == "Wuerzburg":
            mat1 = pd.read_hdf("Desktop/LPR MATS/time_matrix_wue_200.h5", "final_df")
        else:
            None
        
        
        # Join Distances for the sampled nodes from our Initial Distance Matrix for all nodes 
        time_matrix = pd.DataFrame(index=self.Nodes_S1)

        # Rename the index for the join with the distance matrix
        time_matrix = time_matrix.rename_axis("osmid")

        # Join new empty time_matrix with filled matrix for every chosen node
        time_matrix = time_matrix.merge(mat1, left_index=True, right_index=True)
        time_matrix = time_matrix[time_matrix.columns.intersection(self.Nodes_S1)]

        #Rename Columns to fit the order of Nodes_S1 = DCs  then Customers. Match Index and columns positionwise
        #time_matrix.index = self.Nodes_S1
        time_matrix = time_matrix.reindex(columns=self.Nodes_S1)

        # Replace column names with I and J variables = DC1, C1 etc.
        nn = np.concatenate((self.I, self.J), axis=None)
        new_colnames = nn.tolist()
        time_matrix.columns = new_colnames
        time_matrix.index = new_colnames

        time_matrix[len(self.DC_S):len(self.DC_S)+len(self.Nodes_S)] = time_matrix[len(self.DC_S):len(self.DC_S)+len(self.Nodes_S)] + service_time
        self.time_matrix_t = time_matrix
        self.time_matrix = time_matrix * cost_per_second
        return self.time_matrix
    
    
    def solve(self,Gap, Time_Limit, Para, DC_Max, Drive_Time_Max):
           
            
        def printSolve(scenStr):
            sLen = len(scenStr)
            print("\n" + "*"*sLen + "\n" + scenStr + "\n" + "*"*sLen + "\n")
            
        printSolve("Model is being built, this may take a moment")    
       
    
    
    # Define Gurobi Model
        self.m = Model()

        # Define Decision Variables
        self.x = self.m.addVars([*self.I,*self.J],[*self.I,*self.J],self.K, name = "x", vtype=GRB.BINARY)
        self.y = self.m.addVars(self.I, name = "y", vtype = GRB.BINARY)
        self.z = self.m.addVars(self.I, self.J, name = "z", vtype = GRB.BINARY)

        U = [(l,k) for l in self.J for k in self.K]
        self.u = self.m.addVars(U, name = "u", vtype= GRB.CONTINUOUS)

        # Set Math. Notation
        fixedCost_depot = quicksum(self.y[i] * self.set_of_all_DC.loc[i].fixed_cost_DC for i in self.I)

        variableCosts_transp = quicksum([(self.dist_matrix.loc[i,j] + self.time_matrix.loc[i,j]) * 
                                         self.x[i,j,k] for i in [*self.I,*self.J]
                                         for j in [*self.I,*self.J] for k in self.K if i!=j])

        variableCosts_DC = quicksum(self.set_of_all_DC.loc[i].varCost_DC *
                                    self.z[i,j] * self.set_of_all_customers.loc[j].Demand_C
                                    for j in self.J for i in self.I)

        fixedCost_vehicle = quicksum(self.x[i,j,k] * self.set_of_all_vehicles.loc[k].fixed_cost_V
                                     for i in self.I for j in self.J
                                     for k in self.K if i!=j)

        #Define Objective Function
        obj = fixedCost_depot + variableCosts_transp + variableCosts_DC + fixedCost_vehicle

        self.m.setObjective(obj, GRB.MINIMIZE)

        # Set Constraints
        for j in self.J:
            self.m.addConstr(quicksum(self.x[i,j,k] for i in [*self.I,*self.J] for k in self.K if i!=j) == 1)

        for k in self.K:
            self.m.addConstr(quicksum(self.set_of_all_customers.loc[j].Demand_C * self.x[i,j,k]
                                 for i in [*self.I,*self.J] 
                                 for j in self.J if i!=j)
                        <= self.set_of_all_vehicles.loc[k].capacity_V)

        for l in self.J:
            for j in self.J:
                if l!=j:
                    for k in self.K:
                        self.m.addConstr(self.u[l,k] - self.u[j,k] + (len(self.set_of_all_customers)
                                                       * self.x[l,j,k]) <= len(self.set_of_all_customers) -1)

        for i in [*self.I,*self.J]:
            for k in self.K:
                self.m.addConstr(quicksum(self.x[i,j,k] for j in [*self.I,*self.J] if i!=j)
                            - quicksum(self.x[j,i,k] for j in [*self.I,*self.J] if i!=j) == 0)


        for k in self.K:
            self.m.addConstr(quicksum(self.x[i,j,k] for i in self.I for j in self.J) <= 1)

        for i in self.I:
            self.m.addConstr(quicksum(self.z[i,j] * self.set_of_all_customers.loc[j].Demand_C
                                 for j in self.J) - (self.set_of_all_DC.loc[i].capacity_DC * self.y[i]) <= 0)


        for i in self.I:
            for j in self.J:
                for k in self.K:
                    self.m.addConstr(quicksum(self.x[i,u,k] + self.x[u,j,k]
                                              for u in [*self.I,*self.J]) - self.z[i,j] <= 1)
                    
        # New Constraints, to ensure that y[i], the opening of depots works            
        for i in self.I:
            for k in self.K:
                self.m.addConstr(quicksum(self.x[i,j,k] for j in self.J) <= self.y[i])

        # At least one depot has to be opened        
        self.m.addConstr(quicksum(self.y[i] for i in self.I) >= 1)

        #new zij 
        for j in self.J:
            self.m.addConstr(quicksum(self.z[i,j] for i in self.I) == 1)

        for i in self.I:
            for j in self.J:
                self.m.addConstr(quicksum(self.x[i,j,k] for k in self.K) <= self.z[i,j])
                
        self.m.update()
        
        # Driving Time Constraint, makes sure every vehicle drives 7 hours max a day
        for k in self.K:
            self.m.addConstr(quicksum(self.x[i,j,k] * self.time_matrix_t.loc[i,j] for i in [*self.I,*self.J] for j in [*self.I,*self.J]) <= Drive_Time_Max)
        
        
        # Add Condition that allows for setting DC Count, e.g. only open 1
        if DC_Max == 0:
            self.m.update()
        elif DC_Max == 1:
            self.m.addConstr(quicksum(self.y[i] for i in self.I) == 1)
        elif DC_Max == 2:
            self.m.addConstr(quicksum(self.y[i] for i in self.I) == 2)
        elif DC_Max == 3:
            self.m.addConstr(quicksum(self.y[i] for i in self.I) == 3)                          
        elif DC_Max == 4:
            self.m.addConstr(quicksum(self.y[i] for i in self.I) == 4)
        else:
            self.m.update()

        # Set Gap
        self.m.Params.MIPGap = Gap
        self.m.update()
        self.m.Params.LogFile = "gurobi_logee.log"
        self.m.setParam("TimeLimit", Time_Limit)
        # Set Model Parameters here if Para is set to 1, else just update the model
        if Para == 0:
            self.m.update()
        elif Para == 1:
            self.m.setParam("Method", 2)
            self.m.setParam("Presolve", 2)
            self.m.update()
        else:
            self.m.update()
        self.m.optimize()
        
    def DC_Print(self):
        print("The following DCs are established:")
        for i in self.I:
            if self.y[i].X >= 0.1:
                print("-{}".format(self.set_of_all_DC.loc[i].name))
        
    def Route_DC(self, DC):
        vals = self.m.getAttr('X', self.x)
        selected = tuplelist((i, j, k) for i, j, k in vals.keys() if vals[i, j, k] > 0.5)
        for i, tup in enumerate(selected.select(DC, "*", "*")):
            k = tup[2]
            print("Route for Vehicle {}: {}".format(k, DC), end = "")
            next_stop = tup[1]
            while next_stop:
                if next_stop == DC:
                    break
                else:
                    print(" -> {}".format(next_stop), end="")
                    next_stop = selected.select(next_stop, "*")[0][1]
            print(" -> {}".format(DC))
            
            
    def Route_Stats(self):
        for k in self.K:
            r_c = sum([self.x[i,j,k].X for i in [*self.I,*self.J] for j in [*self.I,*self.J]])
            time_v = sum([self.x[i,j,k].X * self.time_matrix_t.loc[i,j] for i in [*self.I, *self.J] for j in [*self.I, *self.J]])/60
            r_len = sum([self.x[i,j,k].X * self.dist_matrix_d.loc[i,j] for i in [*self.I,*self.J] for j in [*self.I,*self.J]]).round(1)
            load = sum([self.x[i,j,k].X * self.set_of_all_customers["Demand_C"].loc[j] for i in self.J for j in self.J])
            print(" Vehicle {} has {} stops on its route. Route Distance {}m. Route Time {}min. Route Load {} items".format(k, r_c, r_len, time_v.round(1), load))
            
            
 # Cost Printer
    def Cost_Printer(self):
        
        fixedDC = sum([self.y[i].X * self.set_of_all_DC.loc[i].fixed_cost_DC for i in self.I])
        dist_d = sum([self.x[i,j,k].X *  self.dist_matrix_d.loc[i,j] for i in [*self.I,*self.J] for j in [*self.I,*self.J] for k in self.K if i!=j]).round(2)
    
        time_t = sum([self.x[i,j,k].X * self.time_matrix_t.loc[i,j] for i in [*self.I,*self.J] for j in [*self.I,*self.J] for k in self.K if i!=j]).round(2)
        
        time_cost = sum([self.x[i,j,k].X * self.time_matrix.loc[i,j] for i in [*self.I,*self.J] for j in [*self.I,*self.J] for k in self.K if i!=j]).round(2)
        
        var_dist = sum([self.x[i,j,k].X *  self.dist_matrix.loc[i,j] for i in [*self.I,*self.J] for j in [*self.I,*self.J] for k in self.K if i!=j]).round(2)
        
        fixed_vehicle = sum([self.x[i,j,k].X * self.set_of_all_vehicles.loc[k].fixed_cost_V for i in self.I for j in self.J for k in self.K])
        varDC = sum(self.set_of_all_DC.loc[i].varCost_DC * self.z[i,j].X * self.set_of_all_customers.loc[j].Demand_C for j in self.J for i in self.I).round(2)
        TotalCost = (time_cost + var_dist + fixed_vehicle + varDC + fixedDC).round(2)
        Cost_per_order = (TotalCost/len(self.J)).round(2)
        Cost_per_item = (TotalCost/sum(self.set_of_all_customers["Demand_C"])).round(2)
        var_per_order = (varDC / len(self.J)).round(2)
        trans_per_order = ((var_dist+time_cost) / len(self.J)).round(2)
        trans_per_order_with_vehicle = ((var_dist+time_cost+fixed_vehicle) / len(self.J)).round(2)
        fixedDC_per_order = (fixedDC / len(self.J)).round(2)
        Items_N = sum(self.set_of_all_customers["Demand_C"])
        Stops = sum([self.x[i,j,k].X for i in [*self.I,*self.J] for j in [*self.I,*self.J] for k in self.K])
        Stops_min = (time_t/60)/Stops
        Stops_per_hour = (60/Stops_min).round(1)

                         
        cost_data = {"Customers_N": [len(self.J)],
             "Distance": [dist_d],
             "Time" : [time_t],
             "Stops" : [Stops],
             "Stops_h": [Stops_per_hour],
             "FixedDC_Cost" : [fixedDC],
             "Trans_Cost" : [var_dist+time_cost],
             "Variable_DC" : [varDC],
             "Fixed_V" : [fixed_vehicle],
             "Total_Cost" : [TotalCost],
             "Var_per_order": [var_per_order],
             "Trans_per_order" : [trans_per_order],
             "Transp_order_with_V": [trans_per_order_with_vehicle],
             "FixedDC_per_order" : [fixedDC_per_order],
             "Total_Cost_per_order" : [Cost_per_order],
             "Items" : [Items_N],
             "Cost_per_item" : [Cost_per_item]}
        cost_df = pd.DataFrame(cost_data)
        self.cost_df = cost_df
                         
                         
        print("Distance traveled: {}m - in km {} \nTransportation Cost {}€".format(dist_d, (dist_d/1000).round(1), (var_dist+time_cost)))
        print("Time elapsed on the routes {}s - in Minutes: {} - Stops per hour {}".format(time_t, (time_t/60).round(1), Stops_per_hour))
        print("Delivery Driver labor cost {}€".format(time_cost))
        print("Fuel Cost {}€".format(var_dist))
        print("Fixed Costs for Warehouse/s: {}€ \nVariable Warehousing Cost {}€ \nFixed Vehicle Cost {}€".format(fixedDC, varDC, fixed_vehicle))
        print("Total Cost: {}€".format(TotalCost))
        print("Variable Warehousing Cost per Order {}€".format(var_per_order))
        print("Transportation Cost per Order {}€ - with fixed vehicle cost included {}€".format(trans_per_order, trans_per_order_with_vehicle))
        print("Fixed Warehousing Cost per Order {}€".format(fixedDC_per_order))
        print("Total Cost per Item {}€".format(Cost_per_item))
        print("Cost per Order {}€".format(Cost_per_order))
                         
            
            
    # Function to save the final dataframe        
    def save_data(self, name_hdf):
        self.cost_df.to_hdf(name_hdf, mode = "w", key = "cost_df")



    # First Plot every possible DC and Customers
    def Basic_Plot(self, city, scenario):
        
        if scenario ==  "store":
            self.place = {"city": city, "country": "Germany"}
            G = ox.graph_from_place(self.place, network_type = "drive")
            Gs = ox.utils_graph.get_largest_component(G, strongly = True)
            self.Gs = Gs
            nodes, edges = ox.graph_to_gdfs(self.Gs, nodes=True, edges=True)

            ns = []
            for node in self.Gs.nodes():
                if node in self.DC_S:
                    ns.append(150)
                elif node in self.Nodes_S:
                    ns.append(50)
                else:
                    ns.append(0)

            nc = []
            for node in self.Gs.nodes():
                if node in self.DC_S: 
                    nc.append("red")
                elif node in self.Nodes_S:
                    nc.append("yellow")
                else:
                    nc.append("white")


            fig, ax = ox.plot_graph(self.Gs, node_size = ns, edge_linewidth = 0.5,
                                    node_color = nc, figsize = (22,22), bgcolor = "black")
        elif scenario == "DC" and city =="Wuerzburg":
            self.place = {"city": city, "country": "Germany"}
            G = ox.graph_from_place(self.place, buffer_dist= 10000, network_type = "drive")
            Gs = ox.utils_graph.get_largest_component(G, strongly = True)
            self.Gs = Gs
            nodes, edges = ox.graph_to_gdfs(self.Gs, nodes=True, edges=True)

            ns = []
            for node in self.Gs.nodes():
                if node in self.DC_S:
                    ns.append(150)
                elif node in self.Nodes_C:
                    ns.append(50)
                elif node in self.Nodes_P:
                    ns.append(100)
                    
                else:
                    ns.append(0)

            nc = []
            for node in self.Gs.nodes():
                if node in self.DC_S: 
                    nc.append("red")
                elif node in self.Nodes_C:
                    nc.append("yellow")
                elif node in self.Nodes_P:
                    nc.append("green")
                else:
                    nc.append("white")



            fig, ax = ox.plot_graph(self.Gs, node_size = ns, edge_linewidth = 0.5,
                                    node_color = nc, figsize = (22,22), bgcolor = "black")
        elif scenario =="DC" and city == "Albstadt":
            G = ox.graph_from_bbox(north = 48.33122, south =48.203926, east = 9.142871, west = 8.834133, network_type = "drive")
            Gs = ox.utils_graph.get_largest_component(G, strongly = True)
            self.Gs = Gs

            nodes, edges = ox.graph_to_gdfs(self.Gs, nodes=True, edges=True)


            ns = []
            for node in self.Gs.nodes():
                if node in self.DC_S:
                    ns.append(150)
                elif node in self.Nodes_S:
                    ns.append(50)
                else:
                    ns.append(0)

            nc = []
            for node in self.Gs.nodes():
                if node in self.DC_S: 
                    nc.append("red")
                elif node in self.Nodes_S:
                    nc.append("yellow")
                else:
                    nc.append("white")



            fig, ax = ox.plot_graph(self.Gs, node_size = ns, edge_linewidth = 0.5,
                                    node_color = nc, figsize = (22,22), bgcolor = "black")
        else:
            None
        
        
    def Route_Plot(self, Vehicle):

        nodes, edges = ox.graph_to_gdfs(self.Gs, nodes=True, edges=True)
        
        # Get Tuples of routes from model solution
        vals = self.m.getAttr('X', self.x)
        selected = tuplelist((i, j, k) for i, j, k in vals.keys() if vals[i, j, k] > 0.5)
        
        # Build DF which has all the osmids for DCs and Customers
        # DCs
        dc_set = self.set_of_all_DC.reset_index(level = "DC_ID")
        dc_set.drop(["fixed_cost_DC", "capacity_DC", "varCost_DC", "lat", "lon"], axis=1, inplace = True)
        dc_set.rename(columns={"DC_ID": "ID"}, inplace = True) 
        # Customers
        c_set = self.set_of_all_customers.reset_index(level = "C_ID")
        c_set.drop(["Demand_C", "lat", "lon"], axis=1, inplace = True)
        c_set.rename(columns={"C_ID": "ID"}, inplace = True)
        #join
        set_join = pd.concat([dc_set, c_set], axis=0)
        
        #build route DF, which has Columns for Orig/Destination/Vehicle
        routes_sel = list(selected)
        df_r = pd.DataFrame(routes_sel)
        df_r.rename(columns={0: "Orig", 1: "Dest", 2: "V"}, inplace = True)
        # Join DFs (customer/DCs and Routes)
        joiner = pd.merge(df_r, set_join, left_on = "Orig", right_on = "ID")
        joiner.drop(["ID"], axis = 1, inplace = True)
        joiner.rename(columns={"osmid": "orig_osmid"}, inplace = True)
        joiner = pd.merge(joiner, set_join, left_on = "Dest", right_on = "ID")
        joiner.drop(["ID"], axis = 1, inplace = True)
        joiner.rename(columns={"osmid": "dest_osmid"}, inplace = True)
        # Function that gives the route from Orig to Dest as a list of nodes which have to be traversed through
        def shortest_route(orig, dest):
            route = nx.shortest_path(self.Gs, source=orig, target=dest, weight= "length")                                 
            return route
        # Apply function to add a new column named route which contains lists of routes(as nodes)
        joiner["route"] = joiner.apply(lambda x: shortest_route(x["orig_osmid"], x["dest_osmid"]), axis=1)
        # One big list of lists for all of the routes
        route_list = joiner["route"].tolist()
        # Could plot big list of routes but that would be all vehicles same color etc., so probably best to split according to vehicle
        # Have to split the routes into vehicles
        vehicle_r = joiner[joiner["V"]==Vehicle]
        vehicle_route_list = vehicle_r["route"].tolist()
        
        # Create a list which has all of the nodes for that specific route for plotting purposes
        route_nodes_o= vehicle_r["orig_osmid"].tolist()
        route_nodes_d= vehicle_r["dest_osmid"].tolist()
        route_nodes = route_nodes_o + route_nodes_d
        
        # Plotting 
        ns = []
        for node in self.Gs.nodes():
            if node == route_nodes[0]:
                ns.append(500)
            else:
                ns.append(0)
                
        fig, ax = ox.plot_graph_routes(self.Gs, vehicle_route_list,
                                       route_colors = "orange", route_linewidths = 3, node_size = ns,
                                    figsize = (22,22))