#!/usr/bin/env python
# coding: utf-8

# In[143]:


import pandas as pd 
import numpy as np
input_data = pd.read_csv("AAAI.csv")

print(input_data.columns)


# In[144]:


arr=np.zeros(shape=[150,150])


# In[145]:


import math

I1 = 0
H1 = 0
H2 = 0
classes = {}
N = 150





def func(list1,list2):
    same_count=0
    total_count=0
    
    for i in list1:
        for j in list2:
            if(i==j):
                same_count=same_count+1
                
    total_count=len(list1)+len(list2)-same_count
    
    return same_count/total_count

sum=0
Topic_col=input_data["Topics"]
for i in input_data["Topics"].index:
    x_top=Topic_col[i].split('\n')
    for j in input_data["Topics"].index:
        if(i==j):
            arr[i][j]=-1
            continue
        y_top=Topic_col[j].split('\n')
        arr[i][j]=func(x_top,y_top)






max_val=0
def calc_linkage(matrix,cluster):    
    
    
    max_x=0
    max_y=0
    max_val=-9999
    for i in range(150):
#         print(i)
        for j in range(150):
            if(i==j or i not in cluster_dict.keys() or j not in cluster_dict.keys()):
                continue
            if(max_val < arr[i][j]):
                max_val=arr[i][j]
                max_x=i
                max_y=j

    return [max_val,max_x,max_y]

def process_cluster(cluster):
    r=0
    for q in range(len(cluster)-1):
        q=q-r
        if(cluster[q]==cluster[q+1]):
            r=r+1
            del cluster[q]
    return cluster



def update_matrix(cluster,arr,max_i,max_j):
    for i in range(0, 150):
        for c in cluster[max_i]:
            arr[i][c] = min(arr[i][max_i], arr[i][max_j])
            arr[c][i] = min(arr[max_i][i], arr[max_j][i])
        for c in cluster[max_j]:
            arr[i][c] = min(arr[i][max_i], arr[i][max_j])
            arr[c][i] = min(arr[max_i][i], arr[max_j][i])
    
    return arr
    
    

cluster_dict = {}


for i in range(0,150):
    cluster_dict[i]=[i]
                        

#print(cluster_dict)

def add_to_cluster(cluster_dict,max_x,max_y,arr):

        cluster_dict[max_y].append(max_x)
        arr=update_matrix(cluster_dict,arr,max_y,max_x)
        for k in cluster_dict[max_x]:
            cluster_dict[max_y].append(k)
        
        del cluster_dict[max_x]

    
    

while(len(list(cluster_dict.keys()))>=10):
    [max_val,max_x,max_y]=calc_linkage(arr,cluster_dict)
#     print(max_x,max_y)
    add_to_cluster(cluster_dict,max_x,max_y,arr)

        
for k in cluster_dict.keys():
    if not cluster_dict[k]:
        continue
    else:
        cluster_dict[k]=process_cluster(cluster_dict[k])
        print(cluster_dict[k])

    
    
def cluster_process(cluster_nodes,classes):
    for j in cluster_nodes:
        fg = list(classes.keys()).index(input_data.iloc[j][3]) if input_data.iloc[j][3] in classes else -1
        if fg==-1:
            classes[input_data.iloc[j][3]] = 1
        else:
            classes[input_data.iloc[j][3]] = classes[input_data.iloc[j][3]] + 1
    return [cluster_nodes,classes]




for i in range(0, 150):
    curr=input_data.iloc[i][3]
    if  curr not in classes:
        classes[curr] = 1
    else:
        classes[curr] =classes[curr] + 1

for k in cluster_dict.keys():
    classes1 = {}
    curr = cluster_dict[k]
    H1 = H1 - len(curr)*math.log(len(curr)*1.0/N)
    cluster_nodes = list(curr)
    
    [cluster_nodes,classes1]=cluster_process(cluster_nodes,classes1)
    
    for j in classes1:
        temp=1.0*N*classes1[j]/(len(curr)*1.0*classes[j])
        I1 += classes1[j]*1.0*math.log(temp)/N
            

for i in classes:
    curr=classes[i]
    H2 -= curr*math.log(curr*1.0/N)

H1 = 1.0*H1/N
H2 = 1.0*H2/N
NMI = (2.0*I1)/(H1+H2)
print("Calculated NMI : ", NMI)



            

            
    


        

                
        
        
        


                
            
            





         


# In[146]:


import math

I1 = 0
H1 = 0
H2 = 0
classes = {}
N = 150



def func(list1,list2):
    same_count=0
    total_count=0
    
    for i in list1:
        for j in list2:
            if(i==j):
                same_count=same_count+1
                
    total_count=len(list1)+len(list2)-same_count
    
    return same_count/total_count


Topic_col=input_data["Topics"]
for i in input_data["Topics"].index:
    x_top=Topic_col[i].split('\n')
    for j in input_data["Topics"].index:
        if(i==j):
            arr[i][j]=-1
            continue
        y_top=Topic_col[j].split('\n')
        arr[i][j]=func(x_top,y_top)




max_val=0
def calc_linkage(matrix,cluster):    
    
    
    max_x=0
    max_y=0
    max_val=-9999
    for i in range(150):

        for j in range(150):
            if(i==j or i not in cluster_dict.keys() or j not in cluster_dict.keys()):
                continue
            if(max_val < arr[i][j]):
                max_val=arr[i][j]
                max_x=i
                max_y=j

    return [max_val,max_x,max_y]




def update_matrix(cluster,arr,max_i,max_j):
    for i in range(0, 150):
        for c in cluster[max_i]:
            arr[i][c] = max(arr[i][max_i], arr[i][max_j])
            arr[c][i] = max(arr[max_i][i], arr[max_j][i])
        for c in cluster[max_j]:
            arr[i][c] = max(arr[i][max_i], arr[i][max_j])
            arr[c][i] = max(arr[max_i][i], arr[max_j][i])
    
    return arr
    
    

cluster_dict = {}


for i in range(0,150):
    cluster_dict[i]=[i]
                        

        
def add_to_cluster(cluster_dict,max_x,max_y,arr):

        cluster_dict[max_y].append(max_x)
        arr=update_matrix(cluster_dict,arr,max_y,max_x)
        for k in cluster_dict[max_x]:
            cluster_dict[max_y].append(k)
        
        del cluster_dict[max_x]

        
def process_cluster(cluster):
    r=0
    for q in range(len(cluster)-1):
        q=q-r
        if(cluster[q]==cluster[q+1]):
            r=r+1
            del cluster[q]
    return cluster

    
    

while(len(list(cluster_dict.keys()))>=10):
    [max_val,max_x,max_y]=calc_linkage(arr,cluster_dict)
#     print(max_x,max_y)
    add_to_cluster(cluster_dict,max_x,max_y,arr)

for k in cluster_dict.keys():
    if not cluster_dict[k]:
        continue
    else:
        cluster_dict[k]=process_cluster(cluster_dict[k])
        print(cluster_dict[k])


    
    
def cluster_process(cluster_nodes,classes):
    for j in cluster_nodes:
        fg = list(classes.keys()).index(input_data.iloc[j][3]) if input_data.iloc[j][3] in classes else -1
        if fg==-1:
            classes[input_data.iloc[j][3]] = 1
        else:
            classes[input_data.iloc[j][3]] = classes[input_data.iloc[j][3]] + 1
    return [cluster_nodes,classes]




for i in range(0, 150):
    curr=input_data.iloc[i][3]
    if  curr not in classes:
        classes[curr] = 1
    else:
        classes[curr] =classes[curr] + 1


for k in cluster_dict.keys():
    classes1 = {}
    curr = cluster_dict[k]
    H1 = H1 - len(curr)*math.log(len(curr)*1.0/N)
    cluster_nodes = list(curr)
    
    [cluster_nodes,classes1]=cluster_process(cluster_nodes,classes1)
    
    for j in classes1:
#         print(j, classes1[j], len(i), classes[j])
        temp=1.0*N*classes1[j]/(len(curr)*1.0*classes[j])
        I1 += classes1[j]*1.0*math.log(temp)/N
            

for i in classes:
    curr=classes[i]
    H2 -= curr*math.log(curr*1.0/N)

H1 = 1.0*H1/N
H2 = 1.0*H2/N
NMI = (2.0*I1)/(H1+H2)
print("Calculated NMI : ", NMI)



            

            
    


        

                
        
        
        


                
            
            





         


# In[152]:


import networkx as nx
import math

I1 = 0
H1 = 0
H2 = 0
classes = {}
N = 150
threshold = 0.16


def make_graph(G,input_data):
    
    for i in range(0, 150):
        for j in range(0, i):
            if func(input_data.iloc[i][2].split('\n'), input_data.iloc[j][2].split('\n'))>=threshold:
                G.add_edge(i, j)
                
    return G




G = nx.Graph()
G.add_nodes_from(range(0, 150))


G=make_graph(G,input_data)

cc = nx.connected_components(G)
length = len(list(cc))

while length!=9:
    edge_betw_cent = nx.edge_betweenness_centrality(G)
    max_bc = -9999
    for i in edge_betw_cent:
        if edge_betw_cent[i]>max_bc:
            max_list = i
            max_bc = edge_betw_cent[i]
            
    G.remove_edge(max_list[0], max_list[1])
    cc = nx.connected_components(G)
    length=len(list(cc))





cc = nx.connected_components(G)
for i in cc:
    print(list(i))





    
    
def cluster_process(cluster_nodes,classes):
    for j in cluster_nodes:
        fg = list(classes.keys()).index(input_data.iloc[j][3]) if input_data.iloc[j][3] in classes else -1
        if fg==-1:
            classes[input_data.iloc[j][3]] = 1
        else:
            classes[input_data.iloc[j][3]] = classes[input_data.iloc[j][3]] + 1
    return [cluster_nodes,classes]




for i in range(0, 150):
    curr=input_data.iloc[i][3]
    if  curr not in classes:
        classes[curr] = 1
    else:
        classes[curr] =classes[curr] + 1


cc = nx.connected_components(G)
for k in cc:
    classes1 = {}
    curr=list(k)
    H1 = H1 - len(curr)*math.log(len(curr)*1.0/N)
    cluster_nodes = curr
    
    [cluster_nodes,classes1]=cluster_process(cluster_nodes,classes1)
    
    for j in classes1:
        temp=1.0*N*classes1[j]/(len(curr)*1.0*classes[j])
        I1 += classes1[j]*1.0*math.log(temp)/N
            

for i in classes:
    curr=classes[i]
    H2 -= curr*math.log(curr*1.0/N)

H1 = 1.0*H1/N
H2 = 1.0*H2/N
NMI = (2.0*I1)/(H1+H2)
print("Calculated NMI : ", NMI)



# In[ ]:





# In[ ]:





# In[ ]:




