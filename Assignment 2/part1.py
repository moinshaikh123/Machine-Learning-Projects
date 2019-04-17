import numpy as np
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing




train=pd.read_excel("dataset for part 1.xlsx",sheet_name="Training Data")
test=pd.read_excel("dataset for part 1.xlsx",sheet_name="Test Data")

test_top=test

def calc_IG(data,column_name):
        
        values = data[column_name].value_counts().keys().tolist()
        counts = data[column_name].value_counts().tolist()
        
        n=sum(counts)
        
        
        entropy=0
        for i in counts:
            entropy=entropy-(i/n)*math.log(i/n)/math.log(2)
        
        return entropy

    
def calc_GI(data,column_name):
        values = data[column_name].value_counts().keys().tolist()
        counts = data[column_name].value_counts().tolist()
        
        n=sum(counts)
        
        
        entropy=0
        for i in counts:
            entropy=entropy+(i/n)*(i/n)
        
        return 1-entropy
    
        


def find_best_attribute_IG(data,label):
    ig_max=0
    best_col = ""
   
    for col in data.columns:
        if (col==label):
            continue
        
        row_values=data[col].unique()
    
        parent_entropy = calc_IG(data,label)
        
        total_rows=data.shape[0]
    
        temp=0
        for i in row_values:
            sub_data=data.loc[data[col] == i]
        
            temp=temp+((sub_data.shape[0])/(total_rows))*(calc_IG(sub_data,label))
    
        temp=parent_entropy-temp
        if(temp>ig_max):
            ig_max=temp
            best_col=col
        
        
        
    return best_col


def find_best_attribute_GI(data,label):
    ig_min=99999
    best_col = ""
    
    for col in data.columns:
        if (col==label):
            continue
        
        row_values=data[col].unique()
        
        total_rows=data.shape[0]
    
        temp=0
        for i in row_values:
            sub_data=data.loc[data[col] == i]
        
            temp=temp+((sub_data.shape[0])/(total_rows))*(calc_GI(sub_data,label))
    
        if(temp<ig_min):
            ig_min=temp
            best_col=col
        

        
    return best_col


        
def create_tree_Inf_Gain(tree,data,level,max_level,label):
    
    if(level>=max_level):
        temp_list=tree[level]
        
        lst=list(data[label])
        temp_list.append({max(lst,key=lst.count):None})
        
        tree[level]=temp_list
        return tree
    if calc_IG(data,label)==0:
        temp=data[label].unique()
        for i in temp:
            if level<max_level :
                temp_list=tree[level]
    
    
                temp_list.append({i:None})
    
                tree[level]=temp_list
                break
        return tree
    
    
    best_attr=find_best_attribute_IG(data,label)
    
    unique_vals = data[best_attr].value_counts().keys().tolist()
    temp_list=tree[level]
    
    
    temp_list.append({best_attr:unique_vals})
    
    tree[level]=temp_list
    
    for i in unique_vals:
        
            sub_data=data.loc[data[best_attr] == i]
            create_tree_Inf_Gain( tree, sub_data, level+1,max_level,label)
            
    return tree



        
def create_tree_Gini(tree,data,level,max_level,label):
    
    
    if(level>max_level):
        temp_list=tree[level]
        
        lst=list(data[label])
        temp_list.append({max(lst,key=lst.count):None})
        
        tree[level]=temp_list
        return tree
    
    if calc_GI(data,label)==0:
        temp=data[label].unique()
        for i in temp:
            if level<max_level :
                temp_list=tree[level]
    
              
                temp_list.append({i:None})
    
                tree[level]=temp_list
                break
        return tree
    
    
    best_attr=find_best_attribute_GI(data,label)
    
    unique_vals = data[best_attr].value_counts().keys().tolist()
    temp_list=tree[level]
    
    
    temp_list.append({best_attr:unique_vals})
    
    tree[level]=temp_list
    
    for i in unique_vals:
        
            sub_data=data.loc[data[best_attr] == i]
            create_tree_Gini( tree, sub_data, level+1,max_level,label)
            
    return tree




def predict(data,row_name,tree,num,level):
    
    
    best_attr=list(tree[level][num].keys())
    
    attr_values=tree[level][num][best_attr[0]]
    
    
    if not attr_values:
        return best_attr[0]
    
    count=0
    for k in range(0,num):
        count=count+len(tree[level][k].keys())
    
    r=0
    
    for i in attr_values:
        if data.loc[row_name,best_attr[0]]==i:
            break
        else:
            r=r+1
    
    return predict(data,row_name,tree,r+count,level+1)
    

def print_complete_tree(tree,num,level,max_level):
    
    
    if(level>max_level):
            return
        
    if num>=len(tree[level]):
        return
    
    
    
    best_attr=(tree[level][num])
    counter=0
    
    
    for i in range(num):
        counter=counter+len(list(tree[level][i].keys()))
        
    
    for k in best_attr.keys():
        val=best_attr[k]
        
        if not val:
            for i in range(level):
                print("   ",end="")
            print("|",end="")
            print(k)
            return
        
        for q in val:
            for i in range(level):
                print("   ",end="")
            print("|",end="")
            print(k,end=" ")
            print("::",end="")
            print(q)
            print_complete_tree(tree,counter,level+1,max_level)
            counter=counter+1
    

    

print("Information Gain of root node " + str(calc_IG(train,"profitable")))

print("Gini Index of root node " + str(calc_GI(train,"profitable")))



print("Creating tree by Information Gain method")
tree={0:[],1:[],2:[],3:[]}

print("The tree is as follows\n\n Tree::")

tree=create_tree_Inf_Gain(tree,train,0,4,"profitable")
print(tree)

print("Predicting data using information Gain")
predicted=[]
for row in test_top.index: 
    predicted.append(predict(test,row,tree,0,0))
print(predicted)

y=0
accuracy=0
for i in test["profitable"]:
    if i == predicted[y]:
        accuracy=accuracy+1
    y=y+1

accuracy=accuracy/test.shape[0]

print("The accuracy of my decision tree is ::" + str(accuracy))

print("Creating tree by Gini index method")
tree={0:[],1:[],2:[],3:[]}

print("The tree is as follows\n\n Tree::")
tree=create_tree_Gini(tree,train,0,4,"profitable")

print(tree)

print("Predicting data using Gini index")

predicted=[]
for row in test_top.index: 
    predicted.append(predict(test,row,tree,0,0))
        
print(predicted)

y=0
accuracy=0
for i in test["profitable"]:
    if i == predicted[y]:
        accuracy=accuracy+1
    y=y+1
    
accuracy=accuracy/test.shape[0]

print("The accuracy of my decision tree is ::" + str(accuracy))

    
    
    
    
#     for node in best_attr.keys():
#         for k in node.keys():
            
#             values=node[k]
            
#             for u in values:
#                 for i in range(level):
#                     print(" ",end="")
#                 print(k+' :: ' + u)
#                 count=count+1
                
#                 print_complete_tree(tree,r+count,level+1,max_level)
            
#             print_complete_tree(tree,)
        
        
#     print(best_attr )
#     print("asfasf")
        
#     attr_values=tree[level][num][best_attr[0]]
    
#     if not attr_values:
#         return 
    
#     for i in range(level):
#         print(" ",end="")
    
#     print(best_attr[0],end="")
#     print("::",end=" ")
#     print(attr_values[0])
    
#     count=0
#     for k in range(0,num):
#         count=count+len(tree[level][k].keys())
    
#     r=0
    
#     for i in attr_values:
#         print_complete_tree(tree,r+count,level+1,max_level)
#         r=r+1
    
    
#     print_complete_tree(tree,r+count,level+1,max_level)
    
 
print_complete_tree(tree,0,0,3)



X_train = train.drop('profitable', axis=1)  
y_train = train['profitable']

X_test = test.drop('profitable', axis=1)  
y_test = test['profitable']

#Preprocessing train data
for i in X_train.columns:
    le = preprocessing.LabelEncoder()
    le.fit(X_train[i])
    X_train[i]=le.transform(X_train[i])
    
le = preprocessing.LabelEncoder()
le.fit(train['profitable'])
y_train=le.transform(train['profitable'])


#Preprocessing test data
for i in X_test.columns:
    le = preprocessing.LabelEncoder()
    le.fit(X_test[i])
    X_test[i]=le.transform(X_test[i])
    
le = preprocessing.LabelEncoder()
le.fit(train['profitable'])



classifier = DecisionTreeClassifier(random_state = 0)  
classifier.fit(X_train, y_train)  


y_pred = le.inverse_transform(classifier.predict(X_test))


print(y_pred)

print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred))  





    
