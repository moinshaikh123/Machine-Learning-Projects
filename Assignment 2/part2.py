##########################################################################################
# Part 2 

import numpy as np
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing





train_data = pd.read_csv('./dataset for part 2/traindata.txt', sep="\t", header=None)
train_data.columns = ["docId", "wordId"]

word_data=pd.read_csv('./dataset for part 2/words.txt',header=None)
word_data.columns=["word"]

test_data = pd.read_csv('./dataset for part 2/testdata.txt', sep="\t", header=None)
test_data.columns = ["docId", "wordId"]

test_label=pd.read_csv('./dataset for part 2/testlabel.txt',header=None)
test_label.columns=["class"]


words = []
no_words = 0
with open("./dataset for part 2/words.txt") as f:
    for i in f:
        i = i.replace('\n', '')
        words.append(str(i))
        no_words += 1
f.close()
print("no_words : "+str(no_words))



train_label=pd.read_csv('./dataset for part 2/trainlabel.txt',header=None)
train_label.columns=["class"]


# Making a dataframe for train

no_of_rows=train_label.shape[0]
no_of_cols=word_data.shape[0]

print(no_of_cols)

counter=0


final_data=np.zeros([no_of_rows,no_of_cols])

print(final_data)

for i in train_data.index:
    final_data[train_data.loc[i,"docId"]-1,train_data.loc[i,"wordId"]-1]=1

#final_data = pd.DataFrame(0, index=range(0,no_of_rows), columns=word_data["word"].unique())

final_data=pd.DataFrame(data=final_data,index=list(range(final_data.shape[0])),columns=word_data["word"])

final_data["label"]=train_label["class"]


#making a dataframe for test

no_of_rows=test_label.shape[0]
no_of_cols=word_data.shape[0]

print(no_of_cols)

counter=0


test_final_data=np.zeros([no_of_rows,no_of_cols])

print(test_final_data)

for i in test_data.index:
    test_final_data[test_data.loc[i,"docId"]-1,test_data.loc[i,"wordId"]-1]=1


test_final_data=pd.DataFrame(data=test_final_data,index=list(range(test_final_data.shape[0])),columns=word_data["word"])

test_final_data["label"]=test_label["class"]
    


print("Creating tree by Information Gain method")
tree={0:[]}
for i in range(1,12):
    tree[i]=[]

print("The tree is as follows\n\n Tree::")

tree=create_tree_Inf_Gain(tree,final_data,0,10,"label")
#tree={0: [{'writes': [0.0, 1.0]}], 1: [{'god': [0.0, 1.0]}, {'graphics': [0.0, 1.0]}], 2: [{'that': [0.0, 1.0]}, {'use': [0.0, 1.0]}, {'image': [0.0, 1.0]}, {2: None}], 3: [{'bible': [0.0, 1.0]}, {'wrote': [0.0, 1.0]}, {1: None}, {'archive': [0.0, 1.0]}, {'that': [1.0, 0.0]}, {2: None}], 4: [{'atheist': [0.0, 1.0]}, {1: None}, {'people': [0.0, 1.0]}, {'ve': [0.0, 1.0]}, {2: None}, {1: None}, {'program': [0.0, 1.0]}, {'god': [0.0, 1.0]}], 5: [{2: None}, {1: None}, {2: None}, {1: None}, {1: None}, {2: None}, {1: None}, {2: None}, {2: None}, {1: None}]}




predicted=[]
for row in test_final_data.index: 
    predicted.append(predict(test_final_data,row,tree,0,0))
        
print(predicted)
print(test_label["class"])


y=0
accuracy=0
for i in test_label["class"]:
    if i == predicted[y]:
        accuracy=accuracy+1
    y=y+1
    
accuracy=accuracy/test_label.shape[0]

print("The accuracy of my decision tree is ::" + str(accuracy))





print_complete_tree(tree,0,0,11)


# importing the required module 
import matplotlib.pyplot as plt 
x=[8,11,16,21,25]
y=[0.7153628652214892,0.7313854853911405,0.6870876531573987,0.6870876531573987,0.6870876531573987]
# corresponding y axis values 

  
# plotting the points  
plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12) 
  

# naming the x axis 
plt.xlabel('Accuracy') 
# naming the y axis 
plt.ylabel('Depth of the tree') 
  
# giving a title to my graph 
plt.title('Accuracy vs Depth of the tree') 
  
# function to show the plot 
plt.show()  
# x axis values 



#Scikitlearn Part 2

train = np.zeros([train_label.shape[0], word_data.shape[0]+1])
sum_ = 0
with open("dataset for part 2/traindata.txt") as f:
    for i in f:
        i = i.split()
        train[int(i[0])-1][int(i[1])-1] = 1
        sum_ += 1
f.close()


xtrain = []
ytrain = []
xtest = []
ytest = []


test = np.zeros([train_label.shape[0], word_data.shape[0]+1])
sum_ = 0
with open("dataset for part 2/testdata.txt") as f:
    for i in f:
        i = i.split()
        test[int(i[0])-1][int(i[1])-1] = 1
        sum_ += 1
f.close()

print("\nscikit_learn implementation")
train_ = train.copy()
# train_ = data_encoding(train_)
test_ = test.copy()
# test_ = data_encoding(test_)


for i in train:
    xtrain.append(i[:-1].tolist())
    ytrain.append(i[-1])

for i in test:
    xtest.append(i[:-1].tolist())
    ytest.append(i[-1])    

tree_lib = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
tree_lib.fit(xtrain, ytrain)
ypred = tree_lib.predict(xtest)

confusion_matrix(ytest, ypred)
print ("using info_gain accuracy : ", accuracy_score(ytest,ypred)*100)





        

