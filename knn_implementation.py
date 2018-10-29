"""
K-nearest neighbour
"""
import numpy as np
from sklearn import cross_validation
"""
Distance calculation
"""
def distancecalc(row1):
    distance=np.zeros(len(X_train))
    row1v=np.multiply(row1,np.ones([len(X_train),len(X_train[0])]))
    disc=np.zeros(len(distance))
    distance=abs(np.square(row1v)-np.square(X_train))
    disc=[]
    for disrow in distance:
        dis=np.sum(disrow)
        disc.append(dis)
    disc=np.sqrt(disc)    
    disc=np.column_stack((disc,Y_train))
    dicdisc=dict(disc)
    sdisc=sorted(dicdisc)
    findist=[]
    for i in range(9):
        f=(dicdisc[sdisc[i]])
        findist.append(f)
    labelfin=np.array(findist)    
    labelfin=labelfin.astype(int)
    result=(np.bincount(labelfin).argmax())
    return result


dataset1=np.genfromtxt('D:\Study Files\My Projects\SMAI\will_ferell.csv',delimiter=',')
dataset2=np.genfromtxt('D:\Study Files\My Projects\SMAI\chad_smith.csv',delimiter=',')
Y_label1=np.ones((len(dataset1),1),dtype=int)
dataset1=np.column_stack((dataset1,Y_label1))
Y_label2=np.zeros((len(dataset2),1),dtype=int)
dataset2=np.column_stack((dataset2,Y_label2))
dataset=np.concatenate((dataset1,dataset2),axis=0)
np.random.shuffle(dataset)

for row in dataset:
    for item in row:
        if item=='?':
            item='-99999'
            
          

feat=np.array(dataset)
lab=np.array(dataset[:,[128]],dtype='int')
feat=np.delete(feat,128,1)
X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(feat,lab,test_size=0.2)
predictions=[]
for row in X_test:
    distance=distancecalc(row)
    predictions.append(distance)
predictions=np.array(predictions)
count=0
for i in range(len(predictions)):
    if predictions[i]!=Y_test[i]:
        count+=1
accuracy=((len(Y_test)-count)/len(Y_test))*100        