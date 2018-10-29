"""
svm classifier-single dataset and labels
@author: RAGHU RAM
"""

import numpy as np
from sklearn import cross_validation,svm,preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support

def getresults(clf,X_test):
    resultsprob = clf.predict_proba(X_test)
    """saves the probabilities"""
    np.savetxt("output_probabilities.csv", resultsprob, delimiter=",")
    predictions=[]
    predictions=clf.predict(X_test)
    return resultsprob,predictions  

feat=np.genfromtxt('D:\Study Files\My Projects\SMAI\\faces95.csv',delimiter=',')
labels=np.genfromtxt('D:\Study Files\My Projects\SMAI\\faces95_labes.csv',delimiter=',',dtype=str)
labels=np.delete(labels,1,axis=1)
dataset=np.column_stack((feat,labels))
np.random.shuffle(dataset)
labels=np.array(dataset[:,[128]],dtype='str')
feat=np.delete(dataset,128,axis=1)

scaler=preprocessing.MinMaxScaler()
scaler.fit(feat)
feat=scaler.transform(feat)

X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(feat,labels.ravel(),test_size=0.2)
#X_train=feat
#Y_train=labels
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
     decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',probability=True)

scores = cross_val_score(clf,X_train,Y_train.ravel(), cv=5)
print("cross validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 2))
  
clf.fit(X_train,Y_train)
resultsprob = clf.predict_proba(X_test)
predictions=clf.predict(X_test)

count=0
for i in range(len(predictions)):
    if predictions[i]!=Y_test[i]:
        count+=1
accuracy=((len(Y_test)-count)/len(Y_test))*100 
print("Test accuracy:%0.2f"%accuracy)
[probs,predictions]=getresults(clf,X_test)
"""
to print probabilities
"""
#print('%.2f',probs)
count=0
for i in range(len(predictions)):
    if predictions[i]!=Y_test[i]:
        count+=1
accuracy=((len(Y_test)-count)/len(Y_test))*100 
[precision,recall,f1score,support]=precision_recall_fscore_support(Y_test, predictions)
precision1=precision.mean()
recall1=recall.mean()
f1score1=f1score.mean()
print("Test accuracy:%0.2f"%accuracy)
print("precision:%0.2f"%precision1)
print("recall:%0.2f"%recall1)
print("f1score:%0.2f"%f1score1)
