import pandas as pd
import numpy as np
from sklearn.svm import SVC

#Load up the /Module6/Datasets/parkinsons.data data set into a variable X, being sure to drop the name column.

X = pd.read_csv('Datasets/parkinsons.data')
X = pd.DataFrame(X)
X = X.drop('name', axis=1)


#Splice out the status column into a variable y and delete it from X.

y = X.status
X = X.drop('status', axis=1)
#print(X)


from sklearn import preprocessing

#X = preprocessing.StandardScaler().fit_transform(X)
#X = preprocessing.MinMaxScaler().fit_transform(X)
X = preprocessing.Normalizer().fit_transform(X)
#X = preprocessing.MaxAbsScaler().fit_transform(X)
#X = preprocessing.KernelCenterer().fit_transform(X)
X = preprocessing.scale(X)
#T = X # No Change



#Perform a train/test split. 30% test group size, with a random_state equal to 7.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


#Create a SVC classifier. Don't specify any parameters, just leave everything as default. Fit it against your training data and then score your testing data.

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
score = svc.score(X_test, y_test)


#What accuracy did you score?

print(score)







rC = np.arange(0.05, 2, 0.05)
rgamma = np.arange(0.001, 0.1, 0.001)
print(rgamma)
best_score = 0

for C in rC:
    for gamma in rgamma:             
        svc = SVC(C=C, gamma=gamma)
        svc.fit(X_train, y_train) 
        score = svc.score(X_test, y_test)
        if best_score < score:
            best_score = score
print(best_score)