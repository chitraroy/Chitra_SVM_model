# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:04:30 2022

@author: chitr
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import warnings
warnings.filterwarnings("ignore")
MINST_chitra = pd.read_csv('C:/input/mnist_784.csv')
MINST_chitra.head()
MINST_chitra.shape
MINST_chitra.columns
X_chitra = np.array(MINST_chitra.drop('class', axis = 1))
X_chitra

y_chitra = np.array(MINST_chitra['class'])
y_chitra

some_digit1=X_chitra[7].reshape(28,28)
some_digit2=X_chitra[5].reshape(28,28)
some_digit3=X_chitra[0].reshape(28,28)

plt.imshow(some_digit1)
plt.imshow(some_digit2)

plt.imshow(some_digit3)

y_chitra = y_chitra.astype(np.uint8)
y_chitra

y_chitra_grouped = y_chitra

y_chitra_grouped= np.where(y_chitra_grouped <= 3,0,y_chitra_grouped )

y_chitra_grouped = np.where((y_chitra_grouped>3) & (y_chitra_grouped<=6),1,y_chitra_grouped)

y_chitra_grouped = np.where(y_chitra_grouped> 6,9, y_chitra_grouped)


unique, counts = np.unique(y_chitra_grouped,return_counts = True)
print(np.array((unique, counts)).T)


fig_bar = plt.figure()
axes = plt.axes()
axes.set_xticks([0,1,9])
plt.xlabel("Target")
plt.ylabel("Qunatity")
plt.bar(unique,counts)


X_train=X_chitra[:50000]
y_train=y_chitra[:50000]

X_test=X_chitra[20000:]
y_test=y_chitra[20000:]


                                          # Naive Ba
from sklearn.naive_bayes import GaussianNB

NB_clf_chitra = GaussianNB()

NB_clf_chitra.fit(X_train,y_train)

y_pred = NB_clf_chitra.predict(X_test)

y_pred


from sklearn.model_selection import KFold, cross_val_score

Kfold_score = cross_val_score(NB_clf_chitra, X_chitra, y_chitra, cv=3, scoring='accuracy')
Kfold_score

Kfold_score.mean()

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

confusion_mat = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Pridicted'])

confusion_mat

NB_clf_chitra.predict(X_chitra[[7]])

NB_clf_chitra.predict(X_chitra[[5]])

NB_clf_chitra.predict(X_chitra[[0]])

                 #                   Logistic regression
                 
from sklearn.linear_model import LogisticRegression

                          # solver = lbfgs
LR_clf_chitra_lbfgs = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter = 1200, tol = 0.1)

LR_clf_chitra_lbfgs

LR_clf_chitra_lbfgs.fit(X_train, y_train)

LR_clf_chitra_lbfgs.score(X_train, y_train)

 # LR_clf_chitra_saga.score(X_train, y_train)solver = Saga
 
LR_clf_chitra_saga = LogisticRegression(solver='saga', multi_class='multinomial', max_iter = 1200, tol = 0.1)

LR_clf_chitra_saga

LR_clf_chitra_saga.fit(X_train, y_train)

LR_clf_chitra_saga.score(X_train, y_train)

Kfold_score_logit = cross_val_score(LR_clf_chitra_lbfgs, X_chitra, y_chitra, cv=3, scoring='accuracy')
Kfold_score_logit

Kfold_score_logit.mean()

y_predicted_lbfgs = LR_clf_chitra_lbfgs.predict(X_test)

y_predicted_lbfgs

accuracy_score(y_test, y_predicted_lbfgs)


Kfold_score_logit = cross_val_score(LR_clf_chitra_saga, X_chitra, y_chitra, cv=3, scoring='accuracy')

Kfold_score_logit

Kfold_score_logit.mean()


y_predicted_saga = LR_clf_chitra_saga.predict(X_test)

y_predicted_saga

accuracy_score(y_test, y_predicted_saga)



confusion_matrix = pd.crosstab(y_test, y_predicted_lbfgs, rownames=['Actual'], colnames=['Pridicted'])

confusion_matrix

confusion_matrix = pd.crosstab(y_test, y_predicted_saga, rownames=['Actual'], colnames=['Pridicted'])

confusion_matrix


from sklearn.metrics import precision_score, recall_score

precision_score(y_test, y_predicted_lbfgs, average='weighted')

recall_score(y_test, y_predicted_lbfgs, average='weighted')

precision_score(y_test, y_predicted_saga, average='weighted')

recall_score(y_test, y_predicted_saga, average='weighted')


# Use the classifier that worked from the above point to predict the three variables you defined in
# point 7 above. Note the results in your written response and compare against the actual results


LR_clf_chitra_lbfgs.predict(X_chitra[[7]])

LR_clf_chitra_lbfgs.predict(X_chitra[[5]])

LR_clf_chitra_lbfgs.predict(X_chitra[[0]])

LR_clf_chitra_saga.predict(X_chitra[[7]])

LR_clf_chitra_saga.predict(X_chitra[[5]])

LR_clf_chitra_saga.predict(X_chitra[[0]])



































