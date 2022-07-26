import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from statistics import median
import os
import riskslim
import math


# X_test = np.array([[1,1],[1,2],[1,3]])
# a = np.dot(X_test[:,1],3)
# print(a)

c = math.floor((3+2)/2)
print(c)