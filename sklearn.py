import numpy as np
import pandas as pd

# Reading and visualizing data using scatter plot
CSV_Data = pd.read_csv("D:/heartrate.csv")
X = CSV_Data.iloc[:,0:7]
Y = CSV_Data.iloc[:,7]

Y = np.array(Y[:,np.newaxis]) #Making 2D as X is already 2D no need to add axis

from sklearn.model_selection import train_test_split
# Splitting train and test data in 7:3 ratio
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(2)

scale = StandardScaler()

# Scalling Features to Standardize them and using Mean and St. Deviation of X_Train for both
X_train2 = scale.fit_transform(X_train)
X_test2 = scale.transform(X_test) 

# Adding Polynomial Features to the Data
X_train3=poly.fit_transform(X_train2)
X_test3=poly.transform(X_test2)

from sklearn.linear_model import LogisticRegression

logr1 = LogisticRegression()
logr2 = LogisticRegression()

logr1.fit(X_train2,Y_train) # Fitting model for Simple data
logr2.fit(X_train3,Y_train) # Fitting model for polynomial data

y_pred = logr1.predict(X_test2) # Predictions for simple
y_pred_poly = logr2.predict(X_test3) # Predictions for Polynomial


from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y_test, y_pred) # Confusion Matrix for Simple
cm2 = confusion_matrix(Y_test, y_pred_poly) # Confusion Matrix for Polynomial
  
print ("Confusion Matrix Simple : \n", cm1,"\n")

accuracy1=(cm1[0,0]+cm1[1,1])/np.sum(cm1)

print("Accuracy with sklearn : ",accuracy1,"\n")  # 

print ("Confusion Matrix Polynomial : \n", cm2,"\n")

accuracy2=(cm2[0,0]+cm2[1,1])/np.sum(cm2)

print("Accuracy with Polynomial : ",accuracy2,"\n")