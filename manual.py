import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading and visualizing data using scatter plot
CSV_Data = pd.read_csv("D:/heartrate.csv")
X = CSV_Data.iloc[:,0:7]
Y = CSV_Data.iloc[:,7]

Y = np.array(Y[:,np.newaxis]) #Making 2D as X is already 2D no need to add axis



# Defining important values
m,c = X.shape
iterations  =100000
alpha  =0.01
theta = np.zeros((c+1,1))
ones = np.ones((m,1))
X = np.hstack((ones,X))

from sklearn.model_selection import train_test_split
# Splitting train and test data in 7:3 ratio
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

meanX_train = np.mean(X_train)
stdX_train = np.std(X_train)

# Scalling Features to Standardize them and using Mean and St. Deviation of X_Train for both
X_train2 = (X_train - meanX_train)/stdX_train # 
X_test2 = (X_test - meanX_train)/stdX_train

from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(2)

# Adding Polynomial Features to the Data
X_train3 = poly.fit_transform(X_train2)
X_test3 = poly.transform(X_test2)

m1,c1=X_train3.shape
thetas_p=np.zeros((c1,1))

# sigmaoid fucntion

def sigmoid(h):
    g = 1/(1+np.exp(-h))
    return g
    


# Cost Fucction 

def Get_cost_J(X,Y,Theta,m):
    
    temp1 = np.multiply(Y,np.log(sigmoid(np.dot(X,Theta))))
    temp2 = np.multiply((1-Y),np.log(1-sigmoid(np.dot(X,Theta))))
    
    J  =(-1/m)*np.sum(temp1+temp2)
    return J

# Grdient Decent

def gradient_decent(x,y,theta,alpha,iterations,m):
    history = np.zeros((iterations,1))
    for i in range(iterations):
        z = np.dot(x,theta)
        predictions = sigmoid(z)        
        error = predictions-y
        slope = (1/m)*np.dot(x.T,error)
        theta = theta  - (alpha*slope)
        history[i] = Get_cost_J(x, y, theta, m)
    
    return (theta,history)

# Fitting model for Simple data
theta,hist = gradient_decent(X_train2, Y_train, theta, alpha, iterations, m1)

# Fitting model for polynomial data
theta_p,hist_p =  gradient_decent(X_train3, Y_train, thetas_p, alpha, iterations, m1)

# Predictions for simple
y_pred=sigmoid(np.dot(X_test2,theta))

# Predictions for Polynomial
y_pred_poly=sigmoid(np.dot(X_test3,theta_p))

#converting intermediate values to appropriate values
for i in range(91):
   if y_pred[i] > 0.5:
       y_pred[i]=1        
   else:
       y_pred[i]=0
       
for i in range(91):
   if y_pred_poly[i] > 0.5:
       y_pred_poly[i]=1        
   else:
       y_pred_poly[i]=0

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y_test, y_pred) # Confusion Matrix for Simple
cm2 = confusion_matrix(Y_test, y_pred_poly) # Confusion Matrix for Polynomial
  
print ("Confusion Matrix Simple : \n", cm1,"\n")

accuracy1=(cm1[0,0]+cm1[1,1])/np.sum(cm1)

print("Accuracy with Manual : ",accuracy1,"\n")  # 

print ("Confusion Matrix Polynomial : \n", cm2,"\n")

accuracy2=(cm2[0,0]+cm2[1,1])/np.sum(cm2)

print("Accuracy with Polynomial : ",accuracy2,"\n")