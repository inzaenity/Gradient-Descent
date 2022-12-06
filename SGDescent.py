import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 

A = np.array([[1, 2, 1, -1], [-1, 1, 0, 2], [0, -1, -2, 1]])
b = np.array([[3], [2], [-2]])
x0 = np.array([[1], [1], [1], [1]])

stepsize = 0.1
gamma = 0.2


def gradFunc(x):
    bT = np.transpose(b)
    AT = np.transpose(A)
    xT = np.transpose(x)
    Ax = np.dot(A, x)
    final = np.dot(AT, Ax-b) + gamma*x
    return final

gradient = gradFunc(x0)
print(gradFunc(x0))
i = 0
while np.linalg.norm(gradient) >= 0.001:
    gradient = gradFunc(x0)
    Next = x0 - stepsize * gradient
    print("k={},    x(k)=[{},{},{},{}]".format(i,np.round(x0[0],4) ,np.round(x0[1],4), np.round(x0[2],4), np.round(x0[3],4)))
    x0 = Next
    i = i + 1

data = pd.read_csv(r"\Users\Daniel\Desktop\CarSeats.csv")
NumData = data.drop(['ShelveLoc','Urban','US'], axis=1)
features=['CompPrice','Income','Advertising','Population','Price','Age','Education']
scaler = StandardScaler()
NumData[features] = scaler.fit_transform(NumData[features])
print(NumData[features].mean())
print(NumData[features].var())
SalesMean = NumData['Sales'].mean()
NumData['Sales'] = NumData['Sales']-SalesMean
X = NumData.iloc[:,1:]
y = NumData.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5,shuffle = False)
print(X_train.head(1))
print(X_train.tail(1))
print(X_test.head(1))
print(X_test.tail(1))
print(y_train.head(1))
print(y_train.tail(1))
print(y_test.head(1))
print(y_test.tail(1))


Ridge = np.linalg.inv(X_train.T@X_train+0.5*200*np.identity(7))@X_train.T@y_train
print(Ridge)

BetaStart = np.array([[1],[1],[1],[1],[1],[1],[1]])
phi = 0.5
n = 200
Xtrain = X_train.values.reshape(200,7)
ytrain = y_train.values.reshape(200,1)



def LossFunction(X, y, beta):
    return (1/n) * np.linalg.norm(y - X@beta, ord = 2 )**2 + phi*np.linalg.norm(beta, ord = 2 )**2
def GradientDescent(X, y,alpha,beta):
    temp = beta
    BetaData = []
    BetaData.append(temp)
    for k in range(1000):
        NextBeta = temp - (alpha/200)*(-2 * np.transpose(X)@(y-X@temp) + n * temp)
        temp = NextBeta
        BetaData.append(NextBeta)
    DeltaData = []
    LossHat = LossFunction(X,y,Ridge)
    for beta in BetaData:        
        Loss = LossFunction(X,y,beta)  
        delta = Loss - LossHat
       
        DeltaData.append(delta)
    return DeltaData

AlphaArray = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
for count in range (9):
    # BetaData = GradientDesc(BetaStart,Xtrain,ytrain,AlphaArray[count])
    
    delta = GradientDescent(Xtrain,ytrain,AlphaArray[count],BetaStart)
    plt.suptitle("Plot of Delta K for ascending step sizes")
    plt.rcParams.update({'font.size': 7})
    plt.subplot(3,3,count + 1)
    k = list(range(0, len(delta)))
    plt.title("Step Size of "+str(AlphaArray[count]),y = 1.08)
    plt.plot(k, delta, color='blue')      
    plt.xlabel("k")                                    
    plt.ylabel("Delta")
    
plt.tight_layout()
plt.show()

temp = BetaStart
for k in range(1000):
        NextBeta = temp - (0.001/200)*(-2 * np.transpose(Xtrain)@(ytrain-Xtrain@temp) + n * temp)
        temp = NextBeta

trainMSE = 1/n * np.linalg.norm(ytrain - Xtrain@temp, ord = 2)**2
Xtest = X_test.values.reshape(200,7)
ytest = y_test.values.reshape(200,1)
testMSE = 1/n * np.linalg.norm(ytest - Xtest@temp, ord = 2)**2
print(trainMSE,testMSE)


def SGDescent(beta, Xtrain, Ytrain, alpha):
    y = Ytrain.values.reshape(200,1)
    x = np.asarray(Xtrain)
    DeltaData1 = []
    LossHat = LossFunction(x,y,Ridge)
    temp = beta
    BetaData1 = []
    BetaData1.append(temp)
    for j in range(5):
        for i in range(200):
            X = Xtrain.iloc[i]
            Y = Ytrain.iloc[i]
            Xasarray = X.values.reshape(7,1)
            BetaNext = temp - alpha*(-2*Xasarray@(np.asarray(Y)-np.transpose(Xasarray)@temp)+temp)
            temp = BetaNext
            BetaData1.append(BetaNext)
    for B in BetaData1:
        loss = np.linalg.norm(y - x@B, ord = 2 )**2/200 + 0.5*np.linalg.norm(B, ord = 2 )**2
        delta = loss - LossHat
        DeltaData1.append(delta)
    return DeltaData1 

AlphaArray = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.006, 0.02]
for count in range (9):
    DeltaData1 = SGDescent(BetaStart, X_train, y_train,AlphaArray[count])
    plt.suptitle("Plot of Delta K for ascending step sizes")
    plt.rcParams.update({'font.size': 7})
    plt.subplot(3,3,count + 1)
    k = list(range(0, len(DeltaData1)))
    plt.title("Step Size of "+str(AlphaArray[count]),y = 1.08)
    plt.plot(k, DeltaData1, color='blue')      
    plt.xlabel("k")                                    
    plt.ylabel("Delta")
    
plt.tight_layout()
plt.show()
temp = BetaStart
for k in range(5):
    for i in range(200):
            X = X_train.iloc[i]
            Y = y_train.iloc[i]
            Xasarray = X.values.reshape(7,1)
            BetaNext = temp - 0.001*(-2*Xasarray@(np.asarray(Y)-np.transpose(Xasarray)@temp)+temp)
            temp = BetaNext
trainMSE = 1/n * np.linalg.norm(ytrain - Xtrain@temp, ord = 2)**2
testMSE = 1/n * np.linalg.norm(ytest - Xtest@temp, ord = 2)**2
print(trainMSE,testMSE)

delta = GradientDescent(Xtrain,ytrain,AlphaArray[count],BetaStart)
DeltaData1 = SGDescent(BetaStart, X_train, y_train,AlphaArray[count])
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
overlapping = 0.9
k = list(range(0, len(DeltaData1)))
line1 = plt.plot(k,delta, c='lightgreen', label = "GD, step size 0.001",alpha=overlapping, lw=2)
line2 = plt.plot(k,DeltaData1, c='darkorange', label = "SGD, step size 0.001", alpha=overlapping,
lw=2)
plt.legend(loc="upper right")
plt.xlabel("k")                                    
plt.ylabel("Delta")
plt.show()
