import pandas as pd
import matplotlib.pyplot as plt
import statistics as sta
import numpy as np
from sklearn.metrics import r2_score

data=pd.read_csv('C:/Users/SUDHANSHU/Downloads/car data.csv')
# data.info() # To know about your Data

# Getting the dummie variables
dummy1=pd.get_dummies(data.Fuel_Type)
dummy2=pd.get_dummies(data.Transmission)
dummy3=pd.get_dummies(data.Seller_Type)

data=pd.concat([data,dummy1,dummy2,dummy3],axis=1)
data=data.drop(['Car_Name','CNG','Dealer','Manual','Transmission','Seller_Type','Fuel_Type'],axis=1)

price1=data.get('Present_Price')
year1=data.get('Year')
sp1=data.get('Selling_Price')
kmd1=data.get('Kms_Driven')
disel11=data.get('Diesel')
disel22=data.get('Petrol')
auto1=data.get('Automatic')
deal1=data.get('Individual')
data[['Diesel','Petrol','Automatic','Individual']]=data[['Diesel','Petrol','Automatic','Individual']].astype(int)

price=[]
year=[]
sp=[]
kmd=[]
disel1=[]
disel2=[]
auto=[]
deal=[]

def Normalization(A,B):
    for i in range(len(A)):
        B.append((A[i]-sta.mean(A))/sta.stdev(A))

#  Normalization of all the data

Normalization(price1,price)
Normalization(year1,year)
Normalization(sp1,sp)
Normalization(kmd1,kmd)
Normalization(disel11,disel1)
Normalization(disel22,disel2)
Normalization(auto1,auto)
Normalization(deal1,deal)
print(type(deal1),type(deal))
# initionalization of the varibles
m1=0
m2=0
m3=0
m4=0
m5=0
m6=0
m7=0
c=0

L=0.01 # LEARNING RATE

def Grad_decent(A,B):
    Sa=0
    for i in range(len(A)):
        if type(B)==int:
            s = (m1 * year[i] + m2 * sp[i] + m3 * kmd[i]+m4*disel1[i]+m5*disel2[i]+m6*auto[i]+m7*deal[i]+ c - price[i])
        else:
            s = (m1 * year[i] + m2 * sp[i] + m3 * kmd[i]+m4*disel1[i]+m5*disel2[i]+m6*auto[i]+m7*deal[i] + c - price[i]) * B[i]
        Sa=Sa+s
    return Sa
# For undating the variables

def Update(A,B):
    A=A-(L*B)/len(price)
    return A


JJ=[] # for Storing the values of the Cost function only for graph purpose
ii=[]

for i in range(10000):
    ii.append(i)
    J = 0
    for l in range(len(price)):
        ss = (c + m1 * year[l] + m2 * sp[l] + m3 * kmd[l] + m4 * disel1[l] + m5 * disel2[l] + m6 * auto[l] + m7 * deal[l] - price[l]) ** 2
        J = J + ss
    J1 = J / (2 * len(price))
    JJ.append(J1)

    S1 = Grad_decent(price, 1)
    S2 = Grad_decent(year, year)
    S3 = Grad_decent(sp, sp)
    S4 = Grad_decent(kmd, kmd)
    S5= Grad_decent(disel1,disel1)
    S6=Grad_decent(disel2,disel2)
    S7=Grad_decent(auto,auto)
    S8=Grad_decent(deal1,deal1)
    c = Update(c, S1)
    m1 = Update(m1, S2)
    m2 = Update(m2, S3)
    m3 = Update(m3, S4)
    m4 = Update(m4, S5)
    m5 = Update(m5, S6)
    m6 = Update(m6, S7)
    m7 = Update(m7, S8)

# print(m1,m2,m3,m4,m5,m6,m7,c) to get the values.


# Next we are changing the data types for calculation purpose
price=np.array(price)
year=np.array(year)
sp=np.array(sp)
kmd=np.array(kmd)
disel1=np.array(disel1)
disel2=np.array(disel2)
auto=np.array(auto)
deal=np.array(deal)

y=(m1 * year + m2 * sp + m3 * kmd+m4*disel1+m5*disel2+m6*auto+m7*deal)
r2=r2_score(price,y)

print('The R Sq is :',r2)
# This is the graph between the Cost function an no. of iterations , it helps in deciding the values of Learning rate and no of iteration.
plt.scatter(ii,JJ)
plt.show()