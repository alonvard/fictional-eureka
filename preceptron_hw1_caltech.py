# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:55:19 2020

@author: alonvardy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def calc_prob(X):
    y_xn=-1*-X[1,0]/X[2,0]-X[0,0]/X[2,0]
    y_xp=1*-X[1,0]/X[2,0]-X[0,0]/X[2,0]
    
    if y_xn<=-1 and y_xp<1:
        x_yn=-1*-X[2,0]/X[1,0]-X[0,0]/X[1,0]
        p_blue=(1-x_yn)*(y_xp+1)/2/4
        return [p_blue,1]    
    elif y_xn<=-1 and y_xp>=1:
        x_yn=-1*-X[2,0]/X[1,0]-X[0,0]/X[1,0]
        x_yp=1*-X[2,0]/X[1,0]-X[0,0]/X[1,0]
        p_blue=(1-x_yp+1-x_yn)*2/2/4
        return [p_blue,2]    
    elif -1<y_xn<1 and y_xp<=-1:
        x_yn=-1*-X[2,0]/X[1,0]-X[0,0]/X[1,0]
        p_blue=(x_yn+1)*(y_xn+1)/2/4
        return [p_blue,3]           
    elif -1<y_xn<1 and y_xp>=1:
        x_yp=1*-X[2,0]/X[1,0]-X[0,0]/X[1,0]
        p_red=(x_yp+1)*(1-y_xn)/2/4
        p_blue=1-p_red
        return [p_blue,4]                
    elif -1<y_xn<1 and -1<y_xp<1:
        p_blue=((y_xn+1)+(y_xp+1))*2/2/4
        return [p_blue,5]                             
    elif y_xn>=1 and y_xp<=-1:
        x_yn=-1*-X[2,0]/X[1,0]-X[0,0]/X[1,0]
        x_yp=1*-X[2,0]/X[1,0]-X[0,0]/X[1,0]
        p_blue=(x_yn+1+x_yp+1)*2/2/4
        return [p_blue,6]           
    elif y_xn>=1 and y_xp>-1:
        x_yp=1*-X[2,0]/X[1,0]-X[0,0]/X[1,0]
        p_red=(1-x_yp)*(1-y_xp)/2/4
        p_blue=1-p_red
        return [p_blue,7] 

T=1 # number of trails
N=100 # sample size

dout=np.zeros([T,11])

for k in range(0,T):
    SYn=N
    while SYn==-N or SYn==N: #make sure there is both red and blue data
        t=np.random.uniform(-1, 1, [2,2])
        
        Tp=np.polyfit(t[:,0],t[:,1],1)
        Tx=np.array([[-1, 1]])
        Ty=np.polyval(Tp,Tx)
        Tv=np.concatenate(((np.flipud(Tp)),np.array([-1])))/Tp[1]
        Tv=np.expand_dims(Tv, axis=0).transpose()
        Ef=calc_prob(Tv)
        
        Xn=np.concatenate((np.ones([N,1]),np.random.uniform(-1,1,[N,2])), axis=1)
        Yn=np.sign(np.dot(Xn,Tv))
        SYn=sum(Yn)
    
    #PLA
    MAX_iter=100
    i=1
    Ein=np.zeros([N,1])
    W=np.zeros([3,1])
    
    while i<MAX_iter and np.sum(Ein)<N:
        dE=np.where(Ein<=0)
        Pi=np.random.choice(dE[0],1)
        W=W+np.transpose(Xn[Pi]*Yn[Pi])
        Ein=np.sign((np.dot(Xn,W))*Yn)
        i+=1

    Eg=calc_prob(W)
    dout[k,0]=i
    dout[k,1]=Ef[0]   
    dout[k,2]=Ef[1]
    dout[k,3]=Tv[0]
    dout[k,4]=Tv[1]
    dout[k,5]=Tv[2]    
    dout[k,6]=Eg[0]   
    dout[k,7]=Eg[1]   
    dout[k,8]=W[0]
    dout[k,9]=W[1]
    dout[k,10]=W[2]    
 
col=['red','green','blue']

fig1=plt.figure()  


df = pd.DataFrame(np.concatenate((Xn,Yn), axis=1), columns=["X0","X1","X2","Y"])
value=df['Y']==1
df['color']= np.where(value==True , "red", "blue")

sns.regplot(data=df, x="X1", y="X2", fit_reg=False, scatter_kws={'facecolors':df['color'],'edgecolors':df['color'],'s':50})


plt.plot(t[:,0],t[:,1],'ko',Tx[0,:],Ty[0,:],'k',markersize=12)
plt.axvline(x=0,ymin=-1,ymax=1,color='k',linestyle='--')
plt.axhline(y=0,xmin=-1,xmax=1,color='k',linestyle='--')
plt.xlim(-1,1)
plt.ylim(-1, 1)
fig1.suptitle('perceptron learning algorithm',fontsize=15)       


Wy=-Tx*W[1,0]/W[2,0]-W[0,0]/W[2,0]
plt.plot(Tx[0,:],Wy[0,:],'g')
#print("number of iterations:",i)
#print("P_blue_f:",Ef)
#print("P_blue_g:",Eg)

b=np.sum(dout,0)/T
print(b)



