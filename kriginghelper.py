import numpy as np
from math import*
from numpy.linalg import *

def dis(p1,p2):
    a=pow((pow((p1[0]-p2[0]),2)+pow((p1[1]-p2[1]),2)+pow((p1[2]-p2[2]),2)),0.5)
    return a
def rh(z1,z2):
    r=1/2*pow((z1[3]-z2[3]),2)
    return r
def proportional(x,y):
    xx,xy=0,0
    for i in range(len(x)):
        xx+=pow(x[i],2)
        xy+=x[i]*y[i]
    k=xy/xx
    return k


def OKkriging(h_data):
    h_data=np.array(h_data)
    r=[]#半变异函数值
    pp=[]
    p=[]#半变异函数距离矩阵
    for i in range(len(h_data)):
        pp.append(h_data[i])
    for i in range(len(pp)):
        for j in range(len(pp)):
            p.append(dis(pp[i],pp[j]))
            r.append(rh(pp[i],pp[j]))
    r=np.array(r).reshape(len(h_data),len(h_data))
    r=np.delete(r,len(h_data)-1,axis =0)
    r=np.delete(r,len(h_data)-1,axis =1)

    h=np.array(p).reshape(len(h_data),len(h_data))
    h=np.delete(h,len(h_data)-1,axis =0)
    oh=h[:,len(h_data)-1]
    h=np.delete(h,len(h_data)-1,axis =1)

    hh=np.triu(h,0)
    rr=np.triu(r,0)
    r0=[]
    h0=[]
    for i in range(len(h_data)-1):
        for j in range(len(h_data)-1):
            if i<j:
                ah=h[i][j]
                h0.append(ah)
                ar=rr[i][j]
                r0.append(ar)
          
    k=proportional(h0,r0)
    hnew=h*k
    a2=np.ones((1,len(h_data)-1))
    a1=np.ones((len(h_data)-1,1))
    a1=np.r_[a1,[[0]]]
    hnew=np.r_[hnew,a2]
    hnew=np.c_[hnew,a1]
    # print('半方差联立矩阵：\n',hnew)
    oh=np.array(k*oh)
    oh=np.r_[oh,[1]]
    w=np.dot(inv(hnew),oh)
    # print('权阵运算结果：\n',w)
    z0,s2=0,0
    for i in range(len(h_data)-1):
        z0=w[i]*h_data[i][3]+z0
        s2=w[i]*oh[i]+s2
    s2=s2+w[len(h_data)-1]
    return z0
    print('未知点高程值为：\n',z0)
    # print('半变异值为：\n',pow(s2,0.5))



if __name__ == "__main__":
    
    a=[[69.,76.,60,20.82],[59.,64.,60,10.91],[75.,52. ,60,10.38],[86. , 73.,60, 14.6 ],[69. ,67.,60 , 0.  ]]
 
    
    z0=OKkriging(a)
    print(z0)
