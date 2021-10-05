import numpy as np
import vtk
import os
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.fromnumeric import size
import pandas as pd
import math

import random
###########4.预设回归方法##########
####随机森林回归####
from sklearn import ensemble
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import matplotlib.image as mpimg # mpimg 用于读取图片

import scipy.interpolate as si
import threadpool
import time

fp1 = open('grid100_int.txt')
fp2 = open('property100.txt')

#65*60*5的三维网格
xyzs=[]
for i in range(30):
    for j in range(30):
        for k in range(30):
            xyzs.append([i,j,k])
        

grid100=[]
for line1 in fp1:
    t1=line1.replace(' \n','').split(' ')
    _x=int(t1[0])
    _y=int(t1[1])
    _z=int(t1[2])
    tt1=[_x,_y,_z]
  
    print(tt1)
    grid100.append(tt1)
grid1002=grid100.copy()
property100=[]
for line2 in fp2:
    t2=float(line2)
    property100.append(t2)

print("______")

for idx in range(len(grid100)):
    grid100[idx]=[grid100[idx][0],grid100[idx][1],grid100[idx][2],property100[idx]]




#定义坐标轴

fig = plt.figure()

ax1 = plt.axes(projection='3d')


x=[]
y=[]
z=[]

for i in grid100:
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])


# ax1.scatter3D(x,y,z, cmap='Blues') #绘制散点图



# plt.show()

#计算两点之间的距离
def distance(x1, y1,z1, x2, y2,z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2+(z1-z2)**2)




neighbour=4#邻域个数

def calv(i,j,k,xyz_nei):
    val1=0
    val2=0
    for ii in xyz_nei:
        d=distance(ii[0],ii[1],ii[2],i,j,k)
        if d!=0:
            val1=val1+1/d*ii[3]
            val2=val2+1/d
    
    return val1/val2

ress=[]
for i in range(30):
    for j in range(30):
        for k in range(30):
            v0=[i,j,k]
            dist=[]
            for n in grid100:
                 #遍历所有的样本点，计算未知点到样本点之间的距离
                _dist=distance(n[0],n[1],n[2],i,j,k)
                dist.append(_dist)
            xyz_nei=[]
            while len(xyz_nei)<neighbour:
                for ii in range(0,size(dist)):
                    if min(dist)==dist[ii]:
                        xyz_nei.append(grid100[ii])
                        dist[ii]=999999#移除0，即未知点本身到本身的距离
                        break
            val=calv(i,j,k,xyz_nei)
            ress.append([i,j,k,val])            
            print(i," ",j," ",k," ",val)
     


with open("IDW_Interpolation_result.txt","w") as f:
    for i in ress:
        f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+' '+str(i[3])+'\n')

print("-----over------")
            





