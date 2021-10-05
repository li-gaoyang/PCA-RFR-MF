import numpy as np
 
import os
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.fromnumeric import size
import pandas as pd
import math
 
import random
#

from sklearn import ensemble
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import matplotlib.image as mpimg # mpimg 用于读取图片

import scipy.interpolate as si
import threadpool
import time
import kriginghelper as OK

start=time.clock()

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
#计算两点的角度
def calangle(x0,y0,x1,y1):
    deltaX = x1 - x0+0.000001
    deltaY = y1 - y0+0.000001
    tan = deltaX/deltaY
    formula = math.atan(tan)

    formula = formula ** 2
    formula = math.sqrt(formula)

    formula = math.degrees(formula)
    return formula


            
def calkriging(xyz0,neighbour):
    
    dist=[]
    for i in grid100:#遍历所有的样本点，计算未知点到样本点之间的距离
        _dist=distance(i[0],i[1],i[2],xyz0[0],xyz0[1],xyz0[2])
        dist.append(_dist)
    xyz_nei=[]
    while len(xyz_nei)<neighbour:
        for i in range(0,size(dist)):
            if min(dist)==dist[i]:
                xyz_nei.append(grid100[i])
                dist[i]=999999#移除0，即未知点本身到本身的距离
                break
    xyz0.append(0)
    xyz_nei.append(xyz0)
    z0=OK.OKkriging(xyz_nei)
    
    return z0





neighbour=4#邻域个数
xyzs2=[]
for _xyz in xyzs:
    
    if _xyz not in grid1002:
        xyzs2.append(_xyz)
    else:
        xyzs2.append(-1)

ress=[]
iii=0
for _xyz in xyzs2:
    if _xyz==-1:
        _xyz=grid100[iii]
        _xyz2=[_xyz[0],_xyz[1],_xyz[2],_xyz[3]]
        iii=iii+1
        # print("---------------------",_xyz[0],_xyz[1],_xyz[2],_xyz[3])
        ress.append(_xyz2)
    else:

        res=calkriging(_xyz,neighbour)

        _xyz2=[_xyz[0],_xyz[1],_xyz[2],res]
        ress.append(_xyz2)
        print(_xyz[0],_xyz[1],_xyz[2],res)




with open("OK_Interpolation_result.txt","w") as f:
    for i in ress:
        f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+' '+str(i[3])+'\n')

print("-----over------")
            

# end=time.clock()
# import sys
# cost=end-start
# print("%s cost %s second" % (os.path.basename( sys.argv[0]),cost))



