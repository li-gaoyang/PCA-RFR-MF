# PCA-RFR-MF
grid100_int.txt表示100个样本点的坐标
property100.txt表示100个样本点的属性值
IDW_Interpolation.py表示IDW插值
OK_ interpolation.py表示0K插值
RFR_Interpolation.py表示PCA-RFR-MF插值

三维显示主要用VTK类库，但是python的VTK类库接口说明不完整，所以用C++写了个ConsoleApplication1.cpp。主要目的是把python插值生成的txt转换为VTK类型的文件。然后用paraview显示。
