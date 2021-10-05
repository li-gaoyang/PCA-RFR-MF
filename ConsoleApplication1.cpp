// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <Map>
#include <vector>
#include <vtkSmartPointer.h>
#include <vtkParametricFunctionSource.h>
#include <vtkParametricSpline.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkGlyph3DMapper.h>
#include <vtkSphereSource.h>
#include <vtkNamedColors.h>
#include <vtkAutoInit.h> 
#include <vtkLine.h>
#include "vtkUnstructuredGrid.h"
#include <vtkHexahedron.h> 
#include "vtkDataSetMapper.h"
#include "vtkContourFilter.h"
#include "vtkFloatArray.h"
#include "vtkDataSetAttributes.h"
#include "vtkDataSet.h"
#include "vtkPointData.h"
#include "vtkScalarBarActor.h"
#include "vtkLookupTable.h"
#include "vtkUnstructuredGridWriter.h"
#include<iostream>
#include<fstream>
#include <windows.h>
#include<iostream>
#include<string>
#include<sstream>
#include<vector>
#include "vtkPolyDataWriter.h"
using namespace std;

vector<vector<double>> getxyzs(string file)
{
	vector<vector<double>> xyzs(0);

	ifstream infile;
	infile.open(file.data());   //将文件流对象与文件连接起来 


	string s;
	while (getline(infile, s))
	{
		//用于存放分割后的字符串 
		vector<double> xyz;
		//待分割的字符串，含有很多空格 
		string word = s;
		//暂存从word中读取的字符串 
		string result;
		//将字符串读到input中 
		stringstream input(word);
		//依次输出到result中，并存入res中 
		while (input >> result)
			xyz.push_back(stof(result));



		cout << s << endl;
		xyzs.push_back(xyz);
	}
	infile.close();
	return xyzs;
}

int main()
{

	//string file = "IDW_Interpolation_result.txt";
	//string file = "OK_Interpolation_result.txt";
	string file = "RFR_Interpolation_result.txt";
	
	//string file = "res_blur.txt";
	vector<vector<double>> xyzs = getxyzs(file);
	auto points = vtkSmartPointer<vtkPoints>::New();//point点集存放I*J*K个点
	auto  unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();//实例化非结构化网格
	auto hexahedronActiveScalars = vtkSmartPointer <vtkFloatArray>::New();//活动网格的标量集
	unstructuredGrid->Allocate(1, 1);
	int seq = 0;

	for (auto lin : xyzs) {
		double x = lin[0];
		double y = lin[1];
		double z = lin[2];
		/*if (x != 0 || y != 0 || z != 0 || x != 29 || y != 29 || z != 29)
		{*/
			auto hexahedron = vtkSmartPointer <vtkHexahedron>::New();//实例化六面体
			points->InsertNextPoint(0 + x, 0 + y, 0 + z);
			points->InsertNextPoint(0 + x, 1 + y, 0 + z);
			points->InsertNextPoint(1 + x, 1 + y, 0 + z);
			points->InsertNextPoint(1 + x, 0 + y, 0 + z);
			points->InsertNextPoint(0 + x, 0 + y, 1 + z);
			points->InsertNextPoint(0 + x, 1 + y, 1 + z);
			points->InsertNextPoint(1 + x, 1 + y, 1 + z);
			points->InsertNextPoint(1 + x, 0 + y, 1 + z);


			for (int i = 0; i < 8; i++)
			{
				hexahedron->GetPointIds()->SetId(i, seq);
				seq++;
			}


			unstructuredGrid->InsertNextCell(hexahedron->GetCellType(), hexahedron->GetPointIds());//插入单元--六面体
			/*if (lin[3] > 1)lin[3] = lin[3] * 0.1;*/
			hexahedronActiveScalars->InsertNextTuple1(lin[3]);//如果不是无效值，就把对应的属性值插入
			cout << x << "---" << y << "---" << z << "---" << lin[3] << endl;
		//}
	}
	unstructuredGrid->SetPoints(points);//非结构化网格中插入点





	hexahedronActiveScalars->SetName("property");//给标量数组命名
	unstructuredGrid->GetCellData()->SetScalars(hexahedronActiveScalars);

	vtkSmartPointer<vtkUnstructuredGridWriter> writer =
		vtkSmartPointer<vtkUnstructuredGridWriter>::New();

	
	writer->SetFileName("RFR_Interpolation_result.vtk");
	writer->SetInputData(unstructuredGrid);
	writer->Write();



	cout << "lllll" << endl;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
