#pragma once

#include <opencv2/opencv.hpp>
#include "Convolution.h"
#include "EdgeDetector.h"

using namespace cv;

class EdgeDetector
{
public:
	/*
	Hàm phát hiện biên cạnh của ảnh xám
	sourceImage: ảnh input
	destinationImage: ảnh output
	method: phương pháp detect
		1: Sobel
		2: Prewitt
		3: Laplace		
	Hàm trả về
		0: nếu detect thành công
		1: nếu detect thất bại (không đọc được ảnh input,...)
	*/
	int DetectEdge(const Mat& sourceImage, Mat& destinationImage, int method);	
	EdgeDetector() {}
	~EdgeDetector() {}
};

int EdgeDetector::DetectEdge(const Mat& sourceImage, Mat& destinationImage, int method)
{
	//Ảnh input trống
	if (sourceImage.empty())
		return 1;

	/*
	Nếu là method 3, chỉ cần 1 kernel:
		|1  1 1|
		|1 -8 1|
		|1  1 1|
	Nhân chập rồi dò zero-crossing
	*/
	if (method == 3)
	{
		Convolution convolution;
		vector<float> kernel;
		Mat laplacian;

		kernel.push_back(1); kernel.push_back(1); kernel.push_back(1);

		kernel.push_back(1); kernel.push_back(-8); kernel.push_back(1);

		kernel.push_back(1); kernel.push_back(1); kernel.push_back(1);

		convolution.SetKernel(kernel, 1.0f, 3, 3);
		convolution.DoConvolution(sourceImage, laplacian);

		/*
		Tìm điểm ảnh zero-crossing (điểm đảo dấu)
			{+, 0, -}, {-, 0, +}, {+, -}, {-, +}
		Duyệt qua tất cả điểm ảnh:
			Nếu điểm đang xét bằng 0, kiểm tra từng cặp điểm ảnh theo 4 phương: dọc, ngang, chéo xuôi, chéo ngược
			Ngược lại, chỉ cần kiểm tra điểm bên cạnh theo 3 hướng: dưới, phải, phải-dưới
		Thao tác duyệt tương tự hàm nhân chập (đọc comment hàm nhân chập để hiểu cách duyệt)
		*/
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, CV_8UC1, Scalar(0));
		int rowStep = sourceImage.cols;

		short* pRowLap = (short*)(laplacian.data) + rowStep;
		uchar* pRowDst = destinationImage.data + rowStep;
		int threshold = round(float(sourceImage.cols)/float(sourceImage.rows)*12.0);

		for (int i = 1; i < sourceImage.rows - 1; i++, pRowLap += rowStep, pRowDst += rowStep)
		{
			short* pDataLap = pRowLap;
			uchar* pDataDst = pRowDst;
			for (int j = 1; j < sourceImage.cols - 1; j++, pDataLap++, pDataDst++)
				if (*pDataLap == 0)
				{
					if (pDataLap[rowStep] * pDataLap[-rowStep] < 0 && abs(pDataLap[rowStep] - pDataLap[-rowStep]) >= threshold)
						*pDataDst = 255;
					if (pDataLap[1] * pDataLap[-1] < 0 && abs(pDataLap[1] - pDataLap[-1]) >= threshold)
						*pDataDst = 255;
					if (pDataLap[rowStep + 1] * pDataLap[-rowStep - 1] < 0 && abs(pDataLap[rowStep + 1] - pDataLap[-rowStep - 1]) >= threshold)
						*pDataDst = 255;
					if (pDataLap[rowStep - 1] * pDataLap[-rowStep + 1] < 0 && abs(pDataLap[rowStep - 1] - pDataLap[-rowStep + 1]) >= threshold)
						*pDataDst = 255;
				}
				else
				{
					if (*pDataLap * pDataLap[rowStep] < 0 && abs(*pDataLap - pDataLap[rowStep]) >= threshold)
						*pDataDst = 255;
					if (*pDataLap * pDataLap[rowStep + 1] < 0 && abs(*pDataLap - pDataLap[rowStep + 1]) >= threshold)
						*pDataDst = 255;
					if (*pDataLap * pDataLap[1] < 0 && abs(*pDataLap - pDataLap[1]) >= threshold)
						*pDataDst = 255;
				}
		}

		return 0;
	}

	//Cần 2 kernel cho mỗi method 1 và method 2
	Convolution convolution;
	vector<float> kernelX;
	vector<float> kernelY;
	float division = 1.0f;
	Mat dx, dy;

	//Tạo kernel nhân chập
	if (method == 1)
	{
		/*
			|1  0  -1|   1
			|2  0  -2| * -  =  Kx
			|1  0  -1|   4
		*/
		kernelX.push_back(1); kernelX.push_back(0); kernelX.push_back(-1);

		kernelX.push_back(2); kernelX.push_back(0); kernelX.push_back(-2);

		kernelX.push_back(1); kernelX.push_back(0); kernelX.push_back(-1);

		/*
			|-1  -2  -1|   1
			| 0   0   0| * -  =  Ky
			| 1   2   1|   4
		*/

		kernelY.push_back(-1); kernelY.push_back(-2); kernelY.push_back(-1);

		kernelY.push_back(0); kernelY.push_back(0); kernelY.push_back(0);

		kernelY.push_back(1); kernelY.push_back(2); kernelY.push_back(1);

		division = 4.0f;
	}
	else if (method == 2)
	{
		/*
			|1  0  -1|   1
			|1  0  -1| * -  =  Kx
			|1  0  -1|   3
		*/
		kernelX.push_back(1); kernelX.push_back(0); kernelX.push_back(-1);

		kernelX.push_back(1); kernelX.push_back(0); kernelX.push_back(-1);

		kernelX.push_back(1); kernelX.push_back(0); kernelX.push_back(-1);

		/*
			|-1  -1  -1|   1
			| 0   0   0| * -  =  Ky
			| 1   1   1|   3
		*/

		kernelY.push_back(-1); kernelY.push_back(-1); kernelY.push_back(-1);

		kernelY.push_back(0); kernelY.push_back(0); kernelY.push_back(0);

		kernelY.push_back(1); kernelY.push_back(1); kernelY.push_back(1);

		division = 3.0f;
	}

	convolution.SetKernel(kernelX, division, 3, 3);
	convolution.DoConvolution(sourceImage, dx);

	convolution.SetKernel(kernelY, division, 3, 3);
	convolution.DoConvolution(sourceImage, dy);

	/*
	Trích chọn điểm ảnh thuộc biên bằng độ lớn của vector gradient:
		e(x, y) = (dx ^ 2 + dy ^ 2) ^ (1/2)
	Những điểm ảnh có giá trị e lớn hơn hay bằng threshold sẽ thuộc biên cạnh
	Tuy nhiên phép lấy căn rất chậm => không cần lấy căn, bình phương threshold lên
	Thao tác duyệt tương tự hàm nhân chập (đọc comment hàm nhân chập để hiểu cách duyệt)
	*/
	destinationImage = Mat(sourceImage.rows, sourceImage.cols, CV_8UC1, Scalar(0));
	int rowStep = sourceImage.cols;

	short* pRowX = (short*)(dx.data);
	short* pRowY = (short*)(dy.data);
	uchar* pRow = destinationImage.data;
	int threshold = 1024;

	for (int i = 0; i < sourceImage.rows; i++, pRowX += rowStep, pRowY += rowStep, pRow += rowStep)
	{
		short* pDataX = pRowX;
		short* pDataY = pRowY;
		uchar* pData = pRow;
		for (int j = 0; j < sourceImage.cols; j++, pDataX++, pDataY++, pData++)
			if (*pDataX * *pDataX + *pDataY * *pDataY >= threshold)
				*pData = 255;
	}

	return 0;
}