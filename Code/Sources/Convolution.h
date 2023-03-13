#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Convolution
{
	vector<float> _kernel;//Lưu giá trị của ma trận kernel dùng để tích chập
	int _kernelWidth;//chiều rộng kernel
	int _kernelHeight;//chiều dài kernel
	float _division;  //Nhân tử chung của kernel: 1/_division
public:
	//trả về kernel 
	vector<float> GetKernel();
	//set kernel, chiều rộng, chiều dài kernel phải là số lẻ
	void SetKernel(vector<float> kernel, float division, int kWidth, int kHeight);

	/*
	Hàm tính convolution của 1 ảnh xám với kernel được xác định trước
	sourceImage: ảnh input
	destinationImage: ảnh output
	Hàm trả về
	0: nếu tính thành công
	1: nếu tính thất bại (không đọc được ảnh input,...)
	*/
	int DoConvolution(const Mat& sourceImage, Mat& destinationImage);	
	Convolution();
	~Convolution();
};

Convolution::Convolution()
{
	_division = 1;
	_kernelWidth = _kernelHeight = 0;
}

Convolution::~Convolution()
{
	_kernel.clear();
	_division = 1;
	_kernelWidth = _kernelHeight = 0;
}

vector<float> Convolution::GetKernel()
{
	return _kernel;
}

void Convolution::SetKernel(vector<float> kernel, float division, int kernelWidth, int kernelHeight)
{
	//Chiều rộng hoặc chiều cao là số chẵn hoặc division = 0
	if ((kernelWidth & 1) == 0 || (kernelHeight & 1) == 0 || division == 0.0f)
		return;

	//Gán và lưu các giá trị vào class
	_kernelWidth = kernelWidth;
	_kernelHeight = kernelHeight;
	_division = division;
	_kernel = kernel;
}

int Convolution::DoConvolution(const Mat& sourceImage, Mat& destinationImage)
{
	//Ảnh input trống hoặc không có kernel
	if (sourceImage.empty() || _kernelHeight <= 0 || _kernelWidth <= 0)
		return 1;
	
	//Tạo ảnh output (ảnh kết quả của phép tích chập là một ảnh vô nghĩa, sẽ xử lý sau)
	//Kiểu dữ liệu ảnh output là CV_16SC1, để có thể lưu được số âm

	destinationImage = Mat(sourceImage.rows, sourceImage.cols, CV_16SC1, Scalar(0));

	/*
	Truy xuất điểm ảnh bằng con trỏ data
	Bỏ qua một vài dòng và cột ở rìa của ảnh để kernel không bị thừa ra ngoài
	Số dòng và cột bị bỏ qua là một nửa các kích thước (bán kính) của kernel
	*/
	int rowStep = sourceImage.cols;

	int halfHeight = _kernelHeight / 2;
	int halfWidth = _kernelWidth / 2;

	uchar* pRowSrc = sourceImage.data + rowStep * halfHeight + halfWidth;
	short* pRowDst = (short*)(destinationImage.data) + rowStep * halfHeight + halfWidth;

	/*
	Duyệt qua từng điểm ảnh (i, j) và nhân chập với kernel
	Phép tích chập được implement sát với lý thuyết:
		dst[i][j] = src[i - u][j - v] * kernel[u][v]
	Mỗi phần tử kernel có chỉ số là (u, v) với:
		-halfHeight <= u <= halfHeight
		-halfWidth <= v <= halfWidth
	Do kernel được lưu dưới dạng mảng một chiều, vị trí của (u, v) trong mảng này là:
		kernelWidth * (u + halfHeight) + (v + halfWidth)
	*/
	for (int i = halfHeight; i < sourceImage.rows - halfHeight; i++, pRowSrc += rowStep, pRowDst += rowStep)
	{
		uchar* pDataSrc = pRowSrc;
		short* pDataDst = pRowDst;
		for (int j = halfWidth; j < sourceImage.cols - halfWidth; j++, pDataSrc++, pDataDst++)
		{
			float value = 0;
			for (int u = -halfHeight; u <= halfHeight; u++)
				for (int v = -halfWidth; v <= halfWidth; v++)
				{
					int virtualPosition = _kernelWidth * (u + halfHeight) + (v + halfWidth);
					value += pDataSrc[-u * rowStep - v] * _kernel[virtualPosition];
				}
			*pDataDst = (short)(value / _division);
		}
	}
	return 0;
}