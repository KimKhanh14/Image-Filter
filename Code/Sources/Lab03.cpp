// Lab03.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "EdgeDetector.h"
#include "Blur.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	string option, inputPath, outputPath;
	Mat desImg;
	int check = -1;

	if (argc == 6) {
		option = argv[1];
		int x = stof(argv[2]);
		int y = stof(argv[3]);
		inputPath = argv[4];
		outputPath = argv[5];
		
		Blur blr;

		Mat sourceImg = imread(inputPath, IMREAD_GRAYSCALE); //Nhập ảnh trắng đen
		
		if (option == "-avg") {
			check = blr.BlurImage(sourceImg, desImg, x, y, 0);
		}
		else if (option == "-med") {
			check = blr.BlurImage(sourceImg, desImg, x, y, 1);
		}
		else {
			check = blr.BlurImage(sourceImg, desImg, x, y, 2);
		}

		imwrite(outputPath, desImg);
		if (check == 0) cout << "Done\n"; else cout << "Fail\n";
	}
	else if (argc == 4) {
		option = argv[1];
		inputPath = argv[2];
		outputPath = argv[3];

		Mat sourceImg = imread(inputPath, IMREAD_GRAYSCALE); //Nhập ảnh trắng đen

		EdgeDetector Ed;

		if (option == "-sobel") {
			check = Ed.DetectEdge(sourceImg, desImg, 1);
		}
		else if (option == "-prew") {
			check = Ed.DetectEdge(sourceImg, desImg, 2);
		}
		else {
			check = Ed.DetectEdge(sourceImg, desImg, 3);
		}
		imwrite(outputPath, desImg);
		if (check == 0) cout << "Done\n"; else cout << "Fail\n";
	}  
}
