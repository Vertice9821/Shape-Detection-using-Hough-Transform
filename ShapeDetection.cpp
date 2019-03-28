#include <vector>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;

int main()
{
	// load the source image
	const char* filename_src = "src.png";
	Mat img = imread(filename_src, IMREAD_GRAYSCALE);
	//edge detection using canny
	Mat img_canny;
	Canny(img,img_canny,50,200,3);
	// save as 1155123643-edge.png
	const char* filename_edge = "edge.png";
	imwrite(filename_edge, img_canny);
	// copy edge to BGR image
	Mat shape,dst;
	cvtColor(img_canny, shape, COLOR_GRAY2BGR);
	cvtColor(img, dst, COLOR_GRAY2BGR);
	//imshow("Edge Image", img_canny);
	//find the contours
	vector < vector <Point > > contours;
	findContours(img_canny, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	

	for (int j = 0; j < contours.size(); j++)
	{	
		Mat mask = Mat::zeros(img_canny.size(), CV_8UC1);
		// draw the jth segment
		drawContours(mask, contours, j, Scalar(255));
		imshow("segments", mask);
		// hough line detection
		vector<Vec4i> lines;
		HoughLinesP(mask, lines, 2, CV_PI / 180, 40, 50, 20);
		
		if (lines.size() < 15)
		{
			for (size_t i = 0; i < lines.size(); i++)
			{
				Vec4i l = lines[i];
				line(shape, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
				line(dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
			}
			
			switch (lines.size())
			{
			case 3:
				drawContours(dst, contours, j, Scalar(109,15,117), CV_FILLED);
				break;
			case 4:
				drawContours(dst, contours, j, Scalar(0, 163, 221), CV_FILLED);
				break;
			case 5:
				drawContours(dst, contours, j, Scalar(176, 223, 244), CV_FILLED);
				break;
			case 6:
				drawContours(dst, contours, j,Scalar(173, 202,25), CV_FILLED);
				break;
			default:
				drawContours(dst, contours, j, Scalar(129,221, 174), CV_FILLED);
				break;

			}
		}
	}
	imwrite("houghlines.png", shape);

	vector<Vec3f> circles;
	HoughCircles(img_canny, circles, HOUGH_GRADIENT,
		2, img_canny.rows / 2, 200, 100);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// draw the circle outline
		circle(shape, center, radius, Scalar(0, 0, 255), 3, 8, 0);
		circle(dst, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}
	imwrite("houghlines-circles.png", shape);
	imshow("detected", shape);
	imshow("output", dst);
	imwrite("result.png", dst);
	waitKey();
	return 0;
}


