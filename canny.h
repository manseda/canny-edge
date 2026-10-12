#pragma once

#include <math.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
// #include <fstream>

#include <opencv2/highgui.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc.hpp>

bool cannyEdgeDetection(const std::string& readLocation, const std::string& writeLocation, double lowerThreshold, double higherThreshold);
std::vector<int> imgToArray(cv::Mat img, uint8_t* pixelPtr, int sizeRows, int sizeCols, int sizeDepth);
void arrayToImg(std::vector<int>& pixels, uint8_t* pixelPtr, int sizeRows, int sizeCols, int sizeDepth);
void gaussianBlur(const std::vector<int>& pixels, std::vector<int>& pixelsBlur, const std::vector<std::vector<double>>& kernel, double kernelConst, int sizeRows, int sizeCols, int sizeDepth);
void rgbToGrayscale(const std::vector<int>& pixels, std::vector<int>& pixelsGray, int sizeRows, int sizeCols, int sizeDepth);
double sobelGradient(const std::vector<int>& pixels, std::vector<double>& G, std::vector<int>& theta, int sizeRows, int sizeCols);
void nonMaximumSuppression(std::vector<double>& G, const std::vector<int>& theta, std::vector<int>& pixelsCanny, double largestG, int sizeRows, int sizeCols);
void doubleThreshold(std::vector<double>& G, std::vector<int>& pixelsCanny, double largestG, double lowerThreshold, double higherThreshold, int sizeRows, int sizeCols);
