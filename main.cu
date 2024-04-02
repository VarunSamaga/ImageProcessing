#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "src/imgproc.cuh"

void imcopyRGB(unsigned char *src, cv::Mat &dst, const uint rows, const uint cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            cv::Vec3b pixel;
            pixel[2] = src[(i * cols + j) * 3 + 2];
            pixel[1] = src[(i * cols + j) * 3 + 1];
            pixel[0] = src[(i * cols + j) * 3];
            dst.at<cv::Vec3b>(i, j) = pixel;
        }
    }
}

void imcopyGray(unsigned char *src, cv::Mat &dst, const uint rows, const uint cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            dst.at<uchar>(i, j) = src[i * cols + j];
        }
    }
}

int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cout << "Usage: main [FILENAME/FILEPATH]" << std::endl;
        return EXIT_FAILURE;
    }
    std::string fileName = (std::string) argv[1];
    cv::Mat imageGray = cv::imread(fileName + ".jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat imageRGB = cv::imread(fileName + ".jpg", cv::IMREAD_COLOR);
    
    cv::imwrite(cv::format("./Outputs/%s-grayscale.jpg",  fileName.c_str()), imageGray);

    if(imageGray.empty()) {
        std::cout << "Empty image loaded. Please check path or filename." << std::endl;
        return EXIT_FAILURE;
    }

    const uint rows = imageGray.rows, cols = imageGray.cols;
    cv::Mat output(rows, cols, CV_8UC1, cv::Scalar(0));
    cv::Mat outputRGB(rows, cols, CV_8UC3, cv::Scalar(0));

    unsigned char *imgGray = (unsigned char*) malloc(sizeof(unsigned char) * rows * cols);
    unsigned char *imgRGB = (unsigned char*) malloc(sizeof(unsigned char) * rows * cols * 3);
    unsigned char *imoutGray = (unsigned char*) malloc(sizeof(unsigned char) * rows * cols);
    unsigned char *imoutRGB = (unsigned char*) malloc(sizeof(unsigned char) * rows * cols * 3);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            imgGray[i * cols + j] = static_cast<unsigned char>(imageGray.at<uchar>(i, j));
        }
    }

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            cv::Vec3b pixel = imageRGB.at<cv::Vec3b>(i, j);
            imgRGB[(i * cols + j) * 3] = pixel[0];
            imgRGB[(i * cols + j) * 3 + 1] = pixel[1];
            imgRGB[(i * cols + j) * 3 + 2] = pixel[2];
        }
    }

    // Image Negative
    {
    imageNegative(imgGray, imoutGray, rows, cols, 255);

    imcopyGray(imoutGray, output, rows, cols);

    cv::imwrite(cv::format("./Output/%s-negative.jpg",  fileName.c_str()), output);
    }
    
    // log Transform
    {
    logTransform(imgRGB, imoutRGB, rows, cols, 255, 3);

    imcopyRGB(imoutRGB, outputRGB, rows, cols);

    cv::imwrite(cv::format("./Output/%s-logTransform.jpg",  fileName.c_str()), outputRGB);
    }

    // gamma Transform
    {
    gammaTransform(imgGray, imoutGray, rows, cols, 1, .5);

    imcopyGray(imoutGray, output, rows, cols);

    cv::imwrite(cv::format("./Output/%s-gammaTransform.jpg",  fileName.c_str()), output);
    }

    // Intensity Level Slicing
    {
    // Intensity Level Slice Grayscale

    intensitySliceThresh(imgGray, imoutGray, rows, cols, 32, 64, 255);

    imcopyGray(imoutGray, output, rows, cols);

    cv::imwrite(cv::format("./Output/%s-sliceThresh.jpg",  fileName.c_str()), output);

    intensitySliceRange(imgGray, imoutGray, rows, cols, 32, 64, 0);

    imcopyGray(imoutGray, output, rows, cols);

    cv::imwrite(cv::format("./Output/%s-sliceRange.jpg",  fileName.c_str()), output);

    // Intensity Level Slice RGB

    intensitySliceThresh(imgRGB, imoutRGB, rows, cols, 32, 128, 255, 3);

    imcopyRGB(imoutRGB, outputRGB, rows, cols);

    cv::imwrite(cv::format("./Output/%s-sliceThreshRGB.jpg",  fileName.c_str()), outputRGB);

    intensitySliceRange(imgRGB, imoutRGB, rows, cols, 32, 64, 128, 3);

    imcopyRGB(imoutRGB, outputRGB, rows, cols);

    cv::imwrite(cv::format("./Output/%s-sliceRangeRGB.jpg",  fileName.c_str()), outputRGB);
    }

    // Bit Plane Slicing 
    {

    // Bit Plane Slicing Gray
    bitPlaneSlicing(imgGray, imoutGray, rows, cols, 6);

    imcopyGray(imoutGray, output, rows, cols);

    cv::imwrite(cv::format("./Output/%s-bitpSliceGray.jpg",  fileName.c_str()), output);

    // Bit plane Slicing RGB

    bitPlaneSlicing(imgRGB, imoutRGB, rows, cols, 6, 3);

    imcopyRGB(imoutRGB, outputRGB, rows, cols);

    cv::imwrite(cv::format("./Output/%s-bitpSliceRGB.jpg",  fileName.c_str()), outputRGB);

    }

    // Bit Mask 
    {

    // Bit Mask Gray
    bitMaskImage(imgGray, imoutGray, rows, cols, 0b11000100);

    imcopyGray(imoutGray, output, rows, cols);

    cv::imwrite(cv::format("./Output/%s-bitMaskGray.jpg",  fileName.c_str()), output);

    // Bit Mask RGB

    bitMaskImage(imgRGB, imoutRGB, rows, cols, 0b11111100, 3);
    imcopyRGB(imoutRGB, outputRGB, rows, cols);

    cv::imwrite(cv::format("./Output/%s-bitMaskRGB.jpg",  fileName.c_str()), outputRGB);

    }

    int filterSize = 3;
    int *filter = (int*)malloc(sizeof(int) * filterSize * filterSize);

    memset(filter, 1, sizeof(int) * filterSize * filterSize);


    genFilter<int>(imageRGB.data, imoutRGB, rows, cols, filter, filterSize, 3);
    imcopyRGB(imoutRGB, outputRGB, rows, cols);

    cv::imwrite(cv::format("./Output/%s-convFilter.jpg",  fileName.c_str()), outputRGB);
}