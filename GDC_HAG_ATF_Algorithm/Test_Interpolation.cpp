#include <fstream>
#include <libInterpolate/Interpolators/_2D/BilinearInterpolator.hpp>
#include <libInterpolate/Interpolators/_2D/BicubicInterpolator.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "FisheyeEffect.h"

int mainTestInterpolation() {

    using namespace cv;
    using namespace std;
    // 1) Create the gradient image
    Size ImageSize(1280, 720);
    cv::Mat gradientImage(ImageSize, CV_8UC1);

    // Calculate a normalized gradient step
    float gradientStep = 256.0f / (gradientImage.cols - 1);  // Adjust for zero-indexed columns

    for (int x = 0; x < gradientImage.cols; ++x) {
        float gradientValue = x * gradientStep;
        gradientImage.col(x) = cv::Scalar(gradientValue); // Fill with gradient values
    }

    int Grid_Size = 20;
    int size = Grid_Size * Grid_Size;

    Point Grid(Grid_Size, Grid_Size);

    vector<Point> GDC_Fixed_Grid_Points;

    Generate_FixedGrid(gradientImage, GDC_Fixed_Grid_Points, Grid.x, Grid.y);

    _2D::BicubicInterp interp;

    _2D::BicubicInterpolator<double>::VectorType x(size), y(size), z(size);

    cv::Mat mDebug = gradientImage.clone();
    
    cvtColor(mDebug, mDebug, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < size; i++)
    {
        // gnuplot format is essentially row-major
        x(i) = GDC_Fixed_Grid_Points[i].x;
        y(i) = GDC_Fixed_Grid_Points[i].y;
        z(i) = (double)gradientImage.at<uchar>(GDC_Fixed_Grid_Points[i]);

        cv::circle(mDebug, GDC_Fixed_Grid_Points[i], 2, cv::Scalar(0, 255, 0), 2);
    }
    interp.setData(x, y, z);


    // Reconstruct image
    cv::Mat interpolatedImage(gradientImage.size(), CV_8UC1);
    for (int y = 0; y < interpolatedImage.rows; ++y) {
        for (int x = 0; x < interpolatedImage.cols; ++x) {
            interpolatedImage.at<uchar>(y, x) = interp(x, y);
        }
    }

    // Display results (optional)
    displayAndSaveImage(mDebug, "Input Fixed Grid Image");
    displayAndSaveImage(gradientImage, "Ground Truth Image");
    displayAndSaveImage(gradientImage, "Interpolated Image");

    cout << "RMS Error :" << calculateRMSE(gradientImage, interpolatedImage) << "\n";
    cout << "PSNR      :" << getPSNR(gradientImage, interpolatedImage) << "\n";

    cv::waitKey(0);

    return 0;
}
