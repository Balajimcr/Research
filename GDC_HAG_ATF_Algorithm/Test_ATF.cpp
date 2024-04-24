#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void ApplyAdaptiveTileFilter(cv::Mat& mSrc, const cv::Mat& magnitude_of_distortion, int d, double sigmaColor, double sigmaSpace, const float lowThreshold, const float highThreshold);
void on_trackbar(int, void*);

Mat mSrc, magnitude_of_distortion; // Assuming magnitude_of_distortion is initialized
int d = 1;
int sigmaColor_int = 75; // Use integers for trackbars
int sigmaSpace_int = 75;
int lowThreshold_slider = 10;
int highThreshold_slider = 98;
const int max_value = 100;

void ApplyAdaptiveTileFilter(cv::Mat& mSrc, const cv::Mat& magnitude_of_distortion, int d, double sigmaColor, double sigmaSpace, const float lowThreshold, const float highThreshold) {
    cv::Mat mediumMask, highMask;
    cv::inRange(magnitude_of_distortion, cv::Scalar(lowThreshold), cv::Scalar(highThreshold), mediumMask);
    cv::inRange(magnitude_of_distortion, cv::Scalar(highThreshold), cv::Scalar(1.0), highMask);

    cv::Mat tempMedium, tempHigh;
    mSrc.copyTo(tempMedium);
    mSrc.copyTo(tempHigh);

    cv::bilateralFilter(mSrc, tempMedium, d, sigmaColor, sigmaSpace); // Use the double values here
    cv::bilateralFilter(mSrc, tempHigh, d * 2, sigmaColor * 2, sigmaSpace * 2);

    // Initialize filter. Kernel size 5x5, threshold 20
    cv::ximgproc::edgePreservingFilter(mSrc, tempMedium, 3, 10);
    cv::ximgproc::edgePreservingFilter(mSrc, tempMedium, 5, 20);

    cv::Mat result = mSrc.clone();
    tempMedium.copyTo(result, mediumMask);
    tempHigh.copyTo(result, highMask);

    result.copyTo(mSrc);

    printf("Processed!\n");
}

void on_trackbar(int, void*) {
    float lowThreshold = lowThreshold_slider / 10.0f;
    float highThreshold = highThreshold_slider / 10.0f;
    double sigmaColor = sigmaColor_int; // Convert int to double
    double sigmaSpace = sigmaSpace_int; // Convert int to double

    Mat processedImage;
    mSrc.copyTo(processedImage);
    ApplyAdaptiveTileFilter(processedImage, magnitude_of_distortion, d, sigmaColor, sigmaSpace, lowThreshold, highThreshold);
    imshow("Output Image", processedImage);
    cv::waitKey(1);
}

int main1221() {
    mSrc = imread("3_Distorted Image Adaptive Grid.png", IMREAD_COLOR);
    if (mSrc.empty()) {
        cout << "[Error] Invalid Image!\n";
        return -1;
    }

    // Initialize your magnitude_of_distortion Mat here
    magnitude_of_distortion = imread("distortionMagnitude.png", IMREAD_GRAYSCALE);
    magnitude_of_distortion.convertTo(magnitude_of_distortion, CV_32FC1, 1.0 / 255.);
    if (magnitude_of_distortion.empty()) {
        cout << "[Error] Invalid Image!\n";
        return -1;
    }

    namedWindow("Processed Image", WINDOW_AUTOSIZE);
    createTrackbar("Diameter", "Processed Image", &d, 15, on_trackbar);
    createTrackbar("Sigma Color", "Processed Image", &sigmaColor_int, 200, on_trackbar);
    createTrackbar("Sigma Space", "Processed Image", &sigmaSpace_int, 200, on_trackbar);
    createTrackbar("Low Threshold", "Processed Image", &lowThreshold_slider, max_value, on_trackbar);
    createTrackbar("High Threshold", "Processed Image", &highThreshold_slider, max_value, on_trackbar);

    on_trackbar(0, 0); // Initial call to apply filter with initial parameters

    waitKey(0);
    return 0;
}
