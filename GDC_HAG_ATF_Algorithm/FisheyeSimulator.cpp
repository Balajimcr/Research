#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


// Function to update the fisheye effect when parameters change
void updateEffect(int, void* data) {
    Mat* srcImage = (Mat*)data;

    // Get slider values
    int fx_val = getTrackbarPos("fx", "Controls");
    int fy_val = getTrackbarPos("fy", "Controls");
    int cx_val = getTrackbarPos("cx", "Controls");
    int cy_val = getTrackbarPos("cy", "Controls");

    double k1_val = getTrackbarPos("k1", "Controls") / 100.0;
    double k2_val = getTrackbarPos("k2", "Controls") / 100.0;
    double p1_val = getTrackbarPos("p1", "Controls") / 10000.0;
    double p2_val = getTrackbarPos("p2", "Controls") / 10000.0;

    // Update parameters (scaling from initial values)
    double fx = fx_val * 0.75;
    double fy = fy_val * 0.75;
    double cx = cx_val;
    double cy = cy_val;
    double k1 = k1_val - 0.2; // Adjust offset if needed
    double k2 = k2_val - 0.2; // Adjust offset if needed
    double p1 = p1_val;
    double p2 = p2_val;

    // Construct cameraMatrix and distCoeffs
    Mat cameraMatrix = (Mat_<double>(3, 3) << fx, 0, cx,
        0, fy, cy,
        0, 0, 1);
    Mat distCoeffs = (Mat_<double>(1, 5) << k1, k2, p1, p2, 0);

    // Print calculated parameters
    cout << "fx: " << fx << endl;
    cout << "fy: " << fy << endl;
    cout << "cx: " << cx << endl;
    cout << "cy: " << cy << endl;
    cout << "k1: " << k1 << endl;
    cout << "k2: " << k2 << endl;
    cout << "p1: " << p1 << endl;
    cout << "p2: " << p2 << endl;

    cout << "cameraMatrix" << cameraMatrix << "\n";
    cout << "distCoeffs" << distCoeffs << "\n";

    // Get new optimal camera matrix 
    Mat newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, srcImage->size(), 1, srcImage->size());

    // Apply distortion using remap()
    Mat mapX, mapY;
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), newCameraMatrix, srcImage->size(), CV_32FC1, mapX, mapY);
    Mat remappedImage;
    remap(*srcImage, remappedImage, mapX, mapY, INTER_LINEAR);

    imshow("Distorted (remap)", remappedImage);
}
// Function for drawing a grid 
static void DrawGrid(cv::Mat mSrc, const int Grid_X, const int Grid_Y) {
    int width = mSrc.size().width;
    int height = mSrc.size().height;

    for (int i = 0; i < height; i += Grid_X)
        cv::line(mSrc, Point(0, i), Point(width, i), cv::Scalar(255, 0, 0), 2);

    for (int i = 0; i < width; i += Grid_Y)
        cv::line(mSrc, Point(i, 0), Point(i, height), cv::Scalar(255, 0, 0), 2);
}

int main2() {
    
    Size ImageSize(1280, 720);
    Point Grid(35, 35);
    Point2f ptSrc(100, 200), ptDst;

    Mat srcImage(ImageSize, CV_8UC3, Scalar::all(255));
    DrawGrid(srcImage, Grid.x, Grid.y);
    circle(srcImage, ptSrc, 1, Scalar(0, 0, 255), 2); // Blue Low Distorsion

    // Initial Values
    int image_center_x = srcImage.cols / 2;
    int image_center_y = srcImage.rows / 2;

    // Create window
    namedWindow("Controls");
    // Create trackbars using NULL as the 'value' argument
    createTrackbar("fx", "Controls", NULL, srcImage.cols, updateEffect, &srcImage);
    createTrackbar("fy", "Controls", NULL, srcImage.rows, updateEffect, &srcImage);
    createTrackbar("cx", "Controls", NULL, srcImage.cols, updateEffect, &srcImage);
    createTrackbar("cy", "Controls", NULL, srcImage.rows, updateEffect, &srcImage);
    createTrackbar("k1", "Controls", NULL, 200, updateEffect, &srcImage);
    createTrackbar("k2", "Controls", NULL, 200, updateEffect, &srcImage);
    createTrackbar("p1", "Controls", NULL, 100, updateEffect, &srcImage);
    createTrackbar("p2", "Controls", NULL, 100, updateEffect, &srcImage);

    // Initial Values (Modify these for your desired starting distortion)
    int init_fx_val = srcImage.cols * 0.75;  // Initialize based on image size
    int init_fy_val = srcImage.rows * 0.75;
    int init_cx_val = srcImage.cols / 2;
    int init_cy_val = srcImage.rows / 2;
    int init_k1_val = 70;  // Corresponds to k1 = 0.0
    int init_k2_val = 10;
    int init_p1_val = 1;
    int init_p2_val = 2;

    // Set initial trackbar positions
    setTrackbarPos("fx", "Controls", init_fx_val);
    setTrackbarPos("fy", "Controls", init_fy_val);
    setTrackbarPos("cx", "Controls", init_cx_val);
    setTrackbarPos("cy", "Controls", init_cy_val);
    setTrackbarPos("k1", "Controls", init_k1_val);
    setTrackbarPos("k2", "Controls", init_k2_val);
    setTrackbarPos("p1", "Controls", init_p1_val);
    setTrackbarPos("p2", "Controls", init_p2_val);

    updateEffect(0, &srcImage); // Trigger initial update

    waitKey(0);
    return 0;
}
