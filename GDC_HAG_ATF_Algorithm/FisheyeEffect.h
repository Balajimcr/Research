#ifndef FISHEYE_EFFECT_H
#define FISHEYE_EFFECT_H

#include <opencv2/opencv.hpp>

// Struct to represent a 2x2 array of 4 rectangle corners 
struct RectPoints {
    cv::Point   cornersPoint[4][4];
    cv::Point2f cornersMap[4][4];
    int cornersIdx[4][4];
    // For Billinear
    // Top-Left     -  00
    // Top-Right    -  01
    // Bottom-Left  -  10
    // Bottom-Right -  11
};


struct PointCompare {
    bool operator()(const cv::Point& a, const cv::Point& b) const {
        if (a.x < b.x) return true;
        if (a.x > b.x) return false;
        return a.y < b.y;
    }
};

void Generate_FixedGridMap(cv::Size ImageSize, std::map<cv::Point, cv::Point2f, PointCompare>& GDC_Grid_Points, const int Grid_x, const int Grid_y);
void Generate_AdaptiveGridMap(const cv::Mat& magnitude_of_distortion, std::map<cv::Point, cv::Point2f, PointCompare>& GDC_Adaptive_Grid_Points, const int Grid_x, const int Grid_y, const float LowThreshold);

bool findGridPointValue(const std::map<cv::Point, cv::Point2f, PointCompare>& gridPoints, const cv::Point& searchPoint, cv::Point2f& outCorrectedPoint);
bool getTileRectMap(const cv::Point& pt, const cv::Size& imageSize, const cv::Point& gridSize, const std::map<cv::Point, cv::Point2f, PointCompare>& GDC_Adaptive_Grid_Points, RectPoints& cellRect);
bool getTileRectMap4x4(const cv::Point& pt, const cv::Size& imageSize, const cv::Point& gridSize, const std::map<cv::Point, cv::Point2f, PointCompare>& GDC_Adaptive_Grid_Points, RectPoints& cellRect);
bool getTileRectMapFixed(const cv::Point& pt, const cv::Size& imageSize, const cv::Point& gridSize, std::map<cv::Point, cv::Point2f, PointCompare> GDC_Fixed_Grid_Points, RectPoints& cellRect);
// ![get-psnr]
static double getPSNR(const cv::Mat& I1, const cv::Mat& I2)
{
    using namespace cv;
    cv::Mat s1;
    cv::absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    cv::Scalar s = sum(s1);        // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}
// ![get-psnr]

class FisheyeEffect {
public:

    // Interpolation methods
    enum class InterpolationMethod {
        NEAREST_NEIGHBOR,
        BILINEAR,
        BICUBIC
    };

    // Constructor 
    FisheyeEffect(const cv::Size& imageSize);

    // Destructor
    ~FisheyeEffect();

    // Apply the fisheye effect to an image
    void applyDistortion(const cv::Mat& srcImage, cv::Mat& dstImage, int interpolation, int borderMode = cv::BORDER_CONSTANT);

    // Update distortion parameters
    void updateParameters();

    // Accessors for parameters (add as needed)
    double getFx() const { return fx; }
    void setFx(double newFx) { fx = newFx; updateParameters(); }
    // ... Similar accessors for other parameters (fy, cx, cy, k1, k2, p1, p2)

    // Generates distortion maps
    void generateDistortionMaps(double distStrength);
    void computeDistortionMapsfromFixedGrid(cv::Point Grid, const double distStrength);
    void generateDistortionMapsfromFixedGrid(
        const cv::Size& imageSize,
        const cv::Point& gridSize,
        const double distStrength,
        const std::vector<std::vector<cv::Point>>& GDC_Fixed_Grid_Points,
        std::vector<std::vector<cv::Point2f>>& GDC_Fixed_Grid_Map,
        cv::Mat& mapX,
        cv::Mat& mapY,
        InterpolationMethod method = InterpolationMethod::BILINEAR);

    void generateDistortionMapsfromAdaptiveGridMap(
        const cv::Size& imageSize,
        const cv::Point& gridSize,
        const double distStrength,
        std::map<cv::Point, cv::Point2f, PointCompare>& GDC_Adaptive_Grid_Points,
        cv::Mat& mapX,
        cv::Mat& mapY,
        InterpolationMethod method= InterpolationMethod::BILINEAR
    );

    void computeDistortionMapsfromAdaptiveGridMap(cv::Point GridSize, const double distStrength, const float LowThreshold);

    void updateDistorsionMap();
    void testFunction(cv::Mat& image, const std::vector<std::vector<cv::Point2f>>& GDC_Fixed_Grid_Map);

    void computeDistortionMapfromFixedGrid(cv::Point Grid);
    void Generate_FixedGrid(cv::Size ImageSize, std::vector<std::vector<cv::Point>>& GDC_Grid_Points, const int Grid_x, const int Grid_y);
    void generateDistortionMapsfromFixedGridMap(
        const cv::Size& imageSize,
        const cv::Point& gridSize,
        const double distStrength,
        std::map<cv::Point, cv::Point2f, PointCompare>& GDC_Grid_Points,
        cv::Mat& mapX,
        cv::Mat& mapY,
        InterpolationMethod method = InterpolationMethod::BILINEAR
    );

    void computeDistortionMapsfromFixedGridMap(const cv::Size& imageSize, cv::Point GridSize, const double distStrength);
        // Function to return distortion maps
    void getDistortionMaps(cv::Mat& out_mapX, cv::Mat& out_mapY) const {
        out_mapX = mapX.clone();
        out_mapY = mapY.clone();
    }

    double compareDistortionMaps(
        const cv::Mat& mapX1,
        const cv::Mat& mapY1,
        const cv::Mat& mapX2,
        const cv::Mat& mapY2,
        const std::string Name
    );

private:
    // Image properties
    cv::Size imageSize;

    // Distortion parameters
    double fx, fy, cx, cy;
    double k1, k2, p1, p2;

    // OpenCV matrices
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Mat newCameraMatrix;
    cv::Mat mapX, mapY;
    bool bUseGeneratedMaps;
    // Helper functions
    void initDistortionParameters();
    void constructMatrices();
    
    void InitUndistortRectifyMap(
        const cv::Mat& cameraMatrix,
        const cv::Mat& distCoeffs,
        const cv::Mat& R,
        const cv::Mat& newCameraMatrix,
        const cv::Size& imageSize,
        int m1type,
        cv::Mat& mapX,
        cv::Mat& mapY);

    void InitUndistortRectifyGridMap(
        const cv::Mat& cameraMatrix,
        const cv::Mat& distCoeffs,
        const cv::Mat& R,
        const cv::Mat& newCameraMatrix,
        const cv::Size& imageSize,
        const std::vector<std::vector<cv::Point>>& GDC_Fixed_Grid_Points,
        std::vector<std::vector<cv::Point2f>>& GDC_Fixed_Grid_Map,
        cv::Mat& mapX,
        cv::Mat& mapY);
};

#endif // FISHEYE_EFFECT_H
