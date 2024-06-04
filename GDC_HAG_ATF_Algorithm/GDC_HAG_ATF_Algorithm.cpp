#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <fstream>
#include <random> // Include for random number generation
#include "FisheyeEffect.h"
//#include "ImageUtils.h"

using namespace cv;
using namespace std;

// Helper function to find the most frequent segment value in a region
int findMostFrequentValue(const Mat& segmentedRegion) {
    std::map<int, int> segmentCounts;
    for (int y = 0; y < segmentedRegion.rows; ++y) {
        for (int x = 0; x < segmentedRegion.cols; ++x) {
            segmentCounts[segmentedRegion.at<uchar>(y, x)]++;
        }
    }
    // Find the segment with the maximum count (customize if needed)
    return std::max_element(segmentCounts.begin(), segmentCounts.end(),
        [](const auto& p1, const auto& p2) { return p1.second < p2.second; })->first;
}

static std::vector<cv::Point> addPointIfNotPresent(std::vector<cv::Point>& GDC_Adaptive_Grid_Points, cv::Point newPoint) {
    // Check if the newPoint is already present in the vector
    auto it = std::find(GDC_Adaptive_Grid_Points.begin(), GDC_Adaptive_Grid_Points.end(), newPoint);

    // If the point is not present
    if (it == GDC_Adaptive_Grid_Points.end()) {
        // Add the new point to the vector
        GDC_Adaptive_Grid_Points.push_back(newPoint);
    }

    // Return the updated vector
    return GDC_Adaptive_Grid_Points;
}

void Generate_AdaptiveGrid(const Mat& magnitude_of_distortion, vector<Point>& GDC_Adaptive_Grid_Points, const int Grid_x, const int Grid_y, const float LowThreshold) {
    Mat normalized_magnitude = magnitude_of_distortion.clone();
    normalized_magnitude.convertTo(normalized_magnitude, CV_8U, 255);
    cvtColor(normalized_magnitude, normalized_magnitude, COLOR_GRAY2BGR);


    const Scalar Blue(255, 0, 0), Yellow(0, 255, 255), Green(0, 255, 0), Red(0, 0, 255);
    Mat Segmented_DistortionMap;
    segmentDistortionMap(magnitude_of_distortion, Segmented_DistortionMap, LowThreshold, 0.98);

    const int imageWidth = magnitude_of_distortion.cols;
    const int imageHeight = magnitude_of_distortion.rows;

    const float baseCellWidth  = (float)imageWidth /  (float)(Grid_x - 1);
    const float baseCellHeight = (float)imageHeight / (float)(Grid_y - 1);

    // Clear any existing points and reserve space for efficiency
    GDC_Adaptive_Grid_Points.clear();

    for (int i = 0; i < Grid_x; i++) {
        for (int j = 0; j < Grid_y; j++) {
            const int x = i * baseCellWidth;
            const int y = j * baseCellHeight;

            // These are the fixed grid points, directly generated and stored
            circle(normalized_magnitude, Point(x, y), 1, Scalar(255, 0, 0), 2);
            GDC_Adaptive_Grid_Points.push_back(Point(x, y));
        }
    }

    for (int i = 0; i < Grid_x; ++i) {
        for (int j = 0; j < Grid_y; ++j) {
            const int x = i * baseCellWidth;
            const int y = j * baseCellHeight;

            // Ensure cell boundaries are within image limits
            const float cellWidth  = std::min(baseCellWidth,  (float)imageWidth - x);
            const float cellHeight = std::min(baseCellHeight, (float)imageHeight - y);

            if (cellWidth <= 0 || cellHeight <= 0) continue; // Skip invalid cells

            const cv::Rect cellRect(x, y, cellWidth, cellHeight);
            const cv::Mat cellRegion = Segmented_DistortionMap(cellRect);
            const int predominantSegment = findMostFrequentValue(cellRegion);

            const cv::Point newPoint(x + (cellWidth / 2.0), y);
            const cv::Point newPoint2(x + (cellWidth / 2.0), y + cellHeight);

            if (predominantSegment >= 128) {
                //GDC_Adaptive_Grid_Points.push_back(newPoint);
                GDC_Adaptive_Grid_Points=addPointIfNotPresent(GDC_Adaptive_Grid_Points, newPoint);

                if (j == Grid_y - 2) {
                    cv::circle(normalized_magnitude, newPoint2, 2, Red, 2);
                    GDC_Adaptive_Grid_Points = addPointIfNotPresent(GDC_Adaptive_Grid_Points, newPoint2);
                }

                if (predominantSegment == 255) { // High Distortion
                    cv::circle(normalized_magnitude, newPoint, 1, Green, 2);
                }
                else {
                    cv::circle(normalized_magnitude, newPoint, 1, Yellow, 2);
                }
            }
        }
    }
    
    SaveImage(normalized_magnitude, "4_Adaptive Grid Points");
}

void Test_FindNearestPointsinFixedGridMap2x2(const cv::Size& ImageSize, const cv::Point GridSize, const std::map<cv::Point, cv::Point2f, PointCompare> GDC_Fixed_Grid_Points) {

    Mat srcImage(ImageSize, CV_8UC3, Scalar::all(255));

    drawGridPoints(GDC_Fixed_Grid_Points, srcImage, Scalar(255, 0, 0), 1, 2);

    std::cout << "Test_FindNearestPointsinFixedGridMap2x2\n";

    cv::Point PointSrc; // cv::Point Index
    cv::Point CornerIdx, GridIndex;
    RectPoints GridRectMap;

    cv::Point2f CorrectedPoint; // Grid Index


    printf("Full Image Simulation!\n");

    cv::RNG Rndm(2342);

    Mat Image = srcImage.clone();

    // Interpolate and fill the missing pixels
    for (int y = 0; y < ImageSize.height; y += 100) {
        for (int x = 0; x < ImageSize.width; x += 100) {
            PointSrc = cv::Point(x, y);
            // Check if PointSrc is a grid point
            if (1/*y > (Border.y)*/) {
                Image = srcImage.clone();
                if (!findGridPointValue(GDC_Fixed_Grid_Points, PointSrc, CorrectedPoint)) {

                    if (getTileRectMapFixed(PointSrc, ImageSize, GridSize, GDC_Fixed_Grid_Points, GridRectMap)) {

                        Scalar Color(Rndm.uniform(0, 255), Rndm.uniform(0, 255), Rndm.uniform(0, 255));

                        // bilinear interpolation logic
                        rectangle(Image, Rect(GridRectMap.cornersPoint[0][0], GridRectMap.cornersPoint[1][1]), Color, 2);

                        Scalar Color1(Rndm.uniform(0, 255), Rndm.uniform(0, 255), Rndm.uniform(0, 255));

                        for (size_t i = 0; i < 2; i++)
                        {
                            for (size_t j = 0; j < 2; j++)
                            {
                                circle(Image, GridRectMap.cornersPoint[i][j], 2, Color1, 2);
                            }
                        }

                        circle(Image, PointSrc, 2, Scalar(0, 0, 255), 2);
                        cv::imshow("Error Case Adaptive Grid", Image);
                        waitKey(1);

                    }
                    else
                    {
                        circle(Image, PointSrc, 2, Scalar(0, 0, 255), 2);
                        cv::imshow("Error Case Adaptive Grid", Image);
                        waitKey(0);
                    }
                }
                else // Corner Grid Point
                {
                    circle(Image, PointSrc, 2, Scalar(255, 0, 255), 3);
                    cv::imshow("Error Case Adaptive Grid", Image);
                    waitKey(1);
                }
            }
        }
    }
    cv::imshow("Error Case Adaptive Grid", Image);

    printf("Completed Full Image Simulation!\n");
    waitKey(0);
}


void Test_FindNearestPointsinAdaptiveGridMap2x2(const cv::Size& ImageSize, const cv::Point GridSize, const std::map<cv::Point, cv::Point2f, PointCompare> GDC_Adaptive_Grid_Points) {

    Mat srcImage(ImageSize, CV_8UC3, Scalar::all(255));

    const int cellWidth = ImageSize.width / (GridSize.x - 1);
    const int cellHeight = ImageSize.height / (GridSize.y - 1);

    cv::Point CellSize(cellWidth, cellHeight);

    drawGridPoints(GDC_Adaptive_Grid_Points, srcImage, Scalar(255, 0, 0), 1, 2);

    std::cout << "Test_FindNearestPointsinAdaptiveGrid\n";

    cv::Point PointSrc; // cv::Point Index
    cv::Point CornerIdx;
    RectPoints GridRectMap;

    cv::Point2f CorrectedPoint; // Grid Index


    printf("Full Image Simulation!\n");

    cv::RNG Rndm(2342);

    cv::Point Border((GridSize.x - 1) * cellWidth, (GridSize.y - 1) * cellHeight);

    Mat Image = srcImage.clone();

    // Interpolate and fill the missing pixels
    for (int y = 0; y < ImageSize.height; y += 5) {
        for (int x = 0; x < ImageSize.width; x += 5) {
            PointSrc = cv::Point(x, y);
            // Check if PointSrc is a grid point
            if (1/*y > (Border.y)*/) {
                Image = srcImage.clone();
                if (!findGridPointValue(GDC_Adaptive_Grid_Points, PointSrc, CorrectedPoint)) {

                    if (getTileRectMap(PointSrc, ImageSize, GridSize, GDC_Adaptive_Grid_Points, GridRectMap)) {

                        Scalar Color(Rndm.uniform(0, 255), Rndm.uniform(0, 255), Rndm.uniform(0, 255));

                        // bilinear interpolation logic
                        rectangle(Image, Rect(GridRectMap.cornersPoint[0][0], GridRectMap.cornersPoint[1][1]), Color, 2);

                        Scalar Color1(Rndm.uniform(0, 255), Rndm.uniform(0, 255), Rndm.uniform(0, 255));

                        for (size_t i = 0; i < 2; i++)
                        {
                            for (size_t j = 0; j < 2; j++)
                            {
                                circle(Image, GridRectMap.cornersPoint[i][j], 2, Color1, 2);
                            }
                        }

                        circle(Image, PointSrc, 2, Scalar(0, 0, 255), 2);
                        cv::imshow("Error Case Adaptive Grid", Image);
                        waitKey();

                    }
                    else
                    {
                        circle(Image, PointSrc, 2, Scalar(0, 0, 255), 2);
                        cv::imshow("Error Case Adaptive Grid", Image);
                        waitKey();
                    }
                }
                else // Corner Grid Point
                {
                    circle(Image, PointSrc, 2, Scalar(255, 0, 255), 3);
                    cv::imshow("Error Case Adaptive Grid", Image);
                    waitKey();
                }
            }
        }
    }
    cv::imshow("Error Case Adaptive Grid", Image);

    printf("Completed Full Image Simulation!\n");
    waitKey(0);
}


void Test_FindNearestPointsinAdaptiveGridMap4x4(const cv::Size& ImageSize, const cv::Point GridSize, const std::map<cv::Point, cv::Point2f, PointCompare> GDC_Adaptive_Grid_Points) {

    Mat srcImage(ImageSize, CV_8UC3, Scalar::all(255));

    drawGridPoints(GDC_Adaptive_Grid_Points, srcImage, Scalar(255, 0, 0), 1, 2);

    std::cout << "Test_FindNearestPointsinAdaptiveGridMap4x4\n";

    cv::Point PointSrc; // cv::Point Index
    RectPoints GridRectMap, GridRectMapAdaptive;

    cv::Point2f CorrectedPoint; // Grid Index
    

    printf("Performing Full Image Interpolation!\n");

    cv::RNG Rndm(2342);

    Mat Image = srcImage.clone();

    // Interpolate and fill the missing pixels
    for (int y = 0; y < ImageSize.height; y+=50) {
        for (int x = 0; x < ImageSize.width; x+=50) {
            PointSrc = cv::Point(x, y);
            // Check if PointSrc is a grid point
            Image = srcImage.clone();

            if (y > 5 && x > 5) {
                if (!findGridPointValue(GDC_Adaptive_Grid_Points, PointSrc, CorrectedPoint)) {

                    if (getTileRectMap4x4(PointSrc, ImageSize, GridSize, GDC_Adaptive_Grid_Points, GridRectMap, GridRectMapAdaptive)) {
                        
                        Scalar Color(Rndm.uniform(0, 255), Rndm.uniform(0, 255), Rndm.uniform(0, 255));

                        // bilinear interpolation logic
                        rectangle(Image, Rect(GridRectMap.cornersPoint[0][0], GridRectMap.cornersPoint[3][3]), Color, 2);

                        Scalar Color1(255, 0, 0);
                        
                        for (size_t i = 0; i < 4; i++)
                        {
                            for (size_t j = 0; j < 4; j++)
                            {
                                circle(Image, GridRectMap.cornersPoint[i][j], 2, Scalar(255,0,255-(i*50)), 2);
                            }
                        }
                        for (size_t i = 0; i < 4; i++)
                        {
                            for (size_t j = 0; j < 4; j++)
                            {
                                if (GridRectMapAdaptive.cornersIdx[i][j])
                                    circle(Image, GridRectMapAdaptive.cornersPoint[i][j], 2, Scalar(0, 255, 0) - cv::Scalar(0, i * 20, 0), 2);
                            }
                        }

                        circle(Image, PointSrc, 2, Scalar(0, 0, 255), 2);
                        cv::imshow("Error Case Adaptive Grid", Image);
                        waitKey(0);

                    }
                }
                else // Corner Grid Point
                {
                    circle(Image, PointSrc, 2, Scalar(255, 0, 255), 3);
                    cv::imshow("Error Case Adaptive Grid", Image);
                    waitKey(1);
                }
            }
        }
    }
    cv::imshow("Error Case Adaptive Grid", Image);
}

// Function to determine perfect grid dimensions for an image
std::vector<cv::Size> findPerfectGrids(const cv::Size & imageSize) {
    std::vector<cv::Size> perfectGrids;

    int width = imageSize.width;
    int height = imageSize.height;

    // Find the aspect ratio
    double aspectRatio = (double)width / height;

    // Iterate through potential grid dimensions
    for (int gridX = 7; gridX <= 100; ++gridX) {
        for (int gridY = 7; gridY <= 100; ++gridY) {
            // Check if division is perfect for both width and height
            if (width % gridX == 0 && height % gridY == 0) {
                // Check if the aspect ratio is maintained 
                double gridAspectRatio = (double)gridX / gridY;
                if (std::abs(gridAspectRatio - aspectRatio) < 0.91) { // Allow slight tolerance 
                    perfectGrids.push_back(cv::Size(gridX, gridY));
                }
            }
        }
    }

    return perfectGrids;
}


void ApplyAdaptiveTileFilter(cv::Mat& mSrc, const cv::Mat& magnitude_of_distortion, const float lowThreshold, const float highThreshold) {
    printf("Performing ApplyAdaptiveTileFilter\n");

    // Thresholding to create medium and high distortion masks
    cv::Mat mediumMask, highMask;
    cv::inRange(magnitude_of_distortion, cv::Scalar(lowThreshold), cv::Scalar(highThreshold), mediumMask);
    cv::inRange(magnitude_of_distortion, cv::Scalar(highThreshold), cv::Scalar(1.0), highMask);

    // Optional: Apply Gaussian blur to masks for smoother transitions
    int blurSize = 3; // Adjust for smoothing
    cv::GaussianBlur(mediumMask, mediumMask, cv::Size(blurSize, blurSize), 0);
    cv::GaussianBlur(highMask, highMask, cv::Size(blurSize, blurSize), 0);

    // Apply bilateral filter with optimized parameters
    cv::Mat tempMedium, tempHigh;
    cv::ximgproc::jointBilateralFilter(mSrc, mSrc, tempMedium, 1, 50, 50);
    cv::ximgproc::jointBilateralFilter(mSrc, mSrc, tempHigh, 3, 75, 75);

    // Apply guided filtering with adjusted parameters
    int r_medium = 2;
    int r_high = 4;
    double eps_medium = 0.1 * 0.1 * 255 * 255;
    double eps_high = 0.2 * 0.2 * 255 * 255;

    cv::ximgproc::guidedFilter(tempMedium, mSrc, tempMedium, r_medium, eps_medium);
    cv::ximgproc::guidedFilter(tempHigh, mSrc, tempHigh, r_high, eps_high);

    // Create combined images with Poisson blending
    cv::Mat blendedMedium, blendedHigh;
    cv::Point center(mSrc.cols / 2, mSrc.rows / 2); // Center for Poisson blending

    cv::seamlessClone(tempMedium, mSrc, mediumMask, center, blendedMedium, cv::NORMAL_CLONE); // Blend medium-distortion regions
    cv::seamlessClone(tempHigh, mSrc, highMask, center, blendedHigh, cv::NORMAL_CLONE);       // Blend high-distortion regions

    // Combine results based on the masks
    cv::Mat result = mSrc.clone();
    blendedMedium.copyTo(result, mediumMask); // Apply medium regions
    blendedHigh.copyTo(result, highMask);     // Apply high regions

    result.copyTo(mSrc); // Return the result to the source image

    printf("[Success] Completed ApplyAdaptiveTileFilter\n");
}



int main() {
    
    Size ImageSize(1280, 720);
    int Grid_Size = 30, Grid_Size_FC = 33;

    //Grid_Size_FC = Grid_Size;
    
    Point Grid(Grid_Size, Grid_Size), Grid_FG(Grid_Size_FC, Grid_Size_FC);

    Mat srcImage(ImageSize, CV_8UC3, Scalar::all(255));
    DrawGrid(srcImage, 35, 35);

    /*srcImage = imread("C:/Users/balaj/Pictures/Camera_2172x1448.jpg", 1);
    ImageSize = srcImage.size();*/
    
    int interpolation= INTER_LANCZOS4;
    int borderMode = BORDER_REFLECT;

    const double distStrength = 2.75;
    const float LowThreshold = 0.85;

    double  rms_error_FixedGrid, rms_error_AdaptiveGrid;

    FisheyeEffect distorter(ImageSize);

    distorter.generateDistortionMaps(distStrength);
    // Apply distortion
    Mat distortedImage_GT;
    Mat distortedImage_FixedGrid;
    Mat distortedImage_AdaptiveGrid;
    

    // Compute distortion magnitude
    Mat Map_x, Map_y;
    distorter.getDistortionMaps(Map_x, Map_y);
    Mat distortionMagnitude = computeDistortionMagnitude(Map_x, Map_y);
    
    cv::remap(srcImage, distortedImage_GT, Map_x, Map_y, interpolation, borderMode);

    writeCSV("GT_Mapx.csv", Map_x);
    writeCSV("GT_Mapy.csv", Map_y);

    vector<Point> GDC_Fixed_Grid_Points;

    Generate_FixedGrid(distortionMagnitude, GDC_Fixed_Grid_Points,Grid_FG.x, Grid_FG.y);
    
    vector<Point>GDC_Adaptive_Grid_Point;
    Generate_AdaptiveGrid(distortionMagnitude, GDC_Adaptive_Grid_Point, Grid.x, Grid.y, LowThreshold);

    int Total_points_FixedGrid      = GDC_Fixed_Grid_Points.size();
    int Total_points_VariableGrid   = GDC_Adaptive_Grid_Point.size();

    int Points_Diff = Total_points_FixedGrid - Total_points_VariableGrid;

    //Calculate the percentage of points saved as a floating - point number
    double Saved_Percentage = static_cast<double>(Points_Diff) / Total_points_FixedGrid * 100;

    std::cout << "Total No of Points \t Fixed Grid : " << Total_points_FixedGrid
        << " \t Variable Grid : " << Total_points_VariableGrid
        << " : Saved : " << Points_Diff << " Points (" << Saved_Percentage << "%)" << std::endl;

    // Display and save the images
    SaveImage(distortedImage_GT, "1_Distorted Image");

    SaveImage(distortionMagnitude*255, "distortionMagnitude");

#if 1

    Mat Map_x_FG, Map_y_FG;

    distorter.computeDistortionMapsfromFixedGrid(Grid_FG, distStrength);
    //distorter.computeDistortionMapsfromFixedGridMap(ImageSize,Grid_FG, distStrength);
        
    distorter.getDistortionMaps(Map_x_FG, Map_y_FG);

    cv::remap(srcImage, distortedImage_FixedGrid, Map_x_FG, Map_y_FG, interpolation, borderMode);

    rms_error_FixedGrid = distorter.compareDistortionMaps(Map_x, Map_y, Map_x_FG, Map_y_FG, "GT vs Fixed Grid");

    SaveImage(distortedImage_FixedGrid, "2_Distorted Image Fixed Grid");

    distorter.computeDistortionMapsfromAdaptiveGridMap(Grid, distStrength, LowThreshold);

    Mat Map_x_AG, Map_y_AG;
    distorter.getDistortionMaps(Map_x_AG, Map_y_AG);

    writeCSV("GT_Mapx_AG.csv", Map_x_AG);
    writeCSV("GT_Mapy_AG.csv", Map_y_AG);

    cv::remap(srcImage, distortedImage_AdaptiveGrid, Map_x_AG, Map_y_AG, interpolation, borderMode);

    rms_error_AdaptiveGrid = distorter.compareDistortionMaps(Map_x, Map_y, Map_x_AG, Map_y_AG, "GT vs Adaptive Grid");

    displayAndSaveImage(distortedImage_AdaptiveGrid, "3_Distorted Image Adaptive Grid");

    // Calculate and print results for Fixed Grid
    double psnr_fixed = getPSNR(distortedImage_GT, distortedImage_FixedGrid);
    double psnr_adaptive = getPSNR(distortedImage_GT, distortedImage_AdaptiveGrid);

    cout << "------------------------------------------------------------------------------------\n";
    cout << "                              Results Summary             \n";
    cout << "------------------------------------------------------------------------------------\n";
    cout << setw(40) << left << "Metric" << setw(15) <<  "Fixed Grid" << setw(15) <<  "Adaptive Grid\n";
    cout << "------------------------------------------------------------------------------------\n";
    cout << setw(40) << left << "RMS Error of Distortion Map" << setw(15) <<  rms_error_FixedGrid << setw(15) <<  rms_error_AdaptiveGrid << "\n";
    cout << setw(40) << left << "PSNR of Remapped Image" << setw(15) <<  psnr_fixed << setw(15) <<  psnr_adaptive << "\n";
    cout << "------------------------------------------------------------------------------------\n";

    // Compare the RMS error values
    // Calculate the percentage improvement from Fixed Grid to Adaptive Grid
    if (rms_error_FixedGrid > rms_error_AdaptiveGrid) {
        double improvement = ((rms_error_FixedGrid - rms_error_AdaptiveGrid) / rms_error_FixedGrid) * 100.0;
        cout << "[Success] Adaptive Grid Estimation shows a " << improvement << "% improvement over Fixed Grid Estimation. (RMS Error)\n";
    }
    else if (rms_error_FixedGrid < rms_error_AdaptiveGrid) {
        double improvement = ((rms_error_AdaptiveGrid - rms_error_FixedGrid) / rms_error_AdaptiveGrid) * 100.0;
        cout << "[Failed] Fixed Grid Estimation shows a " << improvement << "% improvement over Adaptive Grid Estimation.(Lower RMS Error)\n";
    }
    else {
        cout << "No improvement, both methods have the same RMS error.\n";
    }

    // Compare the PSNR values 
    if (psnr_adaptive > psnr_fixed) {
        double improvement = ((psnr_adaptive - psnr_fixed) / psnr_fixed) * 100.0;
        cout << "[Success] Adaptive Grid Estimation shows a " << improvement << "% improvement over Fixed Grid Estimation (Better PSNR).\n";
    }
    else if (psnr_fixed > psnr_adaptive) {
        double improvement = ((psnr_fixed - psnr_adaptive) / psnr_adaptive) * 100.0;
        cout << "[Failed] Fixed Grid Estimation shows a " << improvement << "% improvement over Adaptive Grid Estimation (Higher PSNR).\n";
    }
    else {
        cout << "No improvement in PSNR, both methods have similar results.\n";
    }

#endif


    // Adaptive Tile Filtering (ATF)

    //Mat mSrc = distortedImage_AdaptiveGrid.clone();

    Mat mSrc = imread("3_Distorted Image Adaptive Grid.png", 1);

    if (mSrc.empty()) {
        std::cout << "[Error] Invalid Image!\n";
        return 0;
    }

    imshow("Input Image", mSrc);

    ApplyAdaptiveTileFilter(mSrc, distortionMagnitude, LowThreshold, 0.98);

    displayAndSaveImage(mSrc, "ATF_Image");

    waitKey(0);

    return 0;
}

