#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <random> // Include for random number generation
#include "FisheyeEffect.h"
//#include "ImageUtils.h"

using namespace cv;
using namespace std;

// Function to compute distortion magnitude 
Mat computeDistortionMagnitude(const Mat& grid_x, const Mat& grid_y) {
    // Validate input matrices
    if (grid_x.type() != CV_32F || grid_y.type() != CV_32F) {
        std::cerr << "Both grid_x and grid_y must be of type CV_32F" << std::endl;
        return Mat();
    }
    if (grid_x.size() != grid_y.size()) {
        std::cerr << "grid_x and grid_y must have the same size" << std::endl;
        return Mat();
    }

    // Compute gradients for both channels (grids)
    Mat grad_x_dx, grad_y_dx, grad_x_dy, grad_y_dy;
    Sobel(grid_x, grad_x_dx, CV_32F, 1, 0, 3);
    Sobel(grid_x, grad_y_dx, CV_32F, 0, 1, 3);
    Sobel(grid_y, grad_x_dy, CV_32F, 1, 0, 3);
    Sobel(grid_y, grad_y_dy, CV_32F, 0, 1, 3);

    // Compute the magnitude of gradients
    Mat magnitude_dx, magnitude_dy;
    magnitude(grad_x_dx, grad_y_dx, magnitude_dx);
    magnitude(grad_x_dy, grad_y_dy, magnitude_dy);

    // Combine the magnitudes to get the total magnitude of distortion
    Mat total_magnitude = magnitude_dx + magnitude_dy; // Simple way to combine

    // Optionally, normalize the total magnitude for visualization
    Mat normalized_magnitude;
    normalize(total_magnitude, normalized_magnitude, 0, 1, NORM_MINMAX);

    return normalized_magnitude;
}

// Function for drawing a grid 
void DrawGrid(cv::Mat mSrc, const int Grid_X, const int Grid_Y) {
    int width = mSrc.size().width;
    int height = mSrc.size().height;

    const int cellwidth = width / Grid_X;
    const int cellheight = width / Grid_X;


    for (int i = 0; i < height; i += cellwidth)
        cv::line(mSrc, Point(0, i), Point(width, i), cv::Scalar(255, 0, 0), 2);

    for (int i = 0; i < width; i += cellheight)
        cv::line(mSrc, Point(i, 0), Point(i, height), cv::Scalar(255, 0, 0), 2);
}

// Function to display and save an image
void displayAndSaveImage(const Mat& image, const string& windowName) {
    imshow(windowName, image);

    // Construct the filename using the window name and ".png" extension
    string filename = windowName + ".png";
    imwrite(filename, image);
}

// Function to display and save an image
void SaveImage(const Mat& image, const string& windowName) {
    // Construct the filename using the window name and ".png" extension
    string filename = windowName + ".png";
    imwrite(filename, image);
}

void drawGridPoints(const vector<Point>& gridPoints, Mat& image, const Scalar& color, int radius, int thickness) {

    // Ensure the image is in a suitable format (like CV_8UC3)
    if (image.type() != CV_8UC3) {
        if (image.type() == CV_8UC1) {
            cvtColor(image, image, COLOR_GRAY2BGR);
        }
        else {
            // Handle other incompatible image types if needed
            std::cerr << "Error: drawGridPoints expects a CV_8UC3 or CV_8UC1 image." << std::endl;
            return;
        }
    }

    for (const Point& pt : gridPoints) {
        circle(image, pt, radius, color, thickness);
    }
}

void drawGridPoints(const std::map<cv::Point, cv::Point2f, PointCompare>& GDC_Adaptive_Grid_Points, Mat& image, const Scalar& color, int radius, int thickness) {

    // Ensure the image is in a suitable format (like CV_8UC3)
    if (image.type() != CV_8UC3) {
        if (image.type() == CV_8UC1) {
            cvtColor(image, image, COLOR_GRAY2BGR);
        }
        else {
            // Handle other incompatible image types if needed
            std::cerr << "Error: drawGridPoints expects a CV_8UC3 or CV_8UC1 image." << std::endl;
            return;
        }
    }

    // Populate gridPoints and gridPointsMap
    for (const auto& pair : GDC_Adaptive_Grid_Points) {
        const cv::Point & pt = pair.first; // Assumes first part of the pair is the grid position
        circle(image, pt, radius, color, thickness);
    }
}

void Generate_FixedGrid(const Mat& magnitude_of_distortion, vector<Point>& GDC_Grid_Points, const int Grid_x, const int Grid_y) {
    // Input magnitude_of_distortion should be in Range 0-1
    Mat image = magnitude_of_distortion.clone();
    image.convertTo(image, CV_8U, 255); // Scale to 0-255 range
    cvtColor(image, image, COLOR_GRAY2BGR);

    // Step 1: Calculate cell dimensions
    float cellWidth  = (float)image.cols /  (float)(Grid_x-1);
    float cellHeight = (float)image.rows /  (float)(Grid_y-1);

    // Step 2: Compute and store only the original grid points 
    GDC_Grid_Points.clear();
    for (int i = 0; i < Grid_x; i++) {
        for (int j = 0; j < Grid_y; j++) {
            int x = i * cellWidth; // Left Top of cell
            int y = j * cellHeight;

            // Draw marker for the original position 
            circle(image, Point(x, y), 1, Scalar(255, 0, 0), 2);
            GDC_Grid_Points.push_back(Point(x, y)); // Store the original point
        }
    }

    SaveImage(image, "3_Fixed Grid Points");
}

void segmentDistortionMap(const Mat& magnitude_of_distortion, Mat& outputMask, double lowThreshold, double highThreshold) {
    outputMask = Mat::zeros(magnitude_of_distortion.size(), CV_8UC1); // Initialize segmentation mask

    // Simple Thresholding 
    Mat lowMask, mediumMask, highMask;
    inRange(magnitude_of_distortion, 0, lowThreshold, lowMask);
    inRange(magnitude_of_distortion, lowThreshold, highThreshold, mediumMask);
    inRange(magnitude_of_distortion, highThreshold, 1.0, highMask);
    
    // Assign values to distinguish segments in the output mask
    outputMask.setTo(0, lowMask);
    outputMask.setTo(128, mediumMask);
    outputMask.setTo(255, highMask);
}

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

void writeCSV(string filename, cv::Mat m)
{
    std::ofstream myfile;
    myfile.open(filename.c_str());
    myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
    myfile.close();
}

int main() {
    
    Size ImageSize(1280, 720);
    int Grid_Size = 35, Grid_Size_FC = 35;

    //Grid_Size_FC = Grid_Size;
    
    Point Grid(Grid_Size, Grid_Size), Grid_FG(Grid_Size_FC, Grid_Size_FC);

    Mat srcImage(ImageSize, CV_8UC3, Scalar::all(255));
    DrawGrid(srcImage, 35, 35);
    
    int interpolation= INTER_LANCZOS4;
    int borderMode = BORDER_REFLECT;

    const double distStrength = 1.5;
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
#if 0

    /*std::map<cv::Point, cv::Point2f, PointCompare> GDC_mFixed_Grid_Points;

    Generate_FixedGridMap(ImageSize, GDC_mFixed_Grid_Points, Grid_FG.x, Grid_FG.y);

    Test_FindNearestPointsinFixedGridMap2x2(ImageSize, Grid_FG, GDC_mFixed_Grid_Points);*/

    /*std::map<cv::Point, cv::Point2f, PointCompare> GDC_Adaptive_Grid_Points;
    Generate_AdaptiveGridMap(distortionMagnitude, GDC_Adaptive_Grid_Points, Grid.x, Grid.y, LowThreshold);

    Test_FindNearestPointsinAdaptiveGridMap2x2(ImageSize, Grid, GDC_Adaptive_Grid_Points);*/

    std::map<cv::Point, cv::Point2f, PointCompare> GDC_Adaptive_Grid_Points;
    Generate_AdaptiveGridMap(distortionMagnitude, GDC_Adaptive_Grid_Points, Grid.x, Grid.y,LowThreshold);

    Test_FindNearestPointsinAdaptiveGridMap4x4(ImageSize, Grid, GDC_Adaptive_Grid_Points);

    waitKey(0);
    return 0;
#endif

    // Display and save the images
    //SaveImage(srcImage, "0_Source Image");
    SaveImage(distortedImage_GT, "1_Distorted Image");
    //imwrite("2_Magnitude of Distortion.png", distortionMagnitude * 255);

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
    waitKey(0);

    return 0;
}