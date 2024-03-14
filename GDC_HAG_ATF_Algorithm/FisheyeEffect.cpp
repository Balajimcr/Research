#include "FisheyeEffect.h"
#include <algorithm>

bool findGridPointValue(const std::map<cv::Point, cv::Point2f, PointCompare>& gridPoints, const cv::Point& searchPoint, cv::Point2f& outCorrectedPoint) {

    auto it = gridPoints.find(searchPoint);

    if (it != gridPoints.end()) {
        outCorrectedPoint = it->second;
        return true; // Found!
    }
    else {
        return false; // Not found
    }
}

// Constructor (updated)
FisheyeEffect::FisheyeEffect(const cv::Size& imageSize) :
    imageSize(imageSize)
{
    initDistortionParameters();
    constructMatrices();
    bUseGeneratedMaps = false;

}

// Destructor
FisheyeEffect::~FisheyeEffect() {
    // Release any resources if needed
}

// Apply fisheye effect
void FisheyeEffect::applyDistortion(const cv::Mat& srcImage, cv::Mat& dstImage, int interpolation, int borderMode) {
    // Check if the source image is the expected size
    CV_Assert(srcImage.size() == imageSize);

    if (!bUseGeneratedMaps)
    {
        // Update matrices if parameters have changed
        updateParameters();
    }
    // Apply distortion using remap()
    cv::remap(srcImage, dstImage, mapX, mapY, interpolation, borderMode);
}


// Update distortion parameters (Call this when any parameter is changed)
void FisheyeEffect::updateParameters() {
    constructMatrices();
}

// Initialize parameters
void FisheyeEffect::initDistortionParameters() {
    // Set initial values based on image size and your desired defaults
    fx = imageSize.width * 0.75;
    fy = imageSize.height * 0.75;
    cx = imageSize.width / 2;
    cy = imageSize.height / 2;
    k1 = 0.5;
    k2 = -0.1;
    p1 = 0.0001; 
    p2 = 0.0002;
}

// Construct matrices needed for distortion
void FisheyeEffect::constructMatrices() {
    cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    distCoeffs = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, 0);

    newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize);
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newCameraMatrix, imageSize, CV_32FC1, mapX, mapY);
}

void FisheyeEffect::updateDistorsionMap() {
    cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    distCoeffs = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, 0);

    newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize);
    InitUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newCameraMatrix, imageSize, CV_32FC1, mapX, mapY);
}

void FisheyeEffect::computeDistortionMapfromFixedGrid(cv::Point Grid) {
    cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    distCoeffs = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, 0);

    newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize);

    std::vector<std::vector<cv::Point>> GDC_Fixed_Grid_Points;
    std::vector<std::vector<cv::Point2f>> GDC_Fixed_Grid_Map;

    Generate_FixedGrid(imageSize, GDC_Fixed_Grid_Points, Grid.x, Grid.y);

    InitUndistortRectifyGridMap(cameraMatrix, distCoeffs, cv::Mat(), newCameraMatrix, imageSize, GDC_Fixed_Grid_Points, GDC_Fixed_Grid_Map, mapX, mapY);
}

void FisheyeEffect::Generate_FixedGrid(cv::Size ImageSize, std::vector<std::vector<cv::Point>>& GDC_Grid_Points, const int Grid_x, const int Grid_y) {
    // Step 1: Calculate cell dimensions
    float cellWidth = (float)ImageSize.width / (float)(Grid_x - 1);
    float cellHeight = (float)ImageSize.height / (float)(Grid_y - 1);

    GDC_Grid_Points.clear();
    GDC_Grid_Points.resize(Grid_x); // Create outer vector with 'Grid_x' empty rows 

    for (int i = 0; i < Grid_x; ++i) {
        for (int j = 0; j < Grid_y; ++j) {
            int x = i * cellWidth; 
            int y = j * cellHeight;

            GDC_Grid_Points[i].push_back(cv::Point(x, y)); // Add points to the i-th row
        }
    }

    printf("Fixed Grid   : Total No of Samples : %d , from %d x %d Grid\n",
        (int)Grid_x * Grid_y, // Total number of samples
        Grid_x, Grid_y // Dimensions of the original grid
    ); // Dimensions of the nearest square grid
}

void FisheyeEffect::InitUndistortRectifyGridMap(
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    const cv::Mat& R,
    const cv::Mat& newCameraMatrix,
    const cv::Size& imageSize,
    const std::vector<std::vector<cv::Point>>& GDC_Fixed_Grid_Points,
    std::vector<std::vector<cv::Point2f>>& GDC_Fixed_Grid_Map,
    cv::Mat& mapX,
    cv::Mat& mapY) {

    // Ensure the type is CV_32FC1 for mapX and mapY as specified
    mapX.create(imageSize, CV_32FC1);
    mapY.create(imageSize, CV_32FC1);

    // Extract intrinsic parameters
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    // Extract new camera intrinsic parameters
    double fx_new = newCameraMatrix.at<double>(0, 0);
    double fy_new = newCameraMatrix.at<double>(1, 1);
    double cx_new = newCameraMatrix.at<double>(0, 2);
    double cy_new = newCameraMatrix.at<double>(1, 2);

    // Extract distortion coefficients
    double k1 = distCoeffs.at<double>(0, 0);
    double k2 = distCoeffs.at<double>(0, 1);
    double p1 = distCoeffs.at<double>(0, 2);
    double p2 = distCoeffs.at<double>(0, 3);
    double k3 = distCoeffs.size().width > 4 ? distCoeffs.at<double>(0, 4) : 0; // Handle optional k3

    // Ensure GDC_Fixed_Grid_MapX/Y are the same size as the grid
    GDC_Fixed_Grid_Map.resize(GDC_Fixed_Grid_Points.size());

    // Iterate through the grid points
    for (int i = 0; i < GDC_Fixed_Grid_Points.size(); ++i) {
        for (int j = 0; j < GDC_Fixed_Grid_Points[i].size(); ++j) {

            const cv::Point& gridPoint = GDC_Fixed_Grid_Points[i][j];

            int x = gridPoint.x;
            int y = gridPoint.y;

            double x_mapped = (x - cx_new) / fx_new;
            double y_mapped = (y - cy_new) / fy_new;

            double r2 = x_mapped * x_mapped + y_mapped * y_mapped;
            double x_distorted = x_mapped * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) +
                2 * p1 * x_mapped * y_mapped + p2 * (r2 + 2 * x_mapped * x_mapped);
            double y_distorted = y_mapped * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) +
                p1 * (r2 + 2 * y_mapped * y_mapped) + 2 * p2 * x_mapped * y_mapped;

            // Store the result in your maps
            GDC_Fixed_Grid_Map[i].push_back(cv::Point2f(static_cast<float>(x_distorted * fx + cx), static_cast<float>(y_distorted * fy + cy)));
        }
    }
}


void FisheyeEffect::InitUndistortRectifyMap(
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    const cv::Mat& R,
    const cv::Mat& newCameraMatrix,
    const cv::Size& imageSize,
    int m1type,
    cv::Mat& mapX,
    cv::Mat& mapY) {

    // Ensure the type is CV_32FC1 for mapX and mapY as specified
    mapX.create(imageSize, CV_32FC1);
    mapY.create(imageSize, CV_32FC1);

    // Extract intrinsic parameters
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    // Extract new camera intrinsic parameters
    double fx_new = newCameraMatrix.at<double>(0, 0);
    double fy_new = newCameraMatrix.at<double>(1, 1);
    double cx_new = newCameraMatrix.at<double>(0, 2);
    double cy_new = newCameraMatrix.at<double>(1, 2);

    // Extract distortion coefficients
    double k1 = distCoeffs.at<double>(0, 0);
    double k2 = distCoeffs.at<double>(0, 1);
    double p1 = distCoeffs.at<double>(0, 2);
    double p2 = distCoeffs.at<double>(0, 3);
    double k3 = distCoeffs.size().width > 4 ? distCoeffs.at<double>(0, 4) : 0; // Handle optional k3

    for (int y = 0; y < imageSize.height; ++y) {
        for (int x = 0; x < imageSize.width; ++x) {
            double x_mapped = (x - cx_new) / fx_new;
            double y_mapped = (y - cy_new) / fy_new;

            double r2 = x_mapped * x_mapped + y_mapped * y_mapped;
            double x_distorted = x_mapped * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) +
                2 * p1 * x_mapped * y_mapped + p2 * (r2 + 2 * x_mapped * x_mapped);
            double y_distorted = y_mapped * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) +
                p1 * (r2 + 2 * y_mapped * y_mapped) + 2 * p2 * x_mapped * y_mapped;

            // Map back using the original camera matrix
            mapX.at<float>(y, x) = static_cast<float>(x_distorted * fx + cx);
            mapY.at<float>(y, x) = static_cast<float>(y_distorted * fy + cy);
        }
    }
}



// Generate distortion maps
void FisheyeEffect::generateDistortionMaps(double distStrength) {
    cv::Mat mapX, mapY;
    mapX.create(imageSize, CV_32FC1);
    mapY.create(imageSize, CV_32FC1);

    cv::Point2f center(imageSize.width / 2.0f, imageSize.height / 2.0f);

    for (int y = 0; y < imageSize.height; y++) {
        for (int x = 0; x < imageSize.width; x++) {
            float deltaX = (x - center.x) / center.x;
            float deltaY = (y - center.y) / center.y;
            float distance = (sqrt(deltaX * deltaX + deltaY * deltaY))/2;
            float distortion = 1.0f + distance * distStrength ;
            float newX = center.x + (deltaX * distortion * center.x);
            float newY = center.y + (deltaY * distortion * center.y);
            mapX.at<float>(y, x) = newX;
            mapY.at<float>(y, x) = newY;
        }
    }

    // Store the generated maps
    this->mapX = mapX;
    this->mapY = mapY;

    bUseGeneratedMaps = true;
}

std::vector<std::vector<cv::Point2f>> convertGridPoints(
    const std::vector<std::vector<cv::Point>>& GDC_Fixed_Grid_Points) {

    std::vector<std::vector<cv::Point2f>> GDC_Fixed_Grid_Points2f;

    // Iterate through each outer vector (representing rows)
    for (const auto& row : GDC_Fixed_Grid_Points) {
        std::vector<cv::Point2f> newRow;

        // Iterate through each point in the row
        for (const auto& point : row) {
            newRow.push_back(cv::Point2f(point.x, point.y)); // Convert to Point2f
        }

        GDC_Fixed_Grid_Points2f.push_back(newRow);
    }

    return GDC_Fixed_Grid_Points2f;
}

void FisheyeEffect::computeDistortionMapsfromFixedGridMap(const cv::Size& imageSize, cv::Point GridSize, const double distStrength) {

    std::map<cv::Point, cv::Point2f, PointCompare> GDC_Fixed_Grid_Points;

    Generate_FixedGridMap(imageSize, GDC_Fixed_Grid_Points, GridSize.x, GridSize.y);

    generateDistortionMapsfromFixedGridMap(imageSize, GridSize, distStrength, GDC_Fixed_Grid_Points, this->mapX, this->mapY);
}

void FisheyeEffect::computeDistortionMapsfromFixedGrid(cv::Point GridSize, const double distStrength) {

    std::vector<std::vector<cv::Point>> GDC_Fixed_Grid_Points;
    std::vector<std::vector<cv::Point2f>> GDC_Fixed_Grid_Map;

    Generate_FixedGrid(imageSize, GDC_Fixed_Grid_Points, GridSize.x, GridSize.y);

    generateDistortionMapsfromFixedGrid(imageSize, GridSize, distStrength, GDC_Fixed_Grid_Points, GDC_Fixed_Grid_Map, this->mapX, this->mapY);
}

static bool getTileRect(const cv::Point& pt, const cv::Size& imageSize, const cv::Point& gridSize, const std::vector<std::vector<cv::Point>>& gridPoints, const std::vector<std::vector<cv::Point2f>>& gridPointsMap, cv::Point& gridIndex, RectPoints& cellRect) {
    // Step 1: Calculate cell dimensions
    float cellWidth = (float)imageSize.width / (float)(gridSize.x - 1);
    float cellHeight = (float)imageSize.height / (float)(gridSize.y - 1);

    // Find the grid cell indices that the point falls into
    gridIndex.x = pt.x / cellWidth;
    gridIndex.y = pt.y / cellHeight;

    //// Ensure bounds
    gridIndex.x = std::max(0, std::min(gridIndex.x, gridSize.x - 2));
    gridIndex.y = std::max(0, std::min(gridIndex.y, gridSize.y - 2));

    // Populate the passed-in cellRect with corners
    cellRect.cornersPoint[0][0] = gridPoints[gridIndex.x][gridIndex.y];          // Top-Left
    cellRect.cornersPoint[0][1] = gridPoints[gridIndex.x + 1][gridIndex.y];      // Top-Right
    cellRect.cornersPoint[1][0] = gridPoints[gridIndex.x][gridIndex.y + 1];      // Bottom-Left
    cellRect.cornersPoint[1][1] = gridPoints[gridIndex.x + 1][gridIndex.y + 1];  // Bottom-Right

    // Populate the passed-in cellRect with corners Map
    cellRect.cornersMap[0][0] = gridPointsMap[gridIndex.x][gridIndex.y];          // Top-Left
    cellRect.cornersMap[0][1] = gridPointsMap[gridIndex.x + 1][gridIndex.y];      // Top-Right
    cellRect.cornersMap[1][0] = gridPointsMap[gridIndex.x][gridIndex.y + 1];      // Bottom-Left
    cellRect.cornersMap[1][1] = gridPointsMap[gridIndex.x + 1][gridIndex.y + 1];  // Bottom-Right

    // Error Checking: Ensure pt lies within the calculated rectangle
    if ((pt.x < cellRect.cornersPoint[0][0].x || pt.x > cellRect.cornersPoint[0][1].x ||
        pt.y < cellRect.cornersPoint[0][0].y || pt.y > cellRect.cornersPoint[1][1].y)) {
        // If the point lies outside the rectangle, print an error and return false
        //printf("[Error] cv::Point lies outside the calculated rectangle!\n");
        return false;
    }

    return true;
}

bool getTileRectMapFixed(const cv::Point& pt, const cv::Size& imageSize, const cv::Point& gridSize, std::map<cv::Point, cv::Point2f, PointCompare> GDC_Fixed_Grid_Points, RectPoints& cellRect) {
    // Step 1: Calculate cell dimensions
    float cellWidth = (float)imageSize.width / (float)(gridSize.x - 1);
    float cellHeight = (float)imageSize.height / (float)(gridSize.y - 1);

    cv::Point gridIndex;

    // Find the grid cell indices that the point falls into
    gridIndex.x = pt.x / cellWidth;
    gridIndex.y = pt.y / cellHeight;

    // Populate the passed-in cellRect with corners
    cellRect.cornersPoint[0][0] = cv::Point(gridIndex.x * cellWidth, gridIndex.y * cellHeight); // Top-Left
    cellRect.cornersPoint[0][1] = cv::Point((gridIndex.x + 1) * cellWidth, gridIndex.y * cellHeight); // Top-Right
    cellRect.cornersPoint[1][0] = cv::Point(gridIndex.x * cellWidth, (gridIndex.y + 1) * cellHeight); // Bottom-Left
    cellRect.cornersPoint[1][1] = cv::Point((gridIndex.x + 1) * cellWidth, (gridIndex.y + 1) * cellHeight); // Bottom-Right

    bool allPointsFound = true;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (!findGridPointValue(GDC_Fixed_Grid_Points, cellRect.cornersPoint[i][j], cellRect.cornersMap[i][j])) {
                allPointsFound = false;
                break; // Exit inner loop if point is not found
            }
        }
    }

    if (!allPointsFound)
    {
        return false;
    }

    // Error Checking: Ensure pt lies within the calculated rectangle
    if ((pt.x < cellRect.cornersPoint[0][0].x || pt.x > cellRect.cornersPoint[0][1].x || 
         pt.y < cellRect.cornersPoint[0][0].y || pt.y > cellRect.cornersPoint[1][1].y)) {
        // If the point lies outside the rectangle, print an error and return false
        //printf("[Error] cv::Point lies outside the calculated rectangle!\n");
        return false;
    }

    return true;
}


bool CheckAdaptiveTile(const cv::Point& pt, const cv::Point AdaptivePoints[2], const std::map<cv::Point, cv::Point2f, PointCompare>& GDC_Adaptive_Grid_Points, RectPoints& cellRect) {

    bool allPointsFound = true;
    for (int j = 0; j < 2; ++j) {
        cv::Point2f Map;
        if (!findGridPointValue(GDC_Adaptive_Grid_Points, AdaptivePoints[j], Map)) {
            return false;
        }
    }
    if (allPointsFound) {
        // Point Lies in the 1st Half of the Tile so update Right Corner points
        if (pt.x<= AdaptivePoints[0].x) {
            cellRect.cornersPoint[0][1] = AdaptivePoints[0]; // Top-Right
            cellRect.cornersPoint[1][1] = AdaptivePoints[1]; // Bottom-Right
        }
        else // Point Lies in the 2nd Half of the Tile so update Left Corner points
        {
            cellRect.cornersPoint[0][0] = AdaptivePoints[0]; // Top-Right
            cellRect.cornersPoint[1][0] = AdaptivePoints[1]; // Bottom-Right
        }
    }
    return allPointsFound;
}

bool getTileRectMap(const cv::Point& pt, const cv::Size& imageSize, const cv::Point& gridSize, const std::map<cv::Point, cv::Point2f, PointCompare>& GDC_Adaptive_Grid_Points, RectPoints& cellRect) {
    // Step 1: Calculate cell dimensions
    float cellWidth = (float)imageSize.width / (float)(gridSize.x - 1);
    float cellHeight = (float)imageSize.height / (float)(gridSize.y - 1);

    cv::Point gridIndex;

    // Find the grid cell indices that the point falls into
    gridIndex.x = pt.x / cellWidth;
    gridIndex.y = pt.y / cellHeight;

    // Populate the passed-in cellRect with corners
    cellRect.cornersPoint[0][0] = cv::Point(gridIndex.x * cellWidth, gridIndex.y * cellHeight); // Top-Left
    cellRect.cornersPoint[0][1] = cv::Point((gridIndex.x + 1) * cellWidth, gridIndex.y * cellHeight); // Top-Right
    cellRect.cornersPoint[1][0] = cv::Point(gridIndex.x * cellWidth, (gridIndex.y + 1) * cellHeight); // Bottom-Left
    cellRect.cornersPoint[1][1] = cv::Point((gridIndex.x + 1) * cellWidth, (gridIndex.y + 1) * cellHeight); // Bottom-Right


    // Calculate midpoint of the cell's width for adaptive points
    int midCellWidth = cellWidth / 2;
    //New adaptive Grid Point
    cv::Point AdaptivePoints[2];

    AdaptivePoints[0] = cv::Point((gridIndex.x * cellWidth) + midCellWidth, gridIndex.y * cellHeight); // New Top Adaptive Grid Point
    AdaptivePoints[1] = cv::Point((gridIndex.x * cellWidth) + midCellWidth, (gridIndex.y + 1) * cellHeight); // New Bottom Adaptive Grid Point

    // To check for the new Grid Point
    CheckAdaptiveTile(pt, AdaptivePoints, GDC_Adaptive_Grid_Points, cellRect);

    bool allPointsFound = true;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (!findGridPointValue(GDC_Adaptive_Grid_Points, cellRect.cornersPoint[i][j], cellRect.cornersMap[i][j])) {
                allPointsFound = false;
                break; // Exit inner loop if point is not found
            }
        }
    }

    if (!allPointsFound)
    {
        gridIndex.x = std::max(0, std::min(gridIndex.x, gridSize.x - 2));
        gridIndex.y = std::max(0, std::min(gridIndex.y, gridSize.y - 2));

            // Populate the passed-in cellRect with corners
        cellRect.cornersPoint[0][0] = cv::Point(gridIndex.x * cellWidth, gridIndex.y * cellHeight); // Top-Left
        cellRect.cornersPoint[0][1] = cv::Point((gridIndex.x + 1) * cellWidth, gridIndex.y * cellHeight); // Top-Right
        cellRect.cornersPoint[1][0] = cv::Point(gridIndex.x * cellWidth, (gridIndex.y + 1) * cellHeight); // Bottom-Left
        cellRect.cornersPoint[1][1] = cv::Point((gridIndex.x + 1) * cellWidth, (gridIndex.y + 1) * cellHeight); // Bottom-Right

        allPointsFound = true;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                if (!findGridPointValue(GDC_Adaptive_Grid_Points, cellRect.cornersPoint[i][j], cellRect.cornersMap[i][j])) {
                    allPointsFound = false;
                    break; // Exit inner loop if point is not found
                }
            }
        }
    }

    if (!allPointsFound)
    {
        return false;
    }


    // Error Checking: Ensure pt lies within the calculated rectangle
    if ((pt.x < cellRect.cornersPoint[0][0].x || pt.x > cellRect.cornersPoint[0][1].x ||
        pt.y < cellRect.cornersPoint[0][0].y || pt.y > cellRect.cornersPoint[1][1].y)) {
        // If the point lies outside the rectangle, print an error and return false
        //printf("[Error] cv::Point lies outside the calculated rectangle!\n");
        return false;
    }

    return true;
}


bool getTileRectMap4x4(const cv::Point& pt, const cv::Size& imageSize, const cv::Point& gridSize, const std::map<cv::Point, cv::Point2f, PointCompare>& GDC_Adaptive_Grid_Points, RectPoints& cellRect, RectPoints& cellRectAdaptive) {
    float cellWidth =  (float)imageSize.width  / (float)(gridSize.x - 1);
    float cellHeight = (float)imageSize.height / (float)(gridSize.y - 1);

    // Calculate the closest grid point to the target point
    cv::Point closestGridIndex((float)pt.x / cellWidth, (float)pt.y / cellHeight);

    // Adjust the closest grid point to ensure the 4x4 neighborhood can be centered around it
    closestGridIndex.x = std::max(1, std::min(closestGridIndex.x, gridSize.x - 3));
    closestGridIndex.y = std::max(1, std::min(closestGridIndex.y, gridSize.y - 3));

    memset(cellRectAdaptive.cornersIdx, 0, sizeof(cellRectAdaptive.cornersIdx));

    // Loop to populate the 4x4 neighborhood
    for (int i = -1; i <= 2; ++i) {
        for (int j = -1; j <= 2; ++j) {
            cv::Point gridPoint = closestGridIndex + cv::Point(i, j);

            // Ensure that the grid point is within bounds
            if (gridPoint.x >= 0 && gridPoint.x < gridSize.x && gridPoint.y >= 0 && gridPoint.y < gridSize.y) {
                
                // Assign the gridPoint to cellRect.cornersPoint
                cellRect.cornersPoint[j + 1][i + 1] = cv::Point(gridPoint.x* cellWidth, gridPoint.y * cellHeight);

                // Find the corresponding mapped point in GDC_Adaptive_Grid_Points
                if (findGridPointValue(GDC_Adaptive_Grid_Points, cellRect.cornersPoint[j + 1][i + 1], cellRect.cornersMap[j + 1][i + 1])) {
                    // Optionally, you could also populate cornersIdx here if needed
                    // cellRect.cornersIdx[j + 1][i + 1] = <appropriate index value>;
                }
                else {
                    // If the point is not found, handle it according to your error handling strategy
                    return false;
                }
            }
            else {
                // Handle out-of-bounds grid points, if necessary
                return false;
            }
        }
    }

    // Populate the Adaptive 3x4 Points
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 2; ++j) {
            cv::Point gridPointIdx = closestGridIndex + cv::Point(i, j);

            // Ensure that the grid point is within bounds
            if (gridPointIdx.x >= 0 && gridPointIdx.x < gridSize.x && gridPointIdx.y >= 0 && gridPointIdx.y < gridSize.y) {

                // Assign the gridPointIdx to cellRect.cornersPoint
                cv::Point AdaptiveGridPoint = cv::Point((gridPointIdx.x * cellWidth), gridPointIdx.y * cellHeight) + cv::Point(cellWidth / 2, 0);

                // Find the Adaptive point in GDC_Adaptive_Grid_Points
                if (findGridPointValue(GDC_Adaptive_Grid_Points, AdaptiveGridPoint, cellRectAdaptive.cornersMap[j + 1][i + 1])) {
                    cellRectAdaptive.cornersPoint[j + 1][i + 1] = AdaptiveGridPoint;
                    cellRectAdaptive.cornersIdx[j + 1][i + 1] = 1;
                }
            }
        }
    }
    return true;
}

// Function to calculate the cubic polynomial coefficients
void getCubicCoeffs(float x, float& a, float& b, float& c, float& d) {
    a = -0.5f * x * x * x + x * x - 0.5f * x;
    b = 1.5f * x * x * x - 2.5f * x * x + 1.0f;
    c = -1.5f * x * x * x + 2.0f * x * x + 0.5f * x;
    d = 0.5f * x * x * x - 0.5f * x * x;
}

// Function to perform Bicubic Interpolation
cv::Point2f bicubicInterpolate(const cv::Point& pt, const cv::Size& imageSize, const cv::Point& gridSize, const RectPoints& cellRect) {
    
    float cellWidth = (float)imageSize.width / (float)(gridSize.x - 1);
    float cellHeight = (float)imageSize.height / (float)(gridSize.y - 1);

    // Calculate the relative position of the target pixel within the 4x4 neighborhood
    const float xRatio = (pt.x - cellRect.cornersPoint[1][1].x) / cellWidth;
    const float yRatio = (pt.y - cellRect.cornersPoint[1][1].y) / cellHeight;

    float x_coeffs[4], y_coeffs[4];
    getCubicCoeffs(xRatio, x_coeffs[0], x_coeffs[1], x_coeffs[2], x_coeffs[3]);
    getCubicCoeffs(yRatio, y_coeffs[0], y_coeffs[1], y_coeffs[2], y_coeffs[3]);

    float interpolatedX = 0.0f, interpolatedY = 0.0f;

    for (int i = 0; i < 4; ++i) {
        float x_sum = 0.0f, y_sum = 0.0f;
        for (int j = 0; j < 4; ++j) {
            x_sum += x_coeffs[j] * cellRect.cornersMap[i][j].x;
            y_sum += x_coeffs[j] * cellRect.cornersMap[i][j].y;
        }
        interpolatedX += y_coeffs[i] * x_sum;
        interpolatedY += y_coeffs[i] * y_sum;
    }

    return cv::Point2f(interpolatedX, interpolatedY);
}
// Function to perform Bilinear Interpolation considering local cell position
cv::Point2f bilinearInterpolate(const cv::Point& pt, const RectPoints& cellRect) {
    const float cellWidth = std::abs(cellRect.cornersPoint[0][1].x - cellRect.cornersPoint[0][0].x);
    const float cellHeight = std::abs(cellRect.cornersPoint[1][0].y - cellRect.cornersPoint[0][0].y);

    const float xRatio = (pt.x - cellRect.cornersPoint[0][0].x) / cellWidth;
    const float yRatio = (pt.y - cellRect.cornersPoint[0][0].y) / cellHeight;

    const float inv_xRatio = 1 - xRatio;
    const float inv_yRatio = 1 - yRatio;

    float x1 = cellRect.cornersMap[0][0].x * inv_xRatio + cellRect.cornersMap[0][1].x * xRatio;
    float x2 = cellRect.cornersMap[1][0].x * inv_xRatio + cellRect.cornersMap[1][1].x * xRatio;
    float y1 = cellRect.cornersMap[0][0].y * inv_xRatio + cellRect.cornersMap[0][1].y * xRatio;
    float y2 = cellRect.cornersMap[1][0].y * inv_xRatio + cellRect.cornersMap[1][1].y * xRatio;

    float interpolatedX = x1 * inv_yRatio + x2 * yRatio;
    float interpolatedY = y1 * inv_yRatio + y2 * yRatio;

    return cv::Point2f(interpolatedX, interpolatedY);
}

void FisheyeEffect::generateDistortionMapsfromFixedGridMap(
    const cv::Size& imageSize,
    const cv::Point& gridSize,
    const double distStrength,
    std::map<cv::Point, cv::Point2f, PointCompare>& GDC_Grid_Points,
    cv::Mat& mapX,
    cv::Mat& mapY,
    InterpolationMethod method
) {
    mapX.create(imageSize, CV_32FC1);
    mapY.create(imageSize, CV_32FC1);

    mapX.setTo(0);
    mapY.setTo(0);

    cv::Point2f center(imageSize.width / 2.0f, imageSize.height / 2.0f);

    // Compute the distortion for Fixed grid points
    for (auto& pair : GDC_Grid_Points) {
        const cv::Point gridPoint = pair.first;
        cv::Point2f& gridPointMap = pair.second;

        float deltaX = (gridPoint.x - center.x) / center.x;
        float deltaY = (gridPoint.y - center.y) / center.y;
        float distance = (sqrt(deltaX * deltaX + deltaY * deltaY)) / 2;
        float distortion = 1.0f + distance * distStrength;

        gridPointMap = cv::Point2f(center.x + (deltaX * distortion * center.x),
            center.y + (deltaY * distortion * center.y));

    }

    cv::Point PointSrc; // cv::Point Index
    cv::Point GridIndex; // Grid Index
    RectPoints GridRectMap, GridRectMapAdaptive;

    cv::Point2f CorrectedPoint; // Grid Index

    // Interpolate and fill the missing pixels
    for (int y = 0; y < imageSize.height; ++y) {
        for (int x = 0; x < imageSize.width; ++x) {
            PointSrc = cv::Point(x, y);

            if (!findGridPointValue(GDC_Grid_Points, PointSrc, CorrectedPoint)) {
                if (method == InterpolationMethod::BILINEAR) {
                    if (getTileRectMapFixed(PointSrc, imageSize, gridSize, GDC_Grid_Points, GridRectMap)) {
                        // bilinear interpolation logic
                        CorrectedPoint = bilinearInterpolate(PointSrc, GridRectMap);
                    }
                }
                else
                {
                    //if (getTileRectMap4x4(PointSrc, imageSize, gridSize, GDC_Grid_Points, GridRectMap, GridRectMapAdaptive)) {
                    //    // bilinear interpolation logic
                    //    CorrectedPoint = bilinearInterpolate(PointSrc, GridRectMap);
                    //}

                }
            }

            // Assign the interpolated values to the distortion maps
            mapX.at<float>(y, x) = CorrectedPoint.x;
            mapY.at<float>(y, x) = CorrectedPoint.y;
        }
    }

    bUseGeneratedMaps = true;
}


void FisheyeEffect::generateDistortionMapsfromFixedGrid(
    const cv::Size& imageSize,
    const cv::Point& gridSize,
    const double distStrength,
    const std::vector<std::vector<cv::Point>>& GDC_Fixed_Grid_Points,
    std::vector<std::vector<cv::Point2f>>& GDC_Fixed_Grid_Map,
    cv::Mat& mapX,
    cv::Mat& mapY,
    InterpolationMethod method
) {
    mapX.create(imageSize, CV_32FC1);
    mapY.create(imageSize, CV_32FC1);

    mapX.setTo(0);
    mapY.setTo(0);

    cv::Point2f center(imageSize.width / 2.0f, imageSize.height / 2.0f);

    // Ensure GDC_Fixed_Grid_MapX/Y are the same size as the grid
    GDC_Fixed_Grid_Map.resize(GDC_Fixed_Grid_Points.size());

    // First, compute the distortion for fixed grid points
    for (int i = 0; i < GDC_Fixed_Grid_Points.size(); ++i) {
        GDC_Fixed_Grid_Map[i].resize(GDC_Fixed_Grid_Points[i].size());
        for (int j = 0; j < GDC_Fixed_Grid_Points[i].size(); ++j) {
            const cv::Point& gridPoint = GDC_Fixed_Grid_Points[i][j];

            float deltaX = (gridPoint.x - center.x) / center.x;
            float deltaY = (gridPoint.y - center.y) / center.y;
            float distance = (sqrt(deltaX * deltaX + deltaY * deltaY))/2;
            float distortion = 1.0f + distance * distStrength ;

            GDC_Fixed_Grid_Map[i][j] = cv::Point2f(center.x + (deltaX * distortion * center.x),
                center.y + (deltaY * distortion * center.y));
        }
    }

    
    cv::Point PointSrc; // cv::Point Index
    cv::Point GridIndex; // Grid Index
    RectPoints GridRectMap;

    cv::Point2f CorrectedPoint; // Grid Index
    cv::Point2f LeftTopPoint; // Grid Index

    const float cellWidth =  (float)imageSize.width  / ((float)gridSize.x - 1.0);
    const float cellHeight = (float)imageSize.height / ((float)gridSize.y - 1.0);

    // Interpolate and fill the missing pixels
    for (int y = 0; y < imageSize.height; ++y) {
        for (int x = 0; x < imageSize.width; ++x) {
            PointSrc = cv::Point(x, y);

            getTileRect(PointSrc, imageSize, gridSize, GDC_Fixed_Grid_Points, GDC_Fixed_Grid_Map, GridIndex, GridRectMap);

            CorrectedPoint = bilinearInterpolate(PointSrc, GridRectMap);

            // Assign the interpolated values to the distortion maps
            mapX.at<float>(y, x) = CorrectedPoint.x;
            mapY.at<float>(y, x) = CorrectedPoint.y;
        }
    }

    bUseGeneratedMaps = true;
}


double FisheyeEffect::compareDistortionMaps(
    const cv::Mat& mapX1,
    const cv::Mat& mapY1,
    const cv::Mat& mapX2,
    const cv::Mat& mapY2,
    const std::string Name
) {
    CV_Assert(mapX1.size() == mapX2.size() && mapY1.size() == mapY2.size());
    CV_Assert(mapX1.type() == mapX2.type() && mapY1.type() == mapY2.type());

    double mseX = 0.0;
    double mseY = 0.0;
    int count = 0;

    for (int y = 0; y < mapX1.rows; ++y) {
        for (int x = 0; x < mapX1.cols; ++x) {
            // Calculate squared difference for each element
            double diffX = mapX1.at<float>(y, x) - mapX2.at<float>(y, x);
            double diffY = mapY1.at<float>(y, x) - mapY2.at<float>(y, x);

            mseX += diffX * diffX;
            mseY += diffY * diffY;
            ++count;
        }
    }

    if (count == 0) return -1; // Return an error code if count is 0 to avoid division by zero

    // Calculate the mean squared error
    mseX /= count;
    mseY /= count;

    // Compute the absolute differences
    cv::Mat diffX, diffY;
    cv::absdiff(mapX1, mapX2, diffX);
    cv::absdiff(mapY1, mapY2, diffY);

    // Normalize the differences for visualization
    cv::normalize(diffX, diffX, 0, 255, cv::NORM_MINMAX);
    cv::normalize(diffY, diffY, 0, 255, cv::NORM_MINMAX);
    diffX.convertTo(diffX, CV_8U);
    diffY.convertTo(diffY, CV_8U);

    // Combine the difference maps into a single image
    cv::Mat visualization;
    cv::Mat channels[] = { cv::Mat::zeros(diffX.size(), CV_8U),diffX, diffY  }; // X differences in red, Y differences in green
    cv::merge(channels, 3, visualization);

    // Display the visualization
    cv::namedWindow(Name, cv::WINDOW_AUTOSIZE);
    cv::imshow(Name, visualization);
    cv::imwrite(Name+".png", visualization);

    // Return the average MSE of the two maps
    return (mseX + mseY) / 2.0;
}

static void segmentDistortionMap(const cv::Mat& magnitude_of_distortion, cv::Mat& outputMask, double lowThreshold, double highThreshold) {
    outputMask = cv::Mat::zeros(magnitude_of_distortion.size(), CV_8UC1); // Initialize segmentation mask

    // Simple Thresholding 
    cv::Mat lowMask, mediumMask, highMask;
    inRange(magnitude_of_distortion, 0, lowThreshold, lowMask);
    inRange(magnitude_of_distortion, lowThreshold, highThreshold, mediumMask);
    inRange(magnitude_of_distortion, highThreshold, 1.0, highMask);

    // Assign values to distinguish segments in the output mask
    outputMask.setTo(0, lowMask);
    outputMask.setTo(128, mediumMask);
    outputMask.setTo(255, highMask);
}

// Helper function to find the most frequent segment value in a region
static int findMostFrequentValue(const cv::Mat& segmentedRegion) {
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


int findNearestSquareRoot(int num) {
    // Handle negative input
    if (num < 0) {
        std::cout << "Square root of a negative number is not a real number.\n";
        return -1; // Or any indicator for error
    }

    // Binary search approach (for larger numbers)
    int low = 1, high = num, mid;

    while (low <= high) {
        mid = low + (high - low) / 2;

        if (mid * mid == num) {
            return mid;
        }
        else if (mid * mid < num) {
            low = mid + 1;
        }
        else {
            high = mid - 1;
        }
    }

    return high; // Return the floor of the square root
}

void Generate_FixedGridMap(cv::Size ImageSize, std::map<cv::Point, cv::Point2f, PointCompare>& GDC_Grid_Points, const int Grid_x, const int Grid_y) {
    // Step 1: Calculate cell dimensions (unchanged)
    float cellWidth = (float)ImageSize.width / (float)(Grid_x - 1);
    float cellHeight = (float)ImageSize.height / (float)(Grid_y - 1);

    GDC_Grid_Points.clear(); // Clear the existing map

    for (int i = 0; i < Grid_x; ++i) {
        for (int j = 0; j < Grid_y; ++j) {
            int x = i * cellWidth;
            int y = j * cellHeight;

            cv::Point gridPoint(x, y);

            GDC_Grid_Points[gridPoint] = cv::Point2f(gridPoint); // Initialize with default Point2f
        }
    }

    printf("Fixed Grid  : Total No of Samples : %d , from %d x %d Grid\n",
        (int)GDC_Grid_Points.size(), // Total number of samples
        Grid_x, Grid_y // Dimensions of the original grid
    );
}

void Generate_AdaptiveGridMap(const cv::Mat& magnitude_of_distortion, std::map<cv::Point, cv::Point2f, PointCompare>& GDC_Adaptive_Grid_Points, const int Grid_x, const int Grid_y, const float LowThreshold) {
    /*cv::Mat normalized_magnitude = magnitude_of_distortion.clone();
    normalized_magnitude.convertTo(normalized_magnitude, CV_8U, 255);
    cvtColor(normalized_magnitude, normalized_magnitude, cv::COLOR_GRAY2BGR);*/

    cv::Mat Segmented_DistortionMap;
    segmentDistortionMap(magnitude_of_distortion, Segmented_DistortionMap, LowThreshold, 0.98);

    const int imageWidth = magnitude_of_distortion.cols;
    const int imageHeight = magnitude_of_distortion.rows;

    const float baseCellWidth = (float)imageWidth / (float)(Grid_x - 1);
    const float baseCellHeight = (float)imageHeight / (float)(Grid_y - 1);

    // Clear any existing points
    GDC_Adaptive_Grid_Points.clear();

    cv::Point Pt;
    cv::Point2f PtMap;

    for (int i = 0; i < Grid_x; i++) {
        for (int j = 0; j < Grid_y; j++) {
            const int x = i * baseCellWidth;
            const int y = j * baseCellHeight;

            Pt = cv::Point(x, y);
            PtMap = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
            // Commented out the visualization code
            //circle(normalized_magnitude, cv::Point(x, y), 1, cv::Scalar(255, 0, 0), 2);
            GDC_Adaptive_Grid_Points[Pt] = PtMap*-1;
        }
    }

    for (int i = 0; i < Grid_x; ++i) {
        for (int j = 0; j < Grid_y; ++j) {
            const int x = i * baseCellWidth;
            const int y = j * baseCellHeight;

            // Ensure cell boundaries are within image limits
            const float cellWidth = std::min(baseCellWidth, (float)imageWidth - x);
            const float cellHeight = std::min(baseCellHeight, (float)imageHeight - y);

            if (cellWidth <= 0 || cellHeight <= 0) continue; // Skip invalid cells

            const cv::Rect cellRect(x, y, cellWidth, cellHeight);
            const cv::Mat cellRegion = Segmented_DistortionMap(cellRect);
            const int predominantSegment = findMostFrequentValue(cellRegion);

            const cv::Point newPoint(x + (cellWidth / 2.0), y);
            const cv::Point newPoint2(x + (cellWidth / 2.0), y + cellHeight);

            if (predominantSegment >= 128) {
                GDC_Adaptive_Grid_Points[newPoint]= cv::Point2f(newPoint) * -1;

                if (j == Grid_y - 2) {
                    //cv::circle(normalized_magnitude, newPoint2, 2, Red, 2);
                    GDC_Adaptive_Grid_Points[newPoint2]= cv::Point2f(newPoint2) * -1;
                }

                if (predominantSegment == 255) { // High Distortion
                    //circle(normalized_magnitude, newPoint, 1, cv::Scalar(0, 255, 0), 2);
                }
                else {
                    //circle(normalized_magnitude, newPoint, 1, cv::Scalar(0, 255, 255), 2);
                }
            }
        }
    }

    // Commented out the display call
    //cv::imwrite("4.1_Adaptive Grid Points.png", normalized_magnitude);

    int Nearest_GridSize = findNearestSquareRoot((int)GDC_Adaptive_Grid_Points.size());

    printf("Adaptive Grid: Total No of Samples : %d , from %d x %d Grid : New Points : %d Close to ( %d x %d ) Grid \n",
        (int)GDC_Adaptive_Grid_Points.size(), // Total number of samples
        Grid_x, Grid_y, // Dimensions of the original grid
        (int)(GDC_Adaptive_Grid_Points.size() - (Grid_x * Grid_y)), // Total new points
        Nearest_GridSize, Nearest_GridSize); // Dimensions of the nearest square grid

}

// Function to compute distortion magnitude 
static cv::Mat computeDistortionMagnitude(const cv::Mat& grid_x, const cv::Mat& grid_y) {
    // Validate input matrices
    if (grid_x.type() != CV_32F || grid_y.type() != CV_32F) {
        std::cerr << "Both grid_x and grid_y must be of type CV_32F" << std::endl;
        return cv::Mat();
    }
    if (grid_x.size() != grid_y.size()) {
        std::cerr << "grid_x and grid_y must have the same size" << std::endl;
        return cv::Mat();
    }

    // Compute gradients for both channels (grids)
    cv::Mat grad_x_dx, grad_y_dx, grad_x_dy, grad_y_dy;
    Sobel(grid_x, grad_x_dx, CV_32F, 1, 0, 3);
    Sobel(grid_x, grad_y_dx, CV_32F, 0, 1, 3);
    Sobel(grid_y, grad_x_dy, CV_32F, 1, 0, 3);
    Sobel(grid_y, grad_y_dy, CV_32F, 0, 1, 3);

    // Compute the magnitude of gradients
    cv::Mat magnitude_dx, magnitude_dy;
    magnitude(grad_x_dx, grad_y_dx, magnitude_dx);
    magnitude(grad_x_dy, grad_y_dy, magnitude_dy);

    // Combine the magnitudes to get the total magnitude of distortion
    cv::Mat total_magnitude = magnitude_dx + magnitude_dy; // Simple way to combine, might need adjustment based on your definition of distortion

    // Optionally, normalize the total magnitude for visualization
    cv::Mat normalized_magnitude;
    normalize(total_magnitude, normalized_magnitude, 0, 1, cv::NORM_MINMAX);

    return normalized_magnitude;
}


// A more descriptive name to reflect the functionality



void FisheyeEffect::generateDistortionMapsfromAdaptiveGridMap(
    const cv::Size& imageSize,
    const cv::Point& gridSize,
    const double distStrength,
    std::map<cv::Point, cv::Point2f, PointCompare>& GDC_Adaptive_Grid_Points,
    cv::Mat& mapX,
    cv::Mat& mapY,
    InterpolationMethod method
) {
    mapX.create(imageSize, CV_32FC1);
    mapY.create(imageSize, CV_32FC1);

    mapX.setTo(0);
    mapY.setTo(0);

    cv::Point2f center(imageSize.width / 2.0f, imageSize.height / 2.0f);

    //printf("Performing Grip Map Computation!\n");
    // Compute the distortion for variable grid points
    for (auto& pair : GDC_Adaptive_Grid_Points) {
        const cv::Point gridPoint = pair.first;
        cv::Point2f& gridPointMap = pair.second;

        float deltaX = (gridPoint.x - center.x) / center.x;
        float deltaY = (gridPoint.y - center.y) / center.y;
        float distance = (sqrt(deltaX * deltaX + deltaY * deltaY))/2;
        float distortion = 1.0f + distance * distStrength ;

        gridPointMap = cv::Point2f(center.x + (deltaX * distortion * center.x),
            center.y + (deltaY * distortion * center.y));

    }

    cv::Point PointSrc; // cv::Point Index
    RectPoints GridRectMap, GridRectMapAdaptive;
    cv::Point2f CorrectedPoint; // Grid Index

    // Interpolate and fill the missing pixels
    for (int y = 0; y < imageSize.height; ++y) {
        for (int x = 0; x < imageSize.width; ++x) {
            PointSrc = cv::Point(x, y);
            // Check if PointSrc is a grid point
            
            if (!findGridPointValue(GDC_Adaptive_Grid_Points, PointSrc, CorrectedPoint)) {

                if (method == InterpolationMethod::BILINEAR) {
                    if (getTileRectMap(PointSrc, imageSize, gridSize, GDC_Adaptive_Grid_Points, GridRectMap)) {
                        // bilinear interpolation logic
                        CorrectedPoint = bilinearInterpolate(PointSrc, GridRectMap);
                    }
                }
                else
                {
                    if (getTileRectMap4x4(PointSrc, imageSize, gridSize, GDC_Adaptive_Grid_Points, GridRectMap, GridRectMapAdaptive)) {
                        // BiCubic interpolation logic
                        CorrectedPoint = bicubicInterpolate(PointSrc, imageSize, gridSize, GridRectMap);
                    }
                }
            }

            // Assign the interpolated values to the distortion maps
            mapX.at<float>(y, x) = CorrectedPoint.x;
            mapY.at<float>(y, x) = CorrectedPoint.y;
        }
    }
}

void FisheyeEffect::computeDistortionMapsfromAdaptiveGridMap(cv::Point GridSize, const double distStrength, const float LowThreshold) {

    generateDistortionMaps(distStrength);

    cv::Mat distortionMagnitude = computeDistortionMagnitude(this->mapX, this->mapY);

    std::map<cv::Point, cv::Point2f, PointCompare> GDC_Adaptive_Grid_Points;

    InterpolationMethod interpolation = InterpolationMethod::BICUBIC;

    if (interpolation == InterpolationMethod::BICUBIC) {
        printf("Adaptive Grid : Bicubic Interpolation\n");
    }
    else
    {
        printf("Adaptive Grid : Bilinear Interpolation\n");
    }
    
    Generate_AdaptiveGridMap(distortionMagnitude, GDC_Adaptive_Grid_Points, GridSize.x, GridSize.y,LowThreshold);

    generateDistortionMapsfromAdaptiveGridMap(imageSize, GridSize, distStrength, GDC_Adaptive_Grid_Points, this->mapX, this->mapY, interpolation);
}