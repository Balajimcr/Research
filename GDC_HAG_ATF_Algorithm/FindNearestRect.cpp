#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random> // Include for random number generation

using namespace cv;
using namespace std;

void drawGridPoints(const vector<vector<Point>>& GDC_Grid_Points, Mat& image, const Scalar& color, int radius, int thickness) {
    // Ensure the image is in a suitable format (like CV_8UC3)
    if (image.type() != CV_8UC3) {
        if (image.type() == CV_8UC1) {
            cvtColor(image, image, COLOR_GRAY2BGR);
        }
        else {
            // Handle other incompatible image types if needed
            cerr << "Error: drawGridPoints expects a CV_8UC3 or CV_8UC1 image." << endl;
            return;
        }
    }

    // Iterate through each row of grid points
    for (const vector<Point>& row : GDC_Grid_Points) {
        // Iterate through each point in the row
        for (const Point& pt : row) {
            circle(image, pt, radius, color, thickness);
        }
    }
}

void drawGridPoints(const vector<vector<Point2f>>& GDC_Grid_Points, Mat& image, const Scalar& color, int radius, int thickness) {
    // Ensure the image is in a suitable format (like CV_8UC3)
    if (image.type() != CV_8UC3) {
        if (image.type() == CV_8UC1) {
            cvtColor(image, image, COLOR_GRAY2BGR);
        }
        else {
            // Handle other incompatible image types if needed
            cerr << "Error: drawGridPoints expects a CV_8UC3 or CV_8UC1 image." << endl;
            return;
        }
    }

    // Iterate through each row of grid points
    for (const vector<Point2f>& row : GDC_Grid_Points) {
        // Iterate through each point in the row
        for (const Point2f& pt : row) {
            circle(image, pt, radius, color, thickness);
        }
    }
}

void Generate_FixedGrid(Size ImageSize, vector<vector<Point>>& GDC_Grid_Points, const int Grid_x, const int Grid_y) {
    int cellWidth = ImageSize.width / (Grid_x-1);
    int cellHeight = ImageSize.height / (Grid_y-1);

    GDC_Grid_Points.clear();
    GDC_Grid_Points.resize(Grid_x); // Create outer vector with 'Grid_x' empty rows 

    for (int i = 0; i < Grid_x; ++i) {
        for (int j = 0; j < Grid_y; ++j) {
            int x = i * cellWidth; // Left most Point
            int y = j * cellHeight;

            GDC_Grid_Points[i].push_back(Point(x, y)); // Add points to i-th row
        }
    }

    printf("GDC_Grid_Points Size : %d x %d \n", GDC_Grid_Points.size(), GDC_Grid_Points[0].size());
}

void Generate_FixedGrid(Size ImageSize, vector<vector<Point2f>>& GDC_Grid_Points, const int Grid_x, const int Grid_y) {
    int cellWidth = ImageSize.width / (Grid_x - 1);
    int cellHeight = ImageSize.height / (Grid_y - 1);

    GDC_Grid_Points.clear();
    GDC_Grid_Points.resize(Grid_x); // Create outer vector with 'Grid_x' empty rows 

    for (int i = 0; i < Grid_x; ++i) {
        for (int j = 0; j < Grid_y; ++j) {
            int x = i * cellWidth; // Left most Point
            int y = j * cellHeight;

            GDC_Grid_Points[i].push_back(Point2f(x, y)); // Add points to i-th row
        }
    }

    printf("GDC_Grid_Points Size : %d x %d \n", GDC_Grid_Points.size(), GDC_Grid_Points[0].size());
}
// Struct to represent a 2x2 array of 4 rectangle corners 
struct RectPoints {
    cv::Point2f corners[2][2];
};

static bool getTileRect(const cv::Point& pt, const cv::Size& imageSize, const cv::Point& gridSize, const std::vector<std::vector<cv::Point2f>>& gridPoints, cv::Point& gridIndex, RectPoints& cellRect) {
    int cellWidth = imageSize.width / (gridSize.x - 1);
    int cellHeight = imageSize.height / (gridSize.y - 1);

    // Find the grid cell indices that the point falls into
    gridIndex.x = pt.x / cellWidth;
    gridIndex.y = pt.y / cellHeight;

    // Ensure bounds
    gridIndex.x = std::max(0, std::min(gridIndex.x, gridSize.x - 2));
    gridIndex.y = std::max(0, std::min(gridIndex.y, gridSize.y - 2));

    // Populate the passed-in cellRect with corners
    cellRect.corners[0][0] = gridPoints[gridIndex.x][gridIndex.y];          // Top-Left
    cellRect.corners[0][1] = gridPoints[gridIndex.x + 1][gridIndex.y];      // Top-Right
    cellRect.corners[1][0] = gridPoints[gridIndex.x][gridIndex.y + 1];      // Bottom-Left
    cellRect.corners[1][1] = gridPoints[gridIndex.x + 1][gridIndex.y + 1];  // Bottom-Right

    // Error Checking: Ensure pt lies within the calculated rectangle
    if (pt.x < cellRect.corners[0][0].x || pt.x > cellRect.corners[0][1].x ||
        pt.y < cellRect.corners[0][0].y || pt.y > cellRect.corners[1][1].y) {
        // If the point lies outside the rectangle, print an error and return false
        printf("[Error] Point lies outside the calculated rectangle!\n");
        return false;
    }

    return true;
}


int main0() {
    Size ImageSize(1280, 720);
    Point Grid(35, 35); // Grid dimensions
    Point GridIndex(35, 35); // Grid dimensions

    vector<vector<Point>> gridPoints;  // To store the generated grid points
    vector<vector<Point2f>> gridPoints2f;  // To store the generated grid points

    Mat srcImage(ImageSize, CV_8UC3, Scalar::all(255)); // White background image
    Generate_FixedGrid(ImageSize, gridPoints2f, Grid.x, Grid.y);
    drawGridPoints(gridPoints2f, srcImage, Scalar( 255,0,0), 1,2); // Red grid points

    int numIterations = 500; // Number of random points to generate

    RectPoints rect2f;

    // Random number generator setup
    random_device rd;
    mt19937 generator(rd());
    uniform_int_distribution<int> xDist(0, ImageSize.width - 1);
    uniform_int_distribution<int> yDist(0, ImageSize.height - 1);

    for (int i = 0; i < numIterations; ++i) {
        // Generate random point
        Point randomPoint(xDist(generator), yDist(generator));

        // Draw a marker at the random point (optional)
        circle(srcImage, randomPoint, 2, Scalar(0, 0, 255), 2);

        // Find the rectangle based on grid points
        if (!getTileRect(randomPoint, ImageSize, Grid, gridPoints2f, GridIndex, rect2f))
        {
            Rect tileRect(rect2f.corners[0][0], rect2f.corners[1][1]);

            // Draw the rectangle
            rectangle(srcImage, tileRect, Scalar(0, 0,255), 2);

            imshow("Image with Grid and Rectangle", srcImage);
            //cv::waitKey();
        }

        Rect tileRect(rect2f.corners[0][0], rect2f.corners[1][1]);

        // Draw the rectangle
        rectangle(srcImage, tileRect, Scalar(0, 255, 0), 2);

        imshow("Image with Grid and Rectangle", srcImage);

        // Introduce a slight delay for visualization
        if (waitKey(1) == 27) break; // Exit if 'ESC' is pressed
    }

    imshow("Image with Grid and Rectangle", srcImage);
    waitKey(0);

    return 0;
}
