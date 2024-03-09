#include <opencv2/opencv.hpp>
#include <iostream>

// Define a comparison function for the map to be able to use cv::Point as keys
struct PointCompare {
    bool operator()(const cv::Point& a, const cv::Point& b) const {
        if (a.x < b.x) return true;
        if (a.x > b.x) return false;
        return a.y < b.y;
    }
};

void Generate_FixedGrid(cv::Size ImageSize, std::map<cv::Point, cv::Point2f, PointCompare>& GDC_Grid_Points, const int Grid_x, const int Grid_y) {
    int cellWidth = ImageSize.width / (Grid_x - 1);
    int cellHeight = ImageSize.height / (Grid_y - 1);

    GDC_Grid_Points.clear();

    for (int i = 0; i < Grid_x; ++i) {
        for (int j = 0; j < Grid_y; ++j) {
            int x = i * cellWidth;
            int y = j * cellHeight;

            // Since we're using a map, we insert a pair of cv::Point and cv::Point2f
            GDC_Grid_Points[cv::Point(x, y)] = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
        }
    }
}

int main() {
    using namespace cv;
    using namespace std;

    Size ImageSize(1280, 720);
    Point Grid(2, 2);

    Mat srcImage(ImageSize, CV_8UC3, Scalar::all(255));
    

    int interpolation = INTER_LINEAR;
    int borderMode = BORDER_CONSTANT;

    const double distStrength = 1.5;

    double  rms_error;

    std::map<cv::Point, cv::Point2f, PointCompare> GDC_Grid_Points;

    Generate_FixedGrid(ImageSize, GDC_Grid_Points, Grid.x, Grid.y);

    cout << "Input :\n";
    for (auto& pair : GDC_Grid_Points) {
        cout << "Point    : " << pair.first << "\n";
        cout << "Point Map: " << pair.second << "\n";
    }


    cout << "Output :\n";
    for (auto& pair : GDC_Grid_Points) {
        // Scale both coordinates by 2.0
        pair.second.x *= 2.0f;
        pair.second.y *= 2.0f;

        cout << "Point    : " << pair.first << "\n";
        cout << "Point Map: " << pair.second << "\n";
    }


    return 1;

}