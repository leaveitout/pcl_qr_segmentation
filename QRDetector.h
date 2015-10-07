//
// Created by sean on 02/10/15.
//

#ifndef PCL_QR_SEGMENTATION_QRDETECTOR_H
#define PCL_QR_SEGMENTATION_QRDETECTOR_H

#include <limits>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Line {
private:
    float coeff[3];

public:
    Line(const Point2f& pt1, const Point2f& pt2) {
        coeff[1] = pt2.x - pt1.x;
        coeff[0] = -(pt2.y - pt1.y);
        coeff[2] = -pt1.x*coeff[0] - pt1.y*coeff[1];
    }

    float getSlope() {
        if(coeff[1] != 0.0)
            return -coeff[0]/coeff[1];
        else
            return std::numeric_limits<float>::quiet_NaN();
    }

    Point2f intersect(const Line& line) {
        float det = this->coeff[0]*line.coeff[1] - line.coeff[0]*this->coeff[1];

        if(det != 0.0){
            float detX = this->coeff[1]*line.coeff[2] - line.coeff[2]*this->coeff[1];
            float detY = this->coeff[0]*line.coeff[2] - line.coeff[2]*this->coeff[0];
            return Point2f(detX/det, detY/det);
        }
        else
            return Point2f(std::numeric_limits<float>::quiet_NaN());

    }
};

class QRDetector {

public:
    std::vector<Point2d> detectQrCodes(Mat image);
private:
    vector<int> findDeepContours(const vector<vector<Point> > &contours,
                                 const vector<Vec4i> &hierarchy, int min_depth);
    int countParentContours(int current_contour, const vector<Vec4i> &hierarchy);

    const double PARENT_AREA_RATIO_MIN = 2.1;
    const double PARENT_AREA_RATIO_MAX = 2.8;
    const double GRANDPARENT_AREA_RATIO_MIN = 4.8;
    const double GRANDPARENT_AREA_RATIO_MAX = 5.6;
    const int MIN_AREA = 100;

    Mat gray, binary;

    double approxPolyArea(vector<Point> &contour, double epsilon);

    Point2d calcContourCOG(vector<Point> &contour);

    bool sortLandmarks(vector<Point2d> &points);
};




#endif //PCL_QR_SEGMENTATION_QRDETECTOR_H
