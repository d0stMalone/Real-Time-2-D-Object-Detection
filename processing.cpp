/* processing.cpp
 * 
 * created by Keval Visaria and Chirag Jain Dhoka
 * 
 * This file contains all the processing function for the object segmentation, detection and classification
 * 
 */

#include <iostream>
#include <vector>
#include <stdio.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/dnn.hpp"
#include "csv_util.h"

using namespace std;
using namespace cv;

/* Separate the object from the back ground according to a threshold given as the second parameter 
 * and the orginalFrame (either the image or the webcam feed) as the first param. 
*/
Mat thresholding(Mat& src, int threshold) {

    /* Converting the image into greyscale*/
    Mat Thresh_Image, grayscale;
    Thresh_Image = Mat(src.size(), CV_8UC1);

    cvtColor(src, grayscale, COLOR_BGR2GRAY);

    /* Changing the pixel value or the thresholdMatrix as per the threshold parameter provided*/
    int i = 0;
    while (i < grayscale.rows) {
        int j = 0;
        while (j < grayscale.cols) {
            if (grayscale.at<uchar>(i, j) > threshold) {
                Thresh_Image.at<uchar>(i, j) = 0;
            }
            else {
                Thresh_Image.at<uchar>(i, j) = 255;
            }
            j++;
        }
        i++;
    }

    return Thresh_Image;
}



/* Function for cleaning up the binary image, uses morphological filtering to first dilate any clean up holes in the image,
   then erode any noies in the image. Uses dilation followed by erosion to remove noise, then
   dilation followed by erosion to remove the holes caused by the reflections.
*/
vector<vector<int>> createStructuringElement(int rows, int cols) {
    vector<vector<int>> element(rows, vector<int>(cols, 1));
    return element;
}

/* Custom erosion function */
Mat customErode(const Mat& src, const vector<vector<int>>& element) {
    Mat dst = src.clone();
    int elementRows = element.size();
    int elementCols = element[0].size();
    int originX = elementRows / 2;
    int originY = elementCols / 2;

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            bool erodePixel = false;
            for (int x = 0; x < elementRows; ++x) {
                for (int y = 0; y < elementCols; ++y) {
                    int relX = i + x - originX;
                    int relY = j + y - originY;
                    if (relX >= 0 && relX < src.rows && relY >= 0 && relY < src.cols) {
                        if (element[x][y] == 1 && src.at<uchar>(relX, relY) == 0) {
                            erodePixel = true;
                            break;
                        }
                    }
                }
                if (erodePixel) break;
            }
            dst.at<uchar>(i, j) = erodePixel ? 0 : 255;
        }
    }
    return dst;
}

/* Custom dilation function */ 
Mat customDilate(const Mat& src, const vector<vector<int>>& element) {
    Mat dst = src.clone();
    int elementRows = element.size();
    int elementCols = element[0].size();
    int originX = elementRows / 2;
    int originY = elementCols / 2;

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            for (int x = 0; x < elementRows; ++x) {
                for (int y = 0; y < elementCols; ++y) {
                    int relX = i + x - originX;
                    int relY = j + y - originY;
                    if (relX >= 0 && relX < src.rows && relY >= 0 && relY < src.cols) {
                        if (element[x][y] == 1 && src.at<uchar>(relX, relY) == 255) {
                            dst.at<uchar>(i, j) = 255;
                            break;
                        }
                    }
                }
            }
        }
    }
    return dst;
}


/* Commutes erode and dilation on the src Mat 
 /* return the dst frame
*/
Mat morphological_operation(Mat src, Mat& dst) {

    Mat dilated = src.clone();
    auto structElem = createStructuringElement(3, 3);

    // Perform custom erosion and dilation
    Mat dilatedImage = customDilate(dilated, structElem);
    Mat erode = dilatedImage.clone();

    Mat erodedImage = customErode(erode, structElem);

    dst = erodedImage;

    return dst;
}


/*Function for connected component analysis, creates segmented, region-colored version of the src image
  Parameters: a src image to be sampled from, then Mat data types for labels, stats, and centroid calculation.
  Returns: a segmented, region colored version of the src image
*/
Mat segment(Mat src, Mat& dst, Mat& colored_dst, Mat& labels, Mat& stats, Mat& centroids) {

    int num = connectedComponentsWithStats(src, labels, stats, centroids, 8);

    //std::cout << num << std::endl;

    // number of colors will equal to number of regions
    vector<Vec3b> colors(num);
    vector<Vec3b> intensity(num);

    // set background to black
    colors[0] = Vec3b(0, 0, 0);
    intensity[0] = Vec3b(0, 0, 0);

    int area = 0;

    for (int i = 1; i < num; i++) {

        colors[i] = Vec3b(255 * i % 256, 170 * i % 256, 200 * i % 256);
        intensity[i] = Vec3b(255, 255, 255);

        // keep only the largest region
        if (stats.at<int>(i, CC_STAT_AREA) > area) {
            area = stats.at<int>(i, CC_STAT_AREA);
        }
        else {
            colors[i] = Vec3b(0, 0, 0);
            intensity[i] = Vec3b(0, 0, 0);
        }
    }
    // assign the colors to regions
    Mat colored_img = Mat::zeros(src.size(), CV_8UC3);
    Mat intensity_img = Mat::zeros(src.size(), CV_8UC3);
    for (int i = 0; i < colored_img.rows; i++) {
        for (int j = 0; j < colored_img.cols; j++)
        {
            int label = labels.at<int>(i, j);
            colored_img.at<Vec3b>(i, j) = colors[label];
            intensity_img.at<Vec3b>(i, j) = intensity[label];
        }
    }

    cvtColor(intensity_img, src, COLOR_BGR2GRAY);
    num = connectedComponentsWithStats(src, labels, stats, centroids, 8);
    dst = intensity_img.clone();
    colored_dst = colored_img.clone();


    
    return dst;
}






/* This function calculates a set of features for a specific area based on a region map and a region identifier. 
 * It computes the Hu moments feature vector and visualizes an oriented bounding box around the area of interest. 
 * The function returns a vector of floating-point numbers representing the Hu moments feature vector for the specified region.
*/
int compute_features(Mat src, Mat& dst, vector<float>& features,string& text2) {
    dst = src.clone();

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Mat gray_pic;
    cvtColor(src, gray_pic, COLOR_BGR2GRAY);
    findContours(gray_pic, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());

    for (size_t i = 0; i < contours.size(); i++) {

        // Calculating Moments for each contour
        Moments moment = moments(contours[i], false);

        // Calculating Hu Moments
        double hu[7];
        HuMoments(moment, hu);

        // Log transform Hu Moments for normalization and add them to the vector
        for (int i = 0; i < 7; i++) {

            hu[i] = -1 * copysign(1.0, hu[i]) * log10(abs(hu[i]));
            features.push_back(hu[i]);
        }

        // Calculating centroid
        Point2f centroid(moment.m10 / moment.m00, moment.m01 / moment.m00);

        // Calculate minimum area rectangle and its properties
        RotatedRect minAreaRec = minAreaRect(contours[i]);
        Point2f rect_points[4];
        minAreaRec.points(rect_points);
        for (int j = 0; j < 4; j++) {
            line(dst, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 255, 0), 2, LINE_8); // Drawing green rotated bounding box
        }

        // Assuming width > height for major and minor axis calculation
        float width = max(minAreaRec.size.width, minAreaRec.size.height);
        float height = min(minAreaRec.size.width, minAreaRec.size.height);

        // Calculate and draw axis line for each region
        double angle = minAreaRec.angle;
        if (minAreaRec.size.width < minAreaRec.size.height) angle += 90.0; // Adjust angle if height is the major axis
        double length = width; // Length of the axis line is consistent
        Point2f endPoint(centroid.x + length * cos(angle * CV_PI / 180.0), centroid.y + length * sin(angle * CV_PI / 180.0));
        line(dst, centroid, endPoint, Scalar(255, 0, 0), 2, LINE_8); // Drawing red axis line

        // Store ratio and percent filled in the struct
        float ratio = width / height;
        float percent_filled = moment.m00 / (width * height);

        features.push_back(ratio);
        features.push_back(percent_filled);

        // Annotate features on the dst image
        stringstream ss;
        ss << "Region " << i + 1 << ": Ratio=" << ratio << ", Percent Filled=" << percent_filled;
        putText(dst, ss.str(), Point(10, 30 + static_cast<int>(i) * 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

        stringstream aa;
        aa << "The Object is: " << text2;

        Point textPosition(100, 100); // Adjust the position as needed
        //putText(dst, ss.str(), Point(10, 30 + static_cast<int>(i) * 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
        putText(dst, aa.str(), Point(100, 400), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2, LINE_AA);
        
    }
    return 0; 
}




/* Calculate the scaled Euclidean Distance of two features: |x1-x2|/st_dev */
float scaledEuclideanDis(std::vector<float>& feature1, std::vector<float>& feature2, std::vector<float>& deviations) {
    float distance = 0.0;
    for (int i = 0; i < feature1.size(); i++) {
        distance += std::sqrt((feature1[i] - feature2[i]) * (feature1[i] - feature2[i]) / deviations[i]);//here is the square root
    }
    return distance;
}




/* Calculate the standard deveation for each entry of the features in the database (for scale), 
 * the result is not square rooted until next function */
int standardDeviation(std::vector<std::vector<float>>& data, std::vector<float>& deviations) {

    std::vector<float> sums = std::vector<float>(data[0].size(), 0.0); //sum of each entry
    
    std::vector<float> avgs = std::vector<float>(data[0].size(), 0.0); //average of each entry
    
    std::vector<float> sumSqure = std::vector<float>(data[0].size(), 0.0); //sum of suqared difference between x_i and x_avg
    
    deviations = std::vector<float>(data[0].size(), 0.0); //final result(deviations not square rooted yet)

    //first loop for the sum of each entry
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[0].size(); j++) {
            sums[j] += data[i][j];
        }
    }

    //calculate the avgs
    for (int i = 0; i < sums.size(); i++) {
        avgs[i] = sums[i] / data.size(); //average
    }

    //second loop, for the sum of  suqared difference of  each entry
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[0].size(); j++) {
            sumSqure[j] += (data[i][j] - avgs[j]) * (data[i][j] - avgs[j]);
        }
    }

    //the deviations
    for (int i = 0; i < sumSqure.size(); i++) {
        deviations[i] = sumSqure[i] / (data.size() - 1);
    }

    return 0;
}






/*
* Simple classifier using Scaled euclidean distance
* vector<vector<double>>  &features: Number of features read from csv
* returns predicted label
*/
void standardDeviation(const vector<vector<float>>& features, vector<float>& deviations) {
    int numFeatures = features.empty() ? 0 : features[0].size();
    deviations.resize(numFeatures, 0.0);
    vector<float> means(numFeatures, 0.0);

    // Calculating means
    for (const auto& feature : features) {
        for (int i = 0; i < numFeatures; i++) {
            means[i] += feature[i];
        }
    }
    for (float& mean : means) {
        mean /= features.size();
    }

    // Calculating standard deviations.
    for (const auto& feature : features) {
        for (int i = 0; i < numFeatures; i++) {
            deviations[i] += pow(feature[i] - means[i], 2);
        }
    }
    for (float& deviation : deviations) {
        deviation = sqrt(deviation / features.size());
    }
}

string classify(vector<float>& features) {

    //char fileName[256] = "D:/My source/Spring2024/PRCV/c++/recognition/RT2D/database.csv";
    char fileName[256] = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/database.csv";

    //std::vector<std::string> labels;
    std::vector<char*> labels;
    std::vector<std::vector<float>> nfeatures;
    read_image_data_csv(fileName, labels, nfeatures, 0);

    double min_dist = std::numeric_limits<double>::infinity();
    string min_label;
    std::vector<float> deviations;
    standardDeviation(nfeatures, deviations);
    for (int i = 0; i < nfeatures.size(); i++) {

        double dist = scaledEuclideanDis(nfeatures[i], features, deviations);
        
        if (dist < min_dist) {

            min_dist = dist;
            min_label = labels[i];
            //cout << "DIST: " << dist << endl;
        }
        //else {
        //    min_label = "Unknown";
        //}
        /*cout << dist;*/
    }
    return min_label;
}

/*
  cv::Mat src        thresholded and cleaned up image in 8UC1 format
  cv::Mat ebmedding  holds the embedding vector after the function returns
  cv::Rect bbox      the axis-oriented bounding box around the region to be identified
  cv::dnn::Net net   the pre-trained network
  int debug          1: show the image given to the network and print the embedding, 0: don't show extra info
 */
int getEmbedding(cv::Mat& src, cv::Mat& embedding, cv::Rect& bbox, cv::dnn::Net& net,  int debug) {
    const int ORNet_size = 128;
    cv::Mat padImg;
    cv::Mat blob;

    cv::Mat roiImg = src(bbox);
    int top = bbox.height > 128 ? 10 : (128 - bbox.height) / 2 + 10;
    int left = bbox.width > 128 ? 10 : (128 - bbox.width) / 2 + 10;
    int bottom = top;
    int right = left;

    cv::copyMakeBorder(roiImg, padImg, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
    cv::resize(padImg, padImg, cv::Size(400, 400));

    cv::dnn::blobFromImage(src, // input image
        blob, // output array
        (1.0 / 255.0) / 0.5, // scale factor
        cv::Size(ORNet_size, ORNet_size), // resize the image to this
        128,   // subtract mean prior to scaling
        false, // input is a single channel image
        true,  // center crop after scaling short side to size
        CV_32F); // output depth/type

    net.setInput(blob);
    embedding = net.forward("onnx_node!/fc1/Gemm");

    if (debug) {
        //cv::imshow("pad image", padImg);
        //std::cout << embedding << std::endl;
        //cv::waitKey(0);
    }

    return(0);
}



/* This function finds the Euclidean distance between the 2 vectors*/
float euclideanDistance(vector<float>& f1, vector<float>& f2) {

    float sum = 0;
    for (int i = 0; i < f1.size(); i++)
        sum += pow((f1[i] - f2[i]), 2);
    return sqrt(sum);
}


/* The cosDistance fucntion takes the dot product of the 2 distances */
float cosDistance(const vector<float>& v1, const vector<float>& v2) {
    float dotProduct = 0.0;
    float normV1 = 0.0;
    float normV2 = 0.0;

    for (size_t i = 0; i < v1.size(); ++i) {
        dotProduct += v1[i] * v2[i];
        normV1 += v1[i] * v1[i];
        normV2 += v2[i] * v2[i];
    }

    normV1 = sqrt(normV1);
    normV2 = sqrt(normV2);

    if (normV1 == 0.0 || normV2 == 0.0) {
        return 0.0; // Prevent division by zero
    }
    else {
        return dotProduct / (normV1 * normV2);
    }
}


double sumSquaredDifference(const vector<float>& a, const vector<float>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
}



/*
* Simple classifier using euclidean distance
* vector<vector<double>>  &features: Number of features read from csv
* returns predicted label
*/
string classifyDNN(vector<float>& features) {

    //char fileName[256] = "D:/My source/Spring2024/PRCV/c++/recognition/RT2D/database.csv";
    char fileName[256] = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/database_DNN.csv";

    //std::vector<std::string> labels;
    std::vector<char*> labels;
    std::vector<std::vector<float>> nfeatures;
    read_image_data_csv(fileName, labels, nfeatures, 0);

    double min_dist = std::numeric_limits<double>::infinity();


    string min_label;
    std::vector<float> deviations;
    standardDeviation(nfeatures, deviations);

    cout << nfeatures.size() << endl;
    for (int i = 0; i < nfeatures.size(); i++) {

        double dist = scaledEuclideanDis(nfeatures[i], features, deviations);
        //cout << "EUC DIS: " << dist << endl;
        if (dist < min_dist) {

            min_dist = dist;
            min_label = labels[i];
            //cout << "Label: " << min_label << endl;
            //cout << "Dist: " << dist << endl;
        }
        //else{
        //    min_label = "Unknown";
        //}
        
    }
    
    return min_label;
}



/* The function will help to convert a Matrix into a vector<float> type.
* THis function was used to convert the embedding matrix from DNN into a vector<float> type 
* to append in the CSV file.
*/
std::vector<float> matToVector(const cv::Mat& mat) {
    std::vector<float> vec;
    // Ensure that the input matrix is not empty
    if (!mat.empty()) {
        // Iterate through the matrix elements
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                // Push the element into the vector
                vec.push_back(mat.at<float>(i, j));
            }
        }
    }
    return vec;
}








