/* processing.h
 *
 * created by Keval Visaria and Chirag Jain Dhoka
 *
 */

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/* Given a src Matrix and an integer  the function creates a black and white frame with a thresholding parameter given as a 2nd parameter*/
Mat thresholding(cv::Mat& src, int thresholding);

/* The thresholded frame is given to the morphological_operation function and the frame is cleaned up by using dilation and erosion
 * The dilate function increases the white region and the erode function does vice-versa
*/
vector<vector<int>> createStructuringElement(int rows, int cols);
Mat customDilate(const Mat& src, const vector<vector<int>>& element);
Mat customErode(const Mat& src, const vector<vector<int>>& element);
Mat morphological_operation(Mat src, Mat& dst);


/* Here the region is detected and the frame is segmented */
Mat segment(Mat src, Mat& dst, Mat& colored_dst, Mat& labels, Mat& stats, Mat& centroids);

/* We compute the features of the region/object detected, features like huMoments, ratio and percent filled
 * and pushes these features in the features to help us append it to the databsae 
 */
int compute_features(Mat src, Mat& dst, vector<float>& features, string& text2);

/* This function helps to classify the objects from the database using the scaled eucledian distance */
string classify(std::vector<float>& features);

/* Scaled Euclidean Distance would take into consideration the standard deviation. 
 * The new distance metric would be |x1- x2|/stdev.
 */
float scaledEuclideanDis(std::vector<float>& feature1, std::vector<float>& feature2, std::vector<float>& deviations);

/* Function to calculate the standard deviation of vector */
int standardDeviation(std::vector<std::vector<float>>& data, std::vector<float>& deviations);

/* We get the embedding vector from this function 
 * we feed in a single channeled thresholded frame and bbox and call the network.
 * An empty matrix called embedding is given to store the embedding features 
*/

int getEmbedding(cv::Mat& src, cv::Mat& embedding, cv::Rect& bbox, cv::dnn::Net& net, int debug);

/* This function is very similar to classify function declared above
 * We use the dataset_DNN instead of the normal dataset
 */
string classifyDNN(vector<float>& features);

/* This function helps to convert the data in a matrix into a vector<float> type*/
std::vector<float> matToVector(const cv::Mat& mat);
