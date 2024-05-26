/* main.cpp
 *
 * created by Keval Visaria and Chirag Jain Dhoka
 *
 * This file contains all the process of key press and calling all the function in order from the processing.h and csv_util.h files.
 * 
 *
 */
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <map>
#include "processing.h"
#include "csv_util.h"

using namespace std;
using namespace cv;

/* Different Mode Flags */
bool trainingModeFlag = false;
bool recognizeModeFlag = false;
bool dnnModeFlag = false;
bool confusionMode = false;
bool trainingDNNModeFLag = false;

bool flag1 = false;
bool flag2 = false;

/* Path to CSV File for Basic Classification and DNN Classification
 * CSV_FILE will store 9 Feature vectors
 * CSV_DNN will store 50 feature vectors
*/
char CSV_FILE[256] = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/database.csv";
char CSV_DNN[256] = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/database_DNN.csv";

/* Path to Image File */ 
char IMAGE[256] = "C:/Users/visar/Downloads/earpods6.jpg";

/*This is for TASK 9 - pre-trained deep network file*/
/* mode file was given */
string mod_filename = "C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RT3D/RT3D/or2d-normmodel-007.onnx";

/* Declaring all the Frame */
Mat originalFrame, thresholdingFrame, cleanUpFrame, segmentedFrame, colorSegmentedFrame, featureImageFrame, imageMat;

/* Store connectcomponents() parameters */ 
Mat labels, stats, centroids;
int image_nLabels;

//int fontFace = FONT_HERSHEY_SIMPLEX;
//double fontScale = 1;
//int thickness = 2;
//Scalar textColor(0, 255, 0);

int main(int argc, char* argv[]) {

    

    /* Creating a Map of the objects that are in the database
     * this will help us int computing the confusion vector
    */
    string temp;

    std::map<std::string, int> mpp;
    mpp["box"] = 1;
    mpp["remote"] = 2;
    mpp["case"] = 3;
    mpp["wallet"] = 4;
    mpp["mouse"] = 5;

    vector<vector<int>> confusionMat(5, vector<int>(5, 0));

    //Thresholding Parameter
    int threshold = 135;

    // read the pre-trained deep network
    cv::dnn::Net net = cv::dnn::readNet(mod_filename);

    imageMat = imread(IMAGE);

    // Open the video device
    VideoCapture* capdev = new cv::VideoCapture(2);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }
    else {
        printf("Stream Started!\n");
    }

    // Get properties of the image
    Size refS((int)capdev->get(CAP_PROP_FRAME_WIDTH), (int)capdev->get(CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    // Create a window to display the video
    namedWindow("Video", 1);

    while (true) {

        /* Initializing the vector to store the features */
        vector<float> feature;
        vector<float> embeddingFeature;

        /* display the Image */
        //imshow("Target Image", imageMat);
       

        *capdev >> originalFrame;
        if (originalFrame.empty()) {
            cerr << "Frame is empty" << endl;
            break;
        }
        imshow("Video", originalFrame);

        feature.clear();

         //Process image with thresholding and cleanup
        /* te original Image is processed that is threshlding and cleanup */
        thresholdingFrame = thresholding(originalFrame, threshold); 
        cleanUpFrame = morphological_operation(thresholdingFrame, cleanUpFrame);
        segmentedFrame = segment(cleanUpFrame, segmentedFrame, colorSegmentedFrame, labels, stats, centroids);
        compute_features(segmentedFrame, featureImageFrame, feature, temp);


        //imshow("After Thresholding", thresholdingFrame);
        //imshow("Clean Image", cleanUpFrame);
        imshow("Boxes and axis", featureImageFrame);
        //imshow("Colored Segmented Frame", colorSegmentedFrame);
        imshow("Segmented image", segmentedFrame);

        /* Waiting for a key press from the user */

        char key = waitKey(25);
        /* Press N to enter a normal TraingingMode */
        if (key == 'n' || key == 'N') {
            trainingModeFlag = true;
        }
        /* Press m to enter DNN training Mode */
        if (key == 'm'|| key == 'M') {
            trainingDNNModeFLag = true;
        }
        /* Press r to recognize the object */
        else if (key == 'r' || key == 'R') {
            if (!recognizeModeFlag) {
                cout << "Recognize Mode " << endl;
                recognizeModeFlag = !recognizeModeFlag;
            }
            else {
                recognizeModeFlag = !recognizeModeFlag;
                cout << "Exiting Recognize Mode!" << endl;
            }
        }
        /* Press k to recoginze object from the DNN database*/
        else if (key == 'd' || key == 'D') {
            //dnnModeFlag = true;
            if (!dnnModeFlag) {
                cout << "DNN Recoginize Mode " << endl;
                dnnModeFlag = !dnnModeFlag;
            }
            else
            {
                dnnModeFlag = !dnnModeFlag;
                cout << "Exiting DNN Recognize Mode!" << endl;
            }
        }
        /* Press C to enter confusion Matrix mode */
        else if (key == 'c' || key == 'C') {
            confusionMode = true;
        }
        /* Press q to exit the program */
        if (key == 'q' || key == 'Q') { // 27 is ASCII for ESC
            break;
        }

        /* Operation of each Modes */
        /* Each mode will have a sequence of operation */
        /* In Training Mode we will train our dataset with the data that is unknown 
         * when the user presses n it goes to training mode and user is asked for Label 
         * and the concurrent data along with the label get stored in the CSV file.
        */
        if (trainingModeFlag) {
            cout << "---------------------------------------" << endl;
            /* Taking the Label name from the User */
            cout << "Training Mode " << endl;
            char label[20];
            cout << "Enter the Label for the object : " << endl;
            cin >> label;

            /* Saves the features under the label provided from the user and saves it in the given CSV_fILE */
            append_image_data_csv(CSV_FILE, label, feature,0);

            //recognizeModeFlag = true;
            /* To exit the Traingin Mode */
            cout << "Exit Trainging Mode!" << endl;
            cout << "---------------------------------------" << endl;
            trainingModeFlag = false;
        }
        else if (recognizeModeFlag) {
            /* In Recognize Mode we do prediction of the object using th features of the trained data stored in our databas (CSV file). 
             * we are taking scaled Eucledian Distance and then finding the minimum distance and giving then least label associated with it.  
             */
            cout << "---------------------------------------" << endl;

            /* Detect what the object and will display the label associated with it on a string*/
            temp = classify(feature);
                 
            //char key2 = waitKey(25);

            /* Printing the Label String*/
            cout << "The Object is: " << temp << endl;

            if (temp == "Unknown" && flag1 == false) {
                //flag1 = true;
                if (flag2 == false) {
                    cout << "Do you want to train?(y/n)" << endl;
                    flag2 = true;
                }
                char key2 = waitKey(1);
                //cin >> key2;

                if (key2 == 'y') {
                    trainingModeFlag = true;
                    recognizeModeFlag = false;
                }
                else if (key2 == 'n') {
                    flag1 = true;
                    recognizeModeFlag = true;
                }
            }
            /* To exit the Recogonize Mode*/
            //cout << "Exiting Recognize Mode!" << endl;
            //cout << "---------------------------------------" << endl;
            //recognizeModeFlag = false;
        }
        else if (trainingDNNModeFLag) {
            /* This for Task 9*/
            /* We are using Pre-trained deep Network for this task
             * we get a embedding vector which is getting stored in a vector declared above.
             * and then similarly like Task 6 running a distance metric and finding out the nearest distance and the assicoated label to it.
            */
            cout << "---------------------------------------" << endl;
            cout << "DNN Training Mode " << endl;

            /* Taking the Label from the User */
            char label[20];
            cout << "Enter the Label for the object : " << endl;
            cin >> label;

            // the getEmbedding function wants a rectangle that contains the object to be recognized
            cv::Rect bbox(0, 0, thresholdingFrame.cols, thresholdingFrame.rows);

            // get the embedding
            cv::Mat embedding;
            getEmbedding(thresholdingFrame, embedding, bbox, net, 1);  // change the 1 to a 0 to turn off debugging

            /* the embedding data received above is in a matrix form 
             * So usin the matToVector function to convert it into a vector<float>.
             * This will make it easy to push to the database (csv file).
             */
            embeddingFeature = matToVector(embedding);

            /* Appending or adding the embeddings vector in the database_DNN.csv 
             * along with the label. We have a different CSV file for the features that we get from getEmbedding function.
             * this is beacuse we are receiving 50 features for each label (object detected).
            */
            append_image_data_csv(CSV_DNN, label, embeddingFeature, 0);

            /* To exit the DNN Training Mode */
            cout << "Exit DNN Trainging Mode!" << endl;
            cout << "---------------------------------------" << endl;
            trainingDNNModeFLag = false;
        }
        else if (dnnModeFlag) {
            /* In this part we are saving the embedding feature vectors in the database_DNN
             * Here similarly to task 5 we are taking a label from the user for and then saving the embedding feature vector obtained
             * from the object into the database_DNN.csv file
            */
            
            cout << "---------------------------------------" << endl;
            cout << "DNN Mode" << endl;

            //cv::cvtColor(originalFrame, originalFrame, cv::COLOR_BGR2GRAY);

            // the getEmbedding function wants a rectangle that contains the object to be recognized
            Rect bbox(0, 0, thresholdingFrame.cols, thresholdingFrame.rows);

            // get the embedding
            Mat embedding;
            getEmbedding(thresholdingFrame, embedding, bbox, net, 1);  // change the 1 to a 0 to turn off debugging

            /* the embedding data received above is in a matrix form
             * So usin the matToVector function to convert it into a vector<float>.
             * This will make it easy to push to the database (csv file).
             */
            embeddingFeature = matToVector(embedding);

            /* Here we run a function similar to the task 6 but instead of the normal databse we use dabase_DNN 
             * We use the normal eucledian distance metric to search for the nearet neighbor and then save the label to a string temp2
             */
            temp = classifyDNN(embeddingFeature);

            /* Displaying what the object is */
            cout << "The Object is: " << temp << endl;

            
            if (temp == "Unknown" && flag1 == false) {
                //flag1 = true;
                if (flag2 == false) {
                    cout << "Do you want to train?(y/n)" << endl;
                    flag2 = true;
                }
                char key2 = waitKey(1);
                //cin >> key2;

                if (key2 == 'y') {
                    trainingModeFlag = true;
                    recognizeModeFlag = false;
                }
                else if (key2 == 'n') {
                    flag1 = true;
                    recognizeModeFlag = true;
                }
            }
            /* Exiting the DNN classification Mode */
            //cout << "Exiting DNN Mode" << endl;
            //dnnModeFlag = false;
            cout << "---------------------------------------" << endl;
        }
        else if (confusionMode) {
            /* In Confusion Matrix mode we try to compare the actual object and the predicted object and try to make a matrix out this
             * this will help us to interpret how good or bad our classification logic function is and how its accuray.
             */
            /* Here we ask the user to what is the actual object in the frame*/
            cout << "---------------------------------------" << endl;
            cout << "Confusion Matrix Mode" << endl;
            string confLabel;
            cout << "Enter the Actual Object Label: " << endl;
            cin >> confLabel;

            /* Here the actualLabel will be detected and the accordingly the index of the matrix will be updated
            */
            int actualLabel = mpp[confLabel] - 1;

            /* Here we run the classify to detect the object that is the the predicted data and thus updating that index of the matric*/
            //string rand = classify(feature);
            
            Rect bbox(0, 0, thresholdingFrame.cols, thresholdingFrame.rows);
            Mat embedding;
            getEmbedding(thresholdingFrame, embedding, bbox, net, 1);  // change the 1 to a 0 to turn off debugging
            embeddingFeature = matToVector(embedding);
            string rand = classifyDNN(embeddingFeature);

            cout << "Pred:  " << rand << endl;
            int predLabel = mpp[rand] - 1;

            /* Printing the index of the Matrix for reference*/
            cout << "Confusion Matrix: " << predLabel << "," << actualLabel << endl;

            /* Updating the element/cell of the Matrix +1  */
            confusionMat[actualLabel][predLabel]++;

            /* Printing the Matrix to visualize */
            cout << "Confusion Matrix elements:" << endl;
            for (const auto& row : confusionMat) {
                for (const auto& element : row) {
                    cout << element << " ";
                }
                cout << endl;
            }

            /* Exiting the COnfusion Matrix Mode */
            cout << "Exiting the Confusion Matriix Mode!" << endl;
            cout << "---------------------------------------" << endl;
            confusionMode = false;
        }


        //stringstream ss;


        //ss << "The Object is: " << temp1;

        //Point textPosition(100, 100); // Adjust the position as needed
        ////putText(dst, ss.str(), Point(10, 30 + static_cast<int>(i) * 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
        //putText(segmentedFrame, ss.str(), Point(10, 30 + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
    }

    /* Cleaing the frames on exiting the program */
    capdev->release();
    destroyAllWindows();

    return 0;
}
