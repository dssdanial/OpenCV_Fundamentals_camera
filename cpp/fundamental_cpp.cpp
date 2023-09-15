#include <iostream>
#include <opencv2/core.hpp>
#include <time.h>
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"
//#include <opencv2/nonfree/features2d.hpp>
//cout<<CV_VERSION<<endl; //check the opencv library version
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;



int main(void){
   
    std::string image1_path=cv::samples::findFile("aloeL.jpg");
    std::string image2_path=cv::samples::findFile("aloeR.jpg");

    cv::Mat imgL=cv::imread(image1_path, IMREAD_GRAYSCALE); ///IMREAD_COLOR, If set, always convert image to the 3 channel BGR color image. 
    cv::Mat imgR=cv::imread(image2_path, IMREAD_GRAYSCALE);

    if(imgL.empty() || imgR.empty())
    {
    std::cout << "Could not read the image: " << image1_path << std::endl;
    return 1;
    }


    std::vector<KeyPoint> keypoints1, keypoints2;
    int hessian_threshold=100;
   
    float begin_time1=clock();
    Ptr<SURF> surf=SURF::create(hessian_threshold);
    surf->detect(imgL, keypoints1);
    surf->detect(imgR, keypoints2);
    std::cout<<"SURF time: "<<float(clock()-begin_time1)/CLOCKS_PER_SEC;

    float begin_time2=clock();
    Ptr<SIFT> sift=SIFT::create();
    sift->detect(imgL, keypoints1);
    sift->detect(imgR, keypoints2);
    std::cout<<"\nSIFT time: "<<float(clock()-begin_time2)/CLOCKS_PER_SEC;



    float begin_time3=clock();
    Ptr<ORB> orb=ORB::create();
    orb->detect(imgL, keypoints1);
    orb->detect(imgR, keypoints2);
    std::cout<<"\nORB time: "<<float(clock()-begin_time3)/CLOCKS_PER_SEC;



    // float begin_time4=clock();
    // Ptr<FAST> fast=FAST::create();
    // fast->detect(imgL, keypoints1);
    // fast->detect(imgR, keypoints2);
    // std::cout<<"\nFAST time: "<<float(clock()-begin_time4)/CLOCKS_PER_SEC;



    cv::Mat img_keypoints;
    cv::drawKeypoints(imgL, keypoints1, img_keypoints);


    while(1){
        cv::imshow("test1", img_keypoints); 

    if(cv::waitKey(0)){
        break;
        } // Wait for a keystroke in the window
    }



    return 0;
}
#endif