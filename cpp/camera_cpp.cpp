#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main (int argc, char **argv){

    cv::namedWindow("webcam", cv::WINDOW_AUTOSIZE);
    cv::VideoCapture cap;

    if(argc==1){
        cap.open(0);

    }
    else{
        cap.open(argv[1]); 
    }


    if(!cap.isOpened()){
        std::cerr <<"Couldn't open the camera"<<std::endl;
        return -1;
    }

    cv::Mat frame;

    while(cap.isOpened()){
        cap>>frame;

        cv::imshow("capture", frame);

        if(cv::waitKey(30)>=0 ){
            break;
        }

    }   

    return 0;
}
