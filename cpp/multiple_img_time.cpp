#include <iostream>
#include <opencv2/core.hpp>
#include <time.h>
#ifdef HAVE_OPENCV_XFEATURES2D
#include <string>
#include "dirent.h"
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

std::vector<KeyPoint> keypoints1;
int hessian_threshold=100;
float mean_time1=0, mean_time2=0, mean_time3=0;
float total_time1=0, total_time2=0, total_time3=0;
int i=0;

int main() {

    std::string directoryPath = "/home/iit.local/dsabzevari/my_workspace/epipolar_geometry/cpp/data/"; // Replace with your directory path
    std::vector<std::string> imageFiles;

    DIR *dir;
    struct dirent *entry;
    if ((dir = opendir(directoryPath.c_str())) != NULL) {
        while ((entry = readdir(dir)) != NULL) {
            std::string filename = entry->d_name;
            if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".jpg") {
                imageFiles.push_back(directoryPath + filename);
            }
        }

        closedir(dir);

    } else {
        std::cerr << "Error opening directory." << std::endl;
        return 1;
    }


    for (const std::string &imageFile : imageFiles) {
        cv::Mat img = cv::imread(imageFile, IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Error reading image: " << imageFile << std::endl;
            continue;
        }

        // cv::imshow("Image", img);
        // cv::waitKey(0);

        float begin_time1=clock();
        Ptr<SURF> surf=SURF::create(hessian_threshold);
        surf->detect(img, keypoints1);
        total_time1=float(clock()-begin_time1)/CLOCKS_PER_SEC;

        float begin_time2=clock();
        Ptr<SIFT> sift=SIFT::create();
        sift->detect(img, keypoints1);
        total_time2=float(clock()-begin_time2)/CLOCKS_PER_SEC;


        float begin_time3=clock();
        Ptr<ORB> orb=ORB::create();
        orb->detect(img, keypoints1);
        total_time3=float(clock()-begin_time3)/CLOCKS_PER_SEC;

        mean_time1+=total_time1;
        mean_time2+=total_time2;
        mean_time3+=total_time3;
        i++;
    }

    cout<<"SURF average time: "<< mean_time1/i<<endl;
    cout<<"SIFT average time: "<< mean_time2/i<<endl;
    cout<<"ORB average time: "<< mean_time3/i<<endl;
    return 0;

}





#endif