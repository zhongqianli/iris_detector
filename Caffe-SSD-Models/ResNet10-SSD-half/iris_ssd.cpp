#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <assert.h>

using namespace cv;
using namespace cv::dnn;

#include <iostream>
#include <cstdlib>
using namespace std;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(128);

const char* about = "This sample uses Single-Shot Detector "
                    "(https://arxiv.org/abs/1512.02325) "
                    "with ResNet-10 architecture to detect faces on camera/video/image.\n"
                    "More information about the training is available here: "
                    "<OPENCV_SRC_DIR>/samples/dnn/face_detector/how_to_train_face_detector.txt\n"
                    ".caffemodel model's file is available here: "
                    "<OPENCV_SRC_DIR>/samples/dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel\n"
                    ".prototxt file is available here: "
                    "<OPENCV_SRC_DIR>/samples/dnn/face_detector/deploy.prototxt\n";

//const char* params
//    = "{ help           | false | print usage          }"
//      "{ proto          | deploy.prototxt      | model configuration (deploy.prototxt) }"
//      "{ model          | res10_300x300_ssd_iter_31000.caffemodel     | model weights (res10_300x300_ssd_iter_140000.caffemodel) }"
//      "{ camera_device  | 0     | camera device number }"
//      "{ video          |       | video or image for detection }"
//      "{ min_confidence | 0.5   | min confidence       }";

const char* params
    = "{ help           | false | print usage          }"
      "{ proto          | deploy.half.prototxt      | model configuration (deploy.prototxt) }"
      "{ model          | res10_300x300_ssd.half_iter_31000.caffemodel     | model weights (res10_300x300_ssd_iter_140000.caffemodel) }"
      "{ camera_device  | 0     | camera device number }"
      "{ video          |       | video or image for detection }"
      "{ min_confidence | 0.5   | min confidence       }";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, params);

    if (parser.get<bool>("help"))
    {
        cout << about << endl;
        parser.printMessage();
        return 0;
    }

    String modelConfiguration = parser.get<string>("proto");
    String modelBinary = parser.get<string>("model");

    //! [Initialize network]
    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);
    //! [Initialize network]

    if (net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "prototxt:   " << modelConfiguration << endl;
        cerr << "caffemodel: " << modelBinary << endl;
        cerr << "Models are available here:" << endl;
        cerr << "<OPENCV_SRC_DIR>/samples/dnn/face_detector" << endl;
        cerr << "or here:" << endl;
        cerr << "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector" << endl;
        exit(-1);
    }

//    net.setPreferableBackend(DNN_BACKEND_HALIDE);
//    net.setPreferableTarget(DNN_TARGET_CPU);

//    VideoCapture cap;
//    if (parser.get<String>("video").empty())
//    {
//        int cameraDevice = parser.get<int>("camera_device");
//        cap = VideoCapture(cameraDevice);
//        if(!cap.isOpened())
//        {
//            cout << "Couldn't find camera: " << cameraDevice << endl;
//            return -1;
//        }
//    }
//    else
//    {
//        cap.open(parser.get<String>("video"));
//        if(!cap.isOpened())
//        {
//            cout << "Couldn't open image or video: " << parser.get<String>("video") << endl;
//            return -1;
//        }
//    }

    int cnt = 0;

    for(;;)
    {
        Mat image;
//        cap >> image; // get a new frame from camera/video or read image

//        if (image.empty())
//        {
//            waitKey();
//            break;
//        }

        image = cv::imread("images/S2353L09.jpg", 1);

//        cv::resize(image, image, cv::Size(0, 0), 0.8, 0.8);

        cv::Mat image_result = image.clone();

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        int bt = cv::getTickCount();

        //! [Prepare blob]
        //!  image: 3 channels
        Mat inputBlob = blobFromImage(gray, inScaleFactor,
                                      Size(inWidth, inHeight), Scalar(128), false, false); //Convert Mat to batch of images
        //! [Prepare blob]

        //! [Set input blob]
        net.setInput(inputBlob, "data"); //set the network input
        //! [Set input blob]

        //! [Make forward pass]
        Mat detection = net.forward("detection_out"); //compute output
        //! [Make forward pass]

        vector<double> layersTimings;
        double freq = getTickFrequency() / 1000;
        double time = net.getPerfProfile(layersTimings) / freq;

        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        int et = cv::getTickCount();
        int t = (et - bt) * 1000.0 / cv::getTickFrequency();

        cout << t << " ms" << endl;

        ostringstream ss;
        ss << "FPS: " << 1000/time << " ; time: " << int(time) << " ms";
        putText(image_result, ss.str(), Point(20,20), 0, 0.5, Scalar(0,0,255));

        float confidenceThreshold = parser.get<float>("min_confidence");
        for(int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);

            if(confidence > confidenceThreshold)
            {
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);

                Rect object((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));

                rectangle(image_result, object, Scalar(0, 255, 0));

                ss.str("");
                ss << confidence;
                String conf(ss.str());
                String label = "Iris: " + conf;
                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                rectangle(image_result, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
                                      Size(labelSize.width, labelSize.height + baseLine)),
                          Scalar(255, 255, 255), CV_FILLED);
                putText(image_result, label, Point(xLeftBottom, yLeftBottom),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
            }
        }

        imshow("detections", image_result);
        int key = waitKey(1);
        if (key == 'q')
            break;
        if(key == 's') {
            imwrite("image.jpg", image_result);
        }

    }

    return 0;
} // main
