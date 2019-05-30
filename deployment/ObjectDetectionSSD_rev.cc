// 
// Object Tracking using SSD method 
// Model: MobileNet_V1
// 
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <math.h>
#include <regex>
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>

#include <ctime>
#include <ratio>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;



// Load the graph (xxx.pb), and creates a session object you can use to run it.
tensorflow::Status loadGraph(const string &graph_file_name, unique_ptr<tensorflow::Session> *session) {
    tensorflow::GraphDef graph_def;
    tensorflow::Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) 
        return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    tensorflow::Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) 
        return session_create_status;
    return tensorflow::Status::OK();
}

// Read a labels map file (xxx.pbtxt) from disk to translate class numbers into human-readable labels.
tensorflow::Status readLabelsMapFile(const string &fileName, map<int, string> &labelsMap) {

    // Read file into a string
    ifstream t(fileName);
    if (t.bad())
        return tensorflow::errors::NotFound("Failed to load labels map at '", fileName, "'");
    stringstream buffer;
    buffer << t.rdbuf();
    string fileString = buffer.str();

    // Search entry patterns of type 'item { ... }' and parse each of them
    smatch matcherEntry;
    smatch matcherId;
    smatch matcherName;
    const regex reEntry("item \\{([\\S\\s]*?)\\}");
    const regex reId("[0-9]+");
    const regex reName("\'.+\'");
    string entry;

    auto stringBegin = sregex_iterator(fileString.begin(), fileString.end(), reEntry);
    auto stringEnd = sregex_iterator();

    int id;
    string name;
    for (sregex_iterator i = stringBegin; i != stringEnd; i++) {
        matcherEntry = *i;
        entry = matcherEntry.str();
        regex_search(entry, matcherId, reId);
        if (!matcherId.empty())
            id = stoi(matcherId[0].str());
        else
            continue;
        regex_search(entry, matcherName, reName);
        if (!matcherName.empty())
            name = matcherName[0].str().substr(1, matcherName[0].str().length() - 2);
        else
            continue;
        labelsMap.insert(pair<int, string>(id, name));
    }
    return tensorflow::Status::OK();
}


// Convert Mat image into tensor of shape (1, height, width, d) where last three dims are equal to the original dims.
tensorflow::Status readTensorFromMat(const Mat &mat, tensorflow::Tensor &outTensor) {

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

    // Trick from https://github.com/tensorflow/tensorflow/issues/8033
    float *p = outTensor.flat<float>().data();
    Mat fakeMat(mat.rows, mat.cols, CV_32FC3, p);
    mat.convertTo(fakeMat, CV_32FC3);

    auto input_tensor = Placeholder(root.WithOpName("input"), tensorflow::DT_FLOAT);
    vector<pair<string, tensorflow::Tensor>> inputs = {{"input", outTensor}};
    auto uint8Caster = Cast(root.WithOpName("uint8_Cast"), outTensor, tensorflow::DT_UINT8);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output outTensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    vector<tensorflow::Tensor> outTensors;
    unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"uint8_Cast"}, {}, &outTensors));

    outTensor = outTensors.at(0);
    return tensorflow::Status::OK();
}


// Draw bounding box and add caption to the image.
// Boolean flag _scaled_ shows if the passed coordinates are in relative units (true by default in tensorflow detection)
void drawBoundingBoxOnImage(Mat &image, double yMin, double xMin, double yMax, double xMax, double score, string label, bool scaled=true) {
    cv::Point tl, br;
    if (scaled) {
        tl = cv::Point((int) (xMin * image.cols), (int) (yMin * image.rows));
        br = cv::Point((int) (xMax * image.cols), (int) (yMax * image.rows));
    } else {
        tl = cv::Point((int) xMin, (int) yMin);
        br = cv::Point((int) xMax, (int) yMax);
    }
    cv::rectangle(image, tl, br, cv::Scalar(0, 255, 255), 1);

    // Ceiling the score down to 3 decimals (weird!)
    float scoreRounded = floorf(score * 1000) / 1000;
    string scoreString = to_string(scoreRounded).substr(0, 5);
    string caption = label + " (" + scoreString + ")";

    // Adding caption of type "LABEL (X.XXX)" to the top-left corner of the bounding box
    int fontCoeff = 12;
    cv::Point brRect = cv::Point(tl.x + caption.length() * fontCoeff / 1.6, tl.y + fontCoeff);
    cv::rectangle(image, tl, brRect, cv::Scalar(0, 255, 255), -1);
    cv::Point textCorner = cv::Point(tl.x, tl.y + fontCoeff * 0.9);
    cv::putText(image, caption, textCorner, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0));
}

// Draw bounding boxes and add captions to the image.
//  Box is drawn only if corresponding score is higher than the _threshold_.
void drawBoundingBoxesOnImage(Mat &image,
                              tensorflow::TTypes<float>::Flat &scores,
                              tensorflow::TTypes<float>::Flat &classes,
                              tensorflow::TTypes<float,3>::Tensor &boxes,
                              map<int, string> &labelsMap,
                              vector<size_t> &idxs) {
    for (int j = 0; j < idxs.size(); j++)
        drawBoundingBoxOnImage(image,
                               boxes(0,idxs.at(j),0), boxes(0,idxs.at(j),1),
                               boxes(0,idxs.at(j),2), boxes(0,idxs.at(j),3),
                               scores(idxs.at(j)), labelsMap[classes(idxs.at(j))]);
}


// Calculate intersection-over-union (IOU) for two given bbox Rects.
double IOU(cv::Rect box1, cv::Rect box2) {

    float xA = max(box1.tl().x, box2.tl().x);
    float yA = max(box1.tl().y, box2.tl().y);
    float xB = min(box1.br().x, box2.br().x);
    float yB = min(box1.br().y, box2.br().y);
    float intersectArea = abs((xB - xA) * (yB - yA));
    float unionArea = abs(box1.area()) + abs(box2.area()) - intersectArea;
    return 1. * intersectArea / unionArea;
}


// Return idxs of good boxes (ones with highest confidence score (>= thresholdScore) & IOU <= thresholdIOU with others).
vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &scores,
                           tensorflow::TTypes<float, 3>::Tensor &boxes,
                           double thresholdIOU, double thresholdScore) {

    vector<size_t> sortIdxs(scores.size());
    iota(sortIdxs.begin(), sortIdxs.end(), 0);
    // Create set of "bad" idxs
    set<size_t> badIdxs = set<size_t>();
    size_t i = 0;
    while (i < sortIdxs.size()) {
        if (scores(sortIdxs.at(i)) < thresholdScore)
            badIdxs.insert(sortIdxs[i]);
        if (badIdxs.find(sortIdxs.at(i)) != badIdxs.end()) {
            i++;
            continue;
        }

        cv::Rect box1 = cv::Rect(Point2f(boxes(0, sortIdxs.at(i), 1), boxes(0, sortIdxs.at(i), 0)),
                             Point2f(boxes(0, sortIdxs.at(i), 3), boxes(0, sortIdxs.at(i), 2)));
        for (size_t j = i + 1; j < sortIdxs.size(); j++) {
            if (scores(sortIdxs.at(j)) < thresholdScore) {
                badIdxs.insert(sortIdxs[j]);
                continue;
            }
            cv::Rect box2 = cv::Rect(Point2f(boxes(0, sortIdxs.at(j), 1), boxes(0, sortIdxs.at(j), 0)),
                                 Point2f(boxes(0, sortIdxs.at(j), 3), boxes(0, sortIdxs.at(j), 2)));
            if (IOU(box1, box2) > thresholdIOU)
                badIdxs.insert(sortIdxs[j]);
        }
        i++;
    }

    // Prepare "good" idxs for return
    vector<size_t> goodIdxs = vector<size_t>();
    for (auto it = sortIdxs.begin(); it != sortIdxs.end(); it++)
        if (badIdxs.find(sortIdxs.at(*it)) == badIdxs.end())
            goodIdxs.push_back(*it);

    return goodIdxs;
}







int main() {

    // Path of SSD_mobilenet_V1 Model & labelsMap
    string LABELS = "../model/ssd_mobilenet_v1/labels_map.pbtxt";
    string graphPath = "../model/ssd_mobilenet_v1/frozen_inference_graph.pb";

    // Set input & output nodes names
    string inputLayer = "image_tensor:0";
    vector<string> outputLayer = {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"};

    // Load model from .pb file
    std::unique_ptr<tensorflow::Session> session;
    //LOG(INFO) << "graphPath:" << graphPath;
    tensorflow::Status loadGraphStatus = loadGraph(graphPath, &session);
    if (!loadGraphStatus.ok()) 
    {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return -1;
    } 
    else
        LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;

    // Load labels map from .pbtxt file
    std::map<int, std::string> labelsMap = std::map<int,std::string>();
    tensorflow::Status readLabelsMapStatus = readLabelsMapFile(tensorflow::io::JoinPath(LABELS), labelsMap);
    if (!readLabelsMapStatus.ok()) 
    {
        LOG(ERROR) << "readLabelsMapFile(): ERROR" << loadGraphStatus;
        return -1;
    } else
        LOG(INFO) << "readLabelsMapFile(): labels map loaded with " << labelsMap.size() << " label(s)" << endl;

    // Define video frames 
    cv::Mat frame;
    // Define input & output tensors
    tensorflow::Tensor tensor;
    std::vector<tensorflow::Tensor> outputs;
    double thresholdScore = 0.5;
    double thresholdIOU = 0.8;

    // FPS count
    int frameNumber = 0;
    int cameraIndex = 0; 

    // Start video capture from first camera 
    VideoCapture cap(cameraIndex);

    tensorflow::TensorShape shape = tensorflow::TensorShape();
    shape.AddDim(1);
    shape.AddDim((int64)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    shape.AddDim((int64)cap.get(CV_CAP_PROP_FRAME_WIDTH));
    shape.AddDim(3);

    while (cap.isOpened()) {
        cap >> frame; // Read frame from input camera
        cvtColor(frame, frame, COLOR_BGR2RGB); // color transformation 
    
        // Get start time    
        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        // Convert Mat to Tensor
        tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, shape);
        tensorflow::Status readTensorStatus = readTensorFromMat(frame, tensor);
        if (!readTensorStatus.ok()) 
        {
            LOG(ERROR) << "\nError in Mat->Tensor conversion. " << readTensorStatus;
            return -1;
        }

        // Run graph on Tensor
        outputs.clear();
        tensorflow::Status runStatus = session->Run({{inputLayer, tensor}}, outputLayer, {}, &outputs);
        if (!runStatus.ok()) 
        {
            LOG(ERROR) << "\nError in running the model. " << runStatus;
            return -1;
        }

        // Extract results from the outputs vector
        tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
        tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
        tensorflow::TTypes<float>::Flat numDetections = outputs[3].flat<float>();
        tensorflow::TTypes<float, 3>::Tensor boxes = outputs[0].flat_outer_dims<float,3>();

        // Calculate time for a frame in milisecond 
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        duration<double, std::milli> time_span = t2 - t1;
        frameNumber++;
        cout << "\nFrame: " << frameNumber << ", time:" << time_span.count();

        vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, thresholdScore);
        for (size_t i = 0; i < goodIdxs.size(); i++)
            cout << ", score:" << scores(goodIdxs.at(i)) << ", class:" << labelsMap[classes(goodIdxs.at(i))]
                      << " (" << classes(goodIdxs.at(i)) << "), box:" << "," << boxes(0, goodIdxs.at(i), 0) << ","
                      << boxes(0, goodIdxs.at(i), 1) << "," << boxes(0, goodIdxs.at(i), 2) << ","
                      << boxes(0, goodIdxs.at(i), 3);

        // Color transform 
        cvtColor(frame, frame, COLOR_BGR2RGB);
        
        // Draw bboxes
        drawBoundingBoxesOnImage(frame, scores, classes, boxes, labelsMap, goodIdxs);
        
        // Draw captions 
        cv::putText(frame, to_string(1/time_span.count()).substr(0, 5), Point(0, frame.rows), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255));
        
        imshow("Output video", frame);
        waitKey(1);
    }
    destroyAllWindows();

    return 0;
}




