
#include <iostream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <fstream>

// opencv lib
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// tensorflow lib
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include "tensorflow/core/framework/op_kernel.h"

using namespace std;
using namespace cv;
using namespace tensorflow;

// global variables
Session* session;
GraphDef graph_def;
float R_MEAN = 123.68;
float G_MEAN = 116.779;
float B_MEAN = 103.939;


void writeMatToFile(cv::Mat& m, const char* filename)
{
    ofstream fout(filename);
    if(!fout){
        cout<<"File Not Opened"<<endl;  return;
    }
    for(int i=0; i<m.rows; i++){
        for(int j=0; j<m.cols; j++){
            fout<<m.at<float>(i,j)<<"\t";
        }
        fout<<endl;
    }
    fout.close();
}

void writeVecToFile(std::vector<float> m, const char* filename)
{
    ofstream fout(filename);
    if(!fout){
        cout<<"File Not Opened"<<endl;  return;
    }
    for(size_t i=0; i<m.size(); i++){
        fout<<m[i]<<"\t";
    }
    fout.close();
}

void Model_Init(string model_name){
    // Initialize a tensorflow session
    // Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
      std::cerr << status.ToString() << "\n";
      return;
    } else {
      std::cout << "Session created successfully" << std::endl;
    }

    // Load graph protobuf
    // GraphDef graph_def;
    std::string graph_path = model_name;
    status = ReadBinaryProto(Env::Default(), graph_path, &graph_def);
    if (!status.ok()) {
      std::cerr << status.ToString() << std::endl;
    } else {
      std::cout << "Load graph protobuf successfully" << std::endl;
    }

    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok()) {
      std::cerr << status.ToString() << std::endl;
      return;
    } else {
      std::cout << "Add graph to session successfully" << std::endl;
    }
    return;
}

cv::Mat Model_Inference(const cv::Mat &img, string model_name){
    Model_Init(model_name);
    cv::Mat inputImage;
    // cv::resize(img, inputImage, cv::Size(), 0.65, 0.65, INTER_CUBIC);
    cv::resize(img, inputImage, cv::Size(480, 480), INTER_CUBIC);
    if(inputImage.empty())
        cout << "Error in loading image " << endl;
    cv::Mat rz_input, float_input;
    std::vector<cv::Mat> rst;
    rz_input = inputImage.clone();
    int height = rz_input.rows;
    int width = rz_input.cols;
    cout << "stage0: input image size HxW is: " << height << " X " << width << endl;
    int depth = 3;
    int mod_h = height % 16;
    int mod_w = width % 16;
    cv::Rect roi;
    roi.x = 0;
    roi.y = 0;
    roi.width = width - mod_w;
    roi.height = height - mod_h;
    cv::Mat input = rz_input(roi);
    height = input.rows;
    width = input.cols;
    cv::Mat f_input;
    std::vector<cv::Mat> M, rgb;
    input.convertTo(f_input, CV_32FC1, 1.0);
    cv::split(f_input, M);
    rgb.push_back(M[2]-R_MEAN);
    rgb.push_back(M[1]-G_MEAN);
    rgb.push_back(M[0]-B_MEAN);

    cv::Mat rgb_input;
    cv::merge(rgb, rgb_input);
    cv::Mat padded_in;
    cv::copyMakeBorder(rgb_input, padded_in, 16, 16, 16, 16, BORDER_REPLICATE);
    height = padded_in.rows;
    width = padded_in.cols;

    //padded_in.convertTo(float_input, CV_32FC1, 1.0/255.0);
    float_input = padded_in;
    cout << "stage1: input image size HxW is: " << rgb_input.rows << " X " << rgb_input.cols << endl;
    // input tensor
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, height, width, depth}));
    auto input_tensor_mapped = input_tensor.tensor<float, 4>();

    cout << "input image size HxW is: " << height << " X " << width << endl;
    // copy data into the corresponding tensor
    const float *source_data = (float*) float_input.data;
    for (int y = 0; y < height; ++y) {
        const float* source_row = source_data + (y * width * depth);
        for (int x = 0; x < width; ++x) {
            const float* source_pixel = source_row + (x * depth);
            for (int c = 0; c < depth; ++c) {
                const float* source_value = source_pixel + c;
                input_tensor_mapped(0, y, x, c) = *source_value;
            }
        }
    }
    const std::string kInputLayerName = "input";
    const std::string kOutputLayerName = "ofuse/Sigmoid";
    std::vector<tensorflow::Tensor> finalOutput, debug_input;
    tensorflow::Status run_status = session->Run({{kInputLayerName, input_tensor}},
                                                         {kOutputLayerName},
                                                         {},
                                                        &finalOutput);
    if (run_status.ok() != true) {
        std::cout << "tf_session->Run error: " << run_status.error_message() << std::endl;
    }

    tensorflow::Tensor output = std::move(finalOutput.at(0));
    cv::Mat outputMat = cv::Mat((int)output.dim_size(1), (int)output.dim_size(2), CV_32FC1, output.flat<float>().data());
    cv::Mat edge;
    edge.create(outputMat.rows, outputMat.cols, CV_32FC1);
    cout << "output edge map size HxW is: " << outputMat.rows << " X " << outputMat.cols <<endl;
    for(int y=0; y<outputMat.rows; y++){
        float *in_row = outputMat.ptr<float>(y);
        float *out_row = edge.ptr<float>(y);
        for(int x=0; x<outputMat.cols; x++){
        //    out_row[x] = 1.0/(1.0+exp(-1.0*in_row[x]));
        //    out_row[x] = in_row[x] / 6.0;  // relu6
            out_row[x] = in_row[x];  // relu
        //    out_row[x] = (exp(1.0*in_row[x]) - exp(-1.0*in_row[x])) / (exp(1.0*in_row[x]) + exp(-1.0*in_row[x])); // tanh activation
        }
    }
    // writeMatToFile(edge, "edge_scoreMap.txt");
    // remove added border
    // cv::Mat crop_edge = hed_edge(cv::Rect(16, 16, ww-32, hh-32));
    return edge;
}

int main(int argc, char* argv[]){
    string img_name = argv[1];
    string model_name = argv[2];
    cv::Mat input_img = imread(img_name, 1);
    if ( input_img.empty() )
        cout << "Loading input image ---> " << img_name << " Error!" << endl;
    int height = input_img.rows;
    int width = input_img.cols;
    cout << "input image size HxW is: " << height << " X " << width << endl;
    struct timeval fun_start, fun_end;
    cv::Mat edge_map, final_map;
    gettimeofday(&fun_start, NULL);
    edge_map = Model_Inference(input_img, model_name);
    gettimeofday(&fun_end, NULL);
    cout << "model inference cost: " << 1000*1000*(fun_end.tv_sec-fun_start.tv_sec) + (fun_end.tv_usec-fun_start.tv_usec)  << " us " << endl;
    edge_map.convertTo(final_map, CV_8UC1, 255.0);
    cv::imwrite("edge_map.jpg", final_map);
    return 0;
}


