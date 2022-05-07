//TensorRT头文件
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <algorithm>
#include <float.h>
#include <string.h>
#include <chrono>
#include <iterator>
#include <map>
#include <iomanip>
#include <string>
#include <unistd.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "NvInferPluginUtils.h"
#include "NvInferVersion.h"
#include "NvCaffeParser.h"
#include "NvUtils.h"
#include "nvblas.h"
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;

#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000*3000

static constexpr int  OUTPUT_SIZE = 8400 * 85;
static char* INPUT_BLOB_NAME = "data";
static char* OUTPUT_BLOB_NAME = "prob";
static constexpr int INPUT_H = 640;
static constexpr int INPUT_W = 640;




#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if(error_code != cudaSuccess){\
            std::cerr << "CUDA error" << error_code << "at" << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
    


//  Instantiate the ILogger interface 
class Logger : public ILogger{
    void log(Severity serverity,const char* msg) noexcept override
    {
        if(serverity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

std::map<std::string, Weights> loadWeights(const std::string file){
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string,Weights> weightMap;

    // open .wts file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weights file. please check if the .wts file path is right !!!!");

    // read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file");

    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }

        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}

cv::Mat static_resize(cv::Mat& img) {
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

float* blobFromImage(cv::Mat& img){
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    return blob;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string,Weights>& weightMap, ITensor& input, std::string lname, float eps){
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    // 将特定的物理内存地址 赋值给一个指针
    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for(int i = 0; i < len; i++){
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for(int i = 0; i < len; i++){
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for(int i = 0; i < len; i++){
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);

    return scale_1;
    
}

ILayer* convBlock(INetworkDefinition *network, std::map<std::string,Weights> weightMap, ITensor &input, int ouch, int ksize, int s,int g, std::string lname){
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int p = (ksize - 1) / 2;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, ouch, Dims2{ ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);

    conv1->setStrideNd(Dims2{ s, s });
    conv1->setPaddingNd(Dims2{ p, p });
    conv1->setNbGroups(g);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);

    // silu = x * sigmoid
    auto sig = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig);
    // 元素方面的点积
    auto ew = network->addElementWise(*bn1->getOutput(0), *sig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);

    return ew;

}

// FOCUS Layer Input (B,C,H,W) -> Output (B,4C,H/2,W/2)
ILayer* focus(INetworkDefinition *network, std::map<std::string,Weights>& weightMap, ITensor& input, int inch, int ouch, int ksize, std::string lname){
    ISliceLayer *s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ inch, INPUT_H / 2, INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s2 = network->addSlice(input, Dims3{ 0, 1, 0 }, Dims3{ inch, INPUT_H / 2, INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s3 = network->addSlice(input, Dims3{ 0, 0, 1 }, Dims3{ inch, INPUT_H / 2, INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ISliceLayer *s4 = network->addSlice(input, Dims3{ 0, 1, 1 }, Dims3{ inch, INPUT_H / 2, INPUT_W / 2 }, Dims3{ 1, 2, 2 });
    ITensor* inputTensors[] = {s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0)}; 
    auto cat = network->addConcatenation(inputTensors,4);
    auto conv = convBlock(network, weightMap, *cat->getOutput(0), ouch, ksize, 1, 1, lname + ".conv");
    return conv;
}

ILayer* SPPbottleneck(INetworkDefinition *network, std::map<std::string,Weights>weightMap, ITensor& input, int inch, int ouch, std::string lname){
    int hidden_channels = inch / 2;
    auto conv1 = convBlock(network, weightMap, input, hidden_channels, 1, 1, 1, lname + ".conv1");


    auto pool1 = network->addPoolingNd(*conv1->getOutput(0), PoolingType::kMAX, Dims2{ 5, 5 });
    pool1->setPaddingNd(Dims2 { 5 / 2, 5 / 2 });
    pool1->setStrideNd(Dims2{ 1, 1 });

    auto pool2 = network->addPoolingNd(*conv1->getOutput(0), PoolingType::kMAX, Dims2{ 9, 9 });
    pool2->setPaddingNd(Dims2 { 9 / 2, 9 / 2 });
    pool2->setStrideNd(Dims2{ 1, 1 });

    auto pool3 = network->addPoolingNd(*conv1->getOutput(0), PoolingType::kMAX, Dims2{ 13, 13 });
    pool3->setPaddingNd(Dims2 { 13 / 2, 13 / 2 });
    pool3->setStrideNd(Dims2{ 1, 1 });

    ITensor* inputTensor[] = { conv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0)};
    auto cat = network->addConcatenation(inputTensor, 4);
    
    auto conv2 = convBlock(network, weightMap, *cat->getOutput(0), ouch, 1, 1, 1, lname + ".conv2");
    return conv2;
}

ILayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, bool shortcut, int g, float e, std::string lname){
    auto cv1 = convBlock(network, weightMap, input, c2, 1, 1, 1, lname + ".conv1");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".conv2");
    if(shortcut && c1 == c2){
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
        return ew; 
    return cv2;
    }
}

ILayer* CSPLayer(INetworkDefinition *network, std::map<std::string,Weights>& weightMap, ITensor& input, int inch, int ouch, int n, float expansion, bool shortcut, std::string lname){
    int hidden_ch = (int)((float)ouch * expansion);

    auto conv1 = convBlock(network, weightMap, input, hidden_ch, 1, 1, 1, lname + ".conv1");
    auto conv2 = convBlock(network, weightMap, input, hidden_ch, 1, 1, 1, lname + ".conv2");
    // bottleneck + catcentation
    ITensor* y = conv1->getOutput(0);
    for(int i = 0; i < n; i++){
        auto b = bottleneck(network, weightMap, *y, hidden_ch, hidden_ch, true, 1, 0.5, lname + ".m." + std::to_string(i));
        y = b->getOutput(0);
    }
    ITensor* inputTensor[] = { y, conv2->getOutput(0) };
    auto cat = network->addConcatenation(inputTensor, 2);

    auto conv3 = convBlock(network, weightMap, *cat->getOutput(0), ouch, 1 ,1, 1, lname + ".conv3");

    return conv3;


}

ILayer* head_convBlock(INetworkDefinition *network, std::map<std::string,Weights> weightMap, ITensor &input, int ouch, int ksize, int s, int g, std::string lname, std::string lname1, int index){
    auto conv1 = convBlock(network, weightMap, input, ouch, ksize, s, g, lname + "." + std::to_string(index) + ".0");
    auto conv2 = convBlock(network, weightMap, *conv1->getOutput(0), ouch, ksize, s, g, lname + "." + std::to_string(index) + ".1");

    auto conv3 = convBlock(network, weightMap, input, ouch, ksize, s, g, lname1 + "." + std::to_string(index) + ".0");
    auto conv4 = convBlock(network, weightMap, *conv3->getOutput(0), ouch, ksize, s, g, lname1 + "." + std::to_string(index) + ".1");

    auto reg = network->addConvolutionNd(*conv2->getOutput(0), 4, Dims2{ 1, 1 }, weightMap["head.reg_preds." + std::to_string(index) + ".weight"], weightMap["head.reg_preds." + std::to_string(index) + ".bias"] );
    assert(reg);

    reg->setStrideNd(Dims2{ 1, 1 });
    reg->setNbGroups(g);

    auto cls = network->addConvolutionNd(*conv4->getOutput(0), 80, Dims2{ 1, 1 }, weightMap["head.cls_preds." + std::to_string(index) + ".weight"], weightMap["head.cls_preds." + std::to_string(index) + ".bias"] );
    assert(cls);

    cls->setStrideNd(Dims2{ 1, 1 });
    cls->setNbGroups(g);
    IActivationLayer* sigmod0 = network->addActivation(*cls->getOutput(0), ActivationType::kSIGMOID);
    assert(sigmod0);

    auto obj = network->addConvolutionNd(*conv2->getOutput(0), 1, Dims2{ 1, 1 }, weightMap["head.obj_preds." + std::to_string(index) + ".weight"], weightMap["head.obj_preds." + std::to_string(index) + ".bias"] );
    assert(obj);

    obj->setStrideNd(Dims2{ 1, 1 });
    obj->setNbGroups(g);

    IActivationLayer* sigmod1 = network->addActivation(*obj->getOutput(0), ActivationType::kSIGMOID);
    assert(sigmod1);

    ITensor* inputTensor[] = {reg->getOutput(0), cls->getOutput(0), obj->getOutput(0)};
    auto cat = network->addConcatenation(inputTensor, 3);
    return cat;

}


void doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(1, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv){
    // 创建builder
    IBuilder* builder = createInferBuilder(gLogger);
    std::cout<< "builder has been built successfully" <<std::endl;

    builder->setMaxBatchSize(1);
    // 创建config
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 30);

    // 创建network(crateNetwork后的参数)
    INetworkDefinition* network = builder->createNetworkV2(0U);
    std::cout<< "network has been built successfully" <<std::endl;

    // 创建 网络层
    // 1. input layer
    ITensor* data = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, Dims3( 3, INPUT_H, INPUT_W ));
    assert(data);
    std::cout<< "data has been built successfully" <<std::endl;

    // load the weight file
    std::map<std::string,Weights> weightMap = loadWeights("/root/Downloads/Yolox_tensorRT/yolox_tensorRT_hand/yolox.wts");
    int base_channel = 32;
    int base_depth = 1;

    /* --------------yolox backbone------------ */
    // yolo.pafpn.py
    // stem 
    auto stem = focus(network, weightMap,  *data, 3, base_channel, 3, "backbone.backbone.stem");

    // dark 2
    auto conv0 = convBlock(network, weightMap, *stem->getOutput(0), base_channel * 2, 3, 2, 1, "backbone.backbone.dark2.0");
    auto dark2 = CSPLayer(network, weightMap, *conv0->getOutput(0), 64, 64, base_depth, 0.5, true, "backbone.backbone.dark2.1");

    // dark 3 x2
    auto conv1 = convBlock(network, weightMap, *dark2->getOutput(0),base_channel * 4, 3, 2, 1, "backbone.backbone.dark3.0");
    auto dark3 = CSPLayer(network, weightMap, *conv1->getOutput(0), 128, 128, base_depth * 3, 0.5, true, "backbone.backbone.dark3.1");
    
    // dark 4  x1
    auto conv2 = convBlock(network, weightMap, *dark3->getOutput(0), base_channel * 8, 3, 2, 1, "backbone.backbone.dark4.0");
    auto dark4 = CSPLayer(network, weightMap, *conv2->getOutput(0), 256, 256, base_depth * 3, 0.5, true, "backbone.backbone.dark4.1");

    // dark 5  x0
    auto conv3 = convBlock(network, weightMap, *dark4->getOutput(0), base_channel * 16, 3, 2, 1, "backbone.backbone.dark5.0");
    auto sppf = SPPbottleneck(network,weightMap,*conv3->getOutput(0),512, 512, "backbone.backbone.dark5.1");
    auto dark5 = CSPLayer(network, weightMap, *conv3->getOutput(0), 512, 512, base_depth, 0.5, true, "backbone.backbone.dark5.2" );

    //* -------------yolox dark3->x2, dark4->x1, dark5->x0------------------ */
    // yolo_head.py
    // fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
    // f_out0 = self.upsample(fpn_out0)  # 512/16
    // f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
    // f_out0 = self.C3_p4(f_out0)  # 1024->512/16

    auto lateral = convBlock(network, weightMap, *dark5->getOutput(0), 256, 1, 1, 1, "backbone.lateral_conv0");

    auto upsample = network->addResize(*lateral->getOutput(0));
    assert(upsample);
    upsample->setResizeMode(ResizeMode::kNEAREST);
    upsample->setOutputDimensions(dark4->getOutput(0)->getDimensions());
    ITensor* inputTensor1[] = { upsample->getOutput(0), dark4->getOutput(0) };
    auto cat1 = network->addConcatenation(inputTensor1, 2);
    auto c3_p4  = CSPLayer(network, weightMap, *cat1->getOutput(0), 256, 256, base_depth, 0.5, false, "backbone.C3_p4");

    // fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
    // f_out1 = self.upsample(fpn_out1)  # 256/8
    // f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
    // pan_out2 = self.C3_p3(f_out1)  # 512->256/8

    auto reduce = convBlock(network, weightMap, *c3_p4->getOutput(0), 128, 1, 1, 1, "backbone.reduce_conv1");

    auto upsample1 = network->addResize(*reduce->getOutput(0));
    assert(upsample1);
    upsample1->setResizeMode(ResizeMode::kNEAREST);
    upsample1->setOutputDimensions( dark3->getOutput(0)->getDimensions());
    ITensor* inputTensor2[] = { dark3->getOutput(0), upsample1->getOutput(0) };
    auto cat2 = network->addConcatenation(inputTensor2, 2);
    auto c3_p3 = CSPLayer(network, weightMap, *cat2->getOutput(0), 128, 128, 1, 0.5, false, "backbone.C3_p3");

    // p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
    // p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
    // pan_out1 = self.C3_n3(p_out1)  # 512->512/16
    auto bu_conv = convBlock(network, weightMap, *c3_p3->getOutput(0), 128, 3, 2, 1, "backbone.bu_conv2");
    ITensor* inputTensor3[] = { bu_conv->getOutput(0), reduce->getOutput(0) };
    auto cat3 = network->addConcatenation(inputTensor3, 2);
    auto c3_n3 = CSPLayer(network, weightMap, *cat3->getOutput(0), 128, 256, 1, 0.5, false, "backbone.C3_n3");

    // p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
    // p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
    // pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
    auto bu_conv1 = convBlock(network, weightMap, *c3_n3->getOutput(0), 256, 3, 2, 1, "backbone.bu_conv1");
    ITensor* inputTensor4[] = { lateral->getOutput(0), bu_conv1->getOutput(0) };
    auto cat4 = network->addConcatenation(inputTensor4, 2);
    auto c3_n4 = CSPLayer(network, weightMap, *cat4->getOutput(0), 256, 512, 1, 0.5, false, "backbone.C3_n4");

    
    /* --------------- yolo head + yolo detect -------------------*/
    auto head_stem0 = convBlock(network, weightMap, *c3_p3->getOutput(0), 128, 1, 1, 1, "head.stems.0");
    auto head_stem1 = convBlock(network, weightMap, *c3_n3->getOutput(0), 128, 1, 1, 1, "head.stems.1");
    auto head_stem2 = convBlock(network, weightMap, *c3_n4->getOutput(0), 128, 1, 1, 1, "head.stems.2");

    auto head_cat0 = head_convBlock(network, weightMap, *head_stem0->getOutput(0), 128, 3, 1, 1, "head.reg_convs", "head.cls_convs", 0);
    auto head_cat1 = head_convBlock(network, weightMap, *head_stem0->getOutput(0), 128, 3, 1, 1, "head.reg_convs", "head.cls_convs", 1);
    auto head_cat2 = head_convBlock(network, weightMap, *head_stem0->getOutput(0), 128, 3, 1, 1, "head.reg_convs", "head.cls_convs", 2);


    ITensor* outputTensor[] = {head_cat0->getOutput(0), head_cat1->getOutput(0), head_cat2->getOutput(0)};
    auto output = network->addConcatenation(outputTensor, 3);
    output->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*output->getOutput(0));
    

    // 创建 engine
    std::cout<< "Building engine, please wait for a while" <<std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network,*config);
    assert(engine != nullptr);
    std::cout<< "Build engine successfully" <<std::endl;

    // 系列化模型
    std::cout<< "Serializing engine, please wait for a while" <<std::endl;
    IHostMemory* serializedModel = engine->serialize();
    assert(serializedModel != nullptr);
    std::cout<< "Serialize engine successfully" <<std::endl;

    // 释放资源
    network->destroy();
    builder->destroy();
    config->destroy();

    // 使用底层C++ API构建网络时 需要释放权重文件的内存
    for(auto& mem : weightMap){
        free((void*) (mem.second.values));
    }

    // 导入模型
    // 或者也可以将序列化的模型导出成文件
    std::cout<< "Memory released " <<std::endl;

    // create runtime
    std::cout<< "Building runtime, please wait for a while " <<std::endl;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    std::cout<< "Build runtime successfully" <<std::endl;


    // 反序列化 模型，之后engine已经hold model
    std::cout<< "Deserializing engine ,please wait for a while" <<std::endl;
    engine = runtime->deserializeCudaEngine(serializedModel->data(),serializedModel->size());
    assert(engine != nullptr);
    std::cout<< "Deserialize engine successfully" <<std::endl;
    
    // ExecutionContext 
    std::cout<< " Building context ,please wait for a while" <<std::endl;
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    std::cout<< " Builde context successfully" <<std::endl;

    // prepare the input data and outputdata
    std::cout<< " Loading test picture" <<std::endl;
    static float prob[OUTPUT_SIZE];
    cv::Mat img = cv::imread("/root/Downloads/Yolox_tensorRT/yolox_tensorRT_hand/dog.jpg");
    int img_w = img.cols;
    int img_h = img.rows;
    cv::Mat pr_img = static_resize(img);
   
    

    float* blob;
    blob = blobFromImage(pr_img);
     std::cout<< " Resized the input successfully " <<std::endl;

    // Run Inference o90p 
     std::cout<< "Running the inference " <<std::endl;
    auto start = std::chrono::system_clock::now();
    doInference(*context, blob, prob, OUTPUT_SIZE, pr_img.size());
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    for(unsigned int i = 0; i < 10; i++){
        std::cout<<prob[i]<<",";
    }
    std::cout<<std::endl;

    delete blob;
    // 释放资源 engine
    serializedModel->destroy();
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}