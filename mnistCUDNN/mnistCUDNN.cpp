/**
* Copyright 2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
 * This example demonstrates how to use CUDNN library to implement forward
 * pass. The sample loads weights and biases from trained network,
 * takes a few images of digits and recognizes them. The network was trained on 
 * the MNIST dataset using Caffe. The network consists of two 
 * convolution layers, two pooling layers, one relu and two 
 * fully connected layers. Final layer gets processed by Softmax. 
 * cublasSgemv is used to implement fully connected layers.

 * The sample can work in single, double, half precision, but it
 * assumes the data in files is stored in single precision
 */

#include <sstream>
#include <fstream>
#include <stdlib.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <vector>

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>

#include <FreeImage.h>
#include "fp16_dev.h"
#include "fp16_emu.h"
#include "gemv.h"
#include "error_util.h"

#define IMAGE_H 28
#define IMAGE_W 28

const char *first_image = "one_28x28.pgm";
const char *second_image = "three_28x28.pgm";
const char *third_image = "five_28x28.pgm";

const char *conv1_bin = "conv1.bin";
const char *conv1_bias_bin = "conv1.bias.bin";
const char *conv2_bin = "conv2.bin";
const char *conv2_bias_bin = "conv2.bias.bin";
const char *ip1_bin = "ip1.bin";
const char *ip1_bias_bin = "ip1.bias.bin";
const char *ip2_bin = "ip2.bin";
const char *ip2_bias_bin = "ip2.bias.bin";

namespace {

typedef std::chrono::high_resolution_clock HostClock;
typedef HostClock::time_point HostTimePoint;

enum ConvAlgoMode
{
    CONV_ALGO_AUTO_IMPLICIT = 0,
    CONV_ALGO_FORCE_IMPLICIT_GEMM = 1,
    CONV_ALGO_FORCE_IMPLICIT_PRECOMP_GEMM = 2
};

double elapsedMs(HostTimePoint start, HostTimePoint end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool fileExists(const std::string& path)
{
    std::ifstream f(path.c_str());
    return f.good();
}

std::string lowerString(const std::string& value)
{
    std::string lowered = value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return lowered;
}

ConvAlgoMode parseConvAlgoMode(const char* value)
{
    if (value == NULL || std::string(value).empty())
    {
        return CONV_ALGO_AUTO_IMPLICIT;
    }

    std::string mode = lowerString(value);
    if (mode == "auto" || mode == "auto_implicit")
    {
        return CONV_ALGO_AUTO_IMPLICIT;
    }
    if (mode == "implicit_gemm" || mode == "gemm")
    {
        return CONV_ALGO_FORCE_IMPLICIT_GEMM;
    }
    if (mode == "implicit_precomp_gemm" || mode == "precomp" || mode == "precomp_gemm")
    {
        return CONV_ALGO_FORCE_IMPLICIT_PRECOMP_GEMM;
    }

    std::cerr << "ERROR: unsupported conv_algo=" << value
              << " (use auto, implicit_gemm, or implicit_precomp_gemm)" << std::endl;
    exit(EXIT_FAILURE);
}

const char* convAlgoModeName(ConvAlgoMode mode)
{
    switch (mode)
    {
        case CONV_ALGO_AUTO_IMPLICIT:
            return "auto";
        case CONV_ALGO_FORCE_IMPLICIT_GEMM:
            return "implicit_gemm";
        case CONV_ALGO_FORCE_IMPLICIT_PRECOMP_GEMM:
            return "implicit_precomp_gemm";
    }
    return "unknown";
}

const char* convAlgoName(cudnnConvolutionFwdAlgo_t algo)
{
    switch (algo)
    {
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
            return "IMPLICIT_GEMM";
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
            return "IMPLICIT_PRECOMP_GEMM";
        case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
            return "GEMM";
        case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
            return "DIRECT";
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
            return "FFT";
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
            return "FFT_TILING";
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
            return "WINOGRAD";
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
            return "WINOGRAD_NONFUSED";
    }
    return "UNKNOWN";
}

std::string csvEscape(const std::string& value)
{
    std::string escaped = "\"";
    for (size_t i = 0; i < value.size(); ++i)
    {
        if (value[i] == '"')
        {
            escaped += "\"\"";
        }
        else
        {
            escaped += value[i];
        }
    }
    escaped += "\"";
    return escaped;
}

struct ImageTiming
{
    std::string image;
    int prediction;
    double total_ms;
    double file_io_preprocess_ms;
    double h2d_ms;
    double d2h_ms;
    int cuda_malloc_count;
    int cuda_free_count;

    ImageTiming()
        : prediction(-1),
          total_ms(0.0),
          file_io_preprocess_ms(0.0),
          h2d_ms(0.0),
          d2h_ms(0.0),
          cuda_malloc_count(0),
          cuda_free_count(0)
    {
    }
};

class CudaEventTimer
{
    bool active_;
    cudaEvent_t start_;
    cudaEvent_t stop_;

  public:
    explicit CudaEventTimer(bool active) : active_(active), start_(0), stop_(0)
    {
        if (active_)
        {
            checkCudaErrors(cudaEventCreate(&start_));
            checkCudaErrors(cudaEventCreate(&stop_));
        }
    }

    ~CudaEventTimer()
    {
        if (active_)
        {
            cudaEventDestroy(start_);
            cudaEventDestroy(stop_);
        }
    }

    void start()
    {
        if (active_)
        {
            checkCudaErrors(cudaEventRecord(start_, 0));
        }
    }

    double stop()
    {
        if (!active_)
        {
            return 0.0;
        }
        checkCudaErrors(cudaEventRecord(stop_, 0));
        checkCudaErrors(cudaEventSynchronize(stop_));
        float elapsed = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&elapsed, start_, stop_));
        return static_cast<double>(elapsed);
    }
};

class MnistProfiler
{
    bool enabled_;
    std::string path_;
    std::string phase_;
    std::ofstream csv_;

  public:
    MnistProfiler() : enabled_(false), phase_("run") {}

    void open(const char* path, const char* phase)
    {
        if (path == NULL || std::string(path).empty())
        {
            return;
        }
        path_ = path;
        if (phase != NULL && std::string(phase).size() > 0)
        {
            phase_ = phase;
        }
        bool write_header = !fileExists(path_);
        csv_.open(path_.c_str(), std::ios::out | std::ios::app);
        if (!csv_)
        {
            FatalError("Could not open profile CSV");
        }
        enabled_ = true;
        if (write_header)
        {
            csv_ << "record_type,phase,image,section,value_ms,prediction,detail\n";
        }
    }

    bool enabled() const
    {
        return enabled_;
    }

    void writeEnvironment(int device, int argc, char** argv)
    {
        if (!enabled_)
        {
            return;
        }

        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, device));

        std::stringstream command;
        for (int i = 0; i < argc; ++i)
        {
            if (i > 0)
            {
                command << " ";
            }
            command << argv[i];
        }

#ifdef CUDART_VERSION
        int cuda_runtime_version = CUDART_VERSION;
#else
        int cuda_runtime_version = CUDA_VERSION;
#endif

        std::stringstream detail;
        detail << "device=" << device
               << ";name=" << prop.name
               << ";compute=" << prop.major << "." << prop.minor
               << ";sms=" << prop.multiProcessorCount
               << ";global_mem_mb=" << static_cast<int>(prop.totalGlobalMem / (1024 * 1024))
               << ";cuda_runtime=" << cuda_runtime_version
               << ";cudnn_header=" << CUDNN_VERSION
               << ";cudnn_runtime=" << cudnnGetVersion()
               << ";command=" << command.str();
        writeRaw("environment", "", "runtime", 0.0, -1, detail.str());
    }

    void writeProcess(const std::string& section, double value_ms, const std::string& detail = "")
    {
        if (!enabled_)
        {
            return;
        }
        writeRaw("process", "", section, value_ms, -1, detail);
    }

    void writeOp(const std::string& image, const std::string& section, double value_ms, const std::string& detail = "")
    {
        if (!enabled_)
        {
            return;
        }
        writeRaw("op", image, section, value_ms, -1, detail);
    }

    void writeImage(const ImageTiming& timing)
    {
        if (!enabled_)
        {
            return;
        }
        std::stringstream detail;
        detail << "file_io_preprocess_ms=" << timing.file_io_preprocess_ms
               << ";host_to_device_ms=" << timing.h2d_ms
               << ";device_to_host_ms=" << timing.d2h_ms
               << ";cuda_malloc_count=" << timing.cuda_malloc_count
               << ";cuda_free_count=" << timing.cuda_free_count;
        writeRaw("image", timing.image, "latency", timing.total_ms, timing.prediction, detail.str());
    }

  private:
    void writeRaw(
        const std::string& record_type,
        const std::string& image,
        const std::string& section,
        double value_ms,
        int prediction,
        const std::string& detail)
    {
        csv_ << csvEscape(record_type) << ","
             << csvEscape(phase_) << ","
             << csvEscape(image) << ","
             << csvEscape(section) << ","
             << std::fixed << std::setprecision(6) << value_ms << ","
             << prediction << ","
             << csvEscape(detail) << "\n";
        csv_.flush();
    }
};

std::vector<std::string> readImageList(const char* list_path)
{
    std::ifstream f(list_path);
    if (!f)
    {
        FatalError("Could not open images list");
    }

    std::vector<std::string> images;
    std::string line;
    while (std::getline(f, line))
    {
        size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#')
        {
            continue;
        }
        size_t last = line.find_last_not_of(" \t\r\n");
        images.push_back(line.substr(first, last - first + 1));
    }
    return images;
}

}  // namespace

/********************************************************
 * Prints the error message, and exits
 * ******************************************************/

#define EXIT_WAIVED 0


void get_path(std::string& sFilename, const char *fname, const char *pname)
{
    sFilename = (std::string("data/") + std::string(fname));
}

// Need the map, since scaling factor is of float type in half precision
// Also when one needs to use float instead of half, e.g. for printing
template <typename T> 
struct ScaleFactorTypeMap { typedef T Type;};
template <> struct ScaleFactorTypeMap<half1>  { typedef float Type;};

// float/double <-> half conversion class
template <class value_type>
class Convert
{
public:
    template <class T>
    value_type operator()(T x) {return value_type(x);}
    value_type operator()(half1 x) {return value_type(cpu_half2float(x));}
};

template <>
class Convert<half1>
{
public:
    template <class T>
    half1 operator()(T x) {return cpu_float2half_rn (T(x));} 
    half1 operator()(half1 x) {return x;}
};

// IO utils
template <class value_type>
void readBinaryFile(const char* fname, int size, value_type* data_h)
{
    std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
    std::stringstream error_s;
    if (!dataFile)
    {
        error_s << "Error opening file " << fname; 
        FatalError(error_s.str());
    }
    // we assume the data stored is always in float precision
    float* data_tmp = new float[size];
    int size_b = size*sizeof(float);
    if (!dataFile.read ((char*) data_tmp, size_b)) 
    {
        error_s << "Error reading file " << fname; 
        FatalError(error_s.str());
    }
    // conversion
    Convert<value_type> fromReal;
    for (int i = 0; i < size; i++)
    {
        data_h[i] = fromReal(data_tmp[i]);
    }
    delete [] data_tmp;
}

template <class value_type>
void readAllocMemcpy(const char* fname, int size, value_type** data_h, value_type** data_d)
{
    *data_h = new value_type[size];

    readBinaryFile<value_type>(fname, size, *data_h);

    int size_b = size*sizeof(value_type);
    checkCudaErrors( cudaMalloc((void**)data_d, size_b) );
    checkCudaErrors( cudaMemcpy(*data_d, *data_h,
                                size_b,
                                cudaMemcpyHostToDevice) );
}

void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char *zMessage)
{
    FatalError(zMessage);
}
template <class value_type>
void readImage(const char* fname, value_type* imgData_h, bool verbose = true)
{
    // declare a host image object for an 8-bit grayscale image
    std::string sFilename(fname);
    if (verbose)
    {
        std::cout << "Loading image " << sFilename << std::endl;
    }
    // Take care of half precision
    Convert<value_type> fromReal;
    
    // load gray-scale image from disk    
    // set your own FreeImage error handler
    FreeImage_SetOutputMessage(FreeImageErrorHandler);

    FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(sFilename.c_str());

    // no signature? try to guess the file format from the file extension
    if (eFormat == FIF_UNKNOWN)
    {
        eFormat = FreeImage_GetFIFFromFilename(sFilename.c_str());
    }

    if (eFormat == FIF_UNKNOWN)
    {
        FatalError("Unknown image format");
    }
    // check that the plugin has reading capabilities ...

    FIBITMAP *pBitmap;
    if (FreeImage_FIFSupportsReading(eFormat))
    {
        pBitmap = FreeImage_Load(eFormat, sFilename.c_str());
    }

    if (pBitmap == 0)
    {
        FatalError("Error reading image");
    }
    
    // make sure this is an 8-bit single channel image
    if (FreeImage_GetColorType(pBitmap) != FIC_MINISBLACK)
    {
        FatalError("This is not 8-bit single channel imagee");    
    }
    if (FreeImage_GetBPP(pBitmap) != 8)
    {
        FatalError("This is not 8-bit single channel imagee");   
    }

    // create an ImageCPU to receive the loaded image data
    //ImageCPU_8u_C1 oImage(FreeImage_GetWidth(pBitmap), FreeImage_GetHeight(pBitmap));

    int width = FreeImage_GetWidth(pBitmap);
    int height = FreeImage_GetHeight(pBitmap);
    
    if (width != IMAGE_W || height != IMAGE_H)
    {
        FatalError("Image dimensions missmatch");
    }
    
    // Normalize image to be in range [0,1]
    for (int i = 0; i < height; ++i)
    { 
        unsigned char *pSrcLine = FreeImage_GetScanLine(pBitmap, height - i - 1);
        for (int j = 0; j < width; j++)
        {
            int idx = IMAGE_W*i + j;
            imgData_h[idx] = fromReal(*(pSrcLine + j) / double(255));
        }
    }

    FreeImage_Unload(pBitmap); 
}

template <class value_type>
void printDeviceVector(int size, value_type* vec_d)
{
    typedef typename ScaleFactorTypeMap<value_type>::Type real_type;
    value_type *vec;
    vec = new value_type[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, size*sizeof(value_type), cudaMemcpyDeviceToHost);
    Convert<real_type> toReal;
    std::cout.precision(7);
    std::cout.setf( std::ios::fixed, std:: ios::floatfield );
    for (int i = 0; i < size; i++)
    {
        std::cout << toReal(vec[i]) << " ";
    }
    std::cout << std::endl;
    delete [] vec;
}

typedef enum {
        FP16_HOST  = 0, 
        FP16_CUDA  = 1,
        FP16_CUDNN = 2
 } fp16Import_t;
template <class value_type>
struct Layer_t
{
    fp16Import_t fp16Import;
    int inputs;
    int outputs;
    // linear dimension (i.e. size is kernel_dim * kernel_dim)
    int kernel_dim;
    value_type *data_h, *data_d;
    value_type *bias_h, *bias_d;
    Layer_t() : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
                inputs(0), outputs(0), kernel_dim(0), fp16Import(FP16_HOST){};
    Layer_t(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights,
            const char* fname_bias, const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
                  : inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim)
    {
        fp16Import = _fp16Import;
        std::string weights_path, bias_path;
        if (pname != NULL)
        {
            get_path(weights_path, fname_weights, pname);
            get_path(bias_path, fname_bias, pname);
        }
        else
        {
            weights_path = fname_weights; bias_path = fname_bias;
        }
        readAllocInit(weights_path.c_str(), inputs * outputs * kernel_dim * kernel_dim, 
                        &data_h, &data_d);
        readAllocInit(bias_path.c_str(), outputs, &bias_h, &bias_d);
    }
    ~Layer_t()
    {
        if (data_h != NULL) delete [] data_h;
        if (data_d != NULL) checkCudaErrors( cudaFree(data_d) );
        if (bias_h != NULL) delete [] bias_h;
        if (bias_d != NULL) checkCudaErrors( cudaFree(bias_d) );
    }
private:
    void readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d)
    {
        readAllocMemcpy<value_type>(fname, size, data_h, data_d);
    }
};

template <>
void Layer_t<half1>::readAllocInit(const char* fname, int size, half1** data_h, half1** data_d)
{
    *data_h = new half1[size];
    int size_b = size*sizeof(half1);
    checkCudaErrors( cudaMalloc((void**)data_d, size_b) );    
    float *data_tmp_h, *data_tmp_d;

    switch(fp16Import)
    {
        case FP16_HOST :
        {
            readBinaryFile<half1>(fname, size, *data_h);
            checkCudaErrors( cudaMemcpy(*data_d, *data_h, size_b,
                                cudaMemcpyHostToDevice) );
            break;
        }
        case FP16_CUDA :
        {
            readAllocMemcpy<float>(fname, size, &data_tmp_h, &data_tmp_d);

            gpu_float2half_rn<float>(size, data_tmp_d, *data_d);

            delete [] data_tmp_h;
            checkCudaErrors( cudaFree(data_tmp_d) );
            break;
        }
        case FP16_CUDNN :
        {
            readAllocMemcpy<float>(fname, size, &data_tmp_h, &data_tmp_d);
            delete [] data_tmp_h;
            cudnnHandle_t cudnnHandle;
            cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
            checkCUDNN( cudnnCreate(&cudnnHandle) );
            checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
            checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
            checkCUDNN( cudnnSetTensor4dDescriptorEx(srcTensorDesc,
                                                CUDNN_DATA_FLOAT,
                                                1, size,
                                                1, 1,
                                                size, 1, 1, 1) );
            checkCUDNN( cudnnSetTensor4dDescriptorEx(dstTensorDesc,
                                                CUDNN_DATA_HALF,
                                                1, size,
                                                1, 1,
                                                size, 1, 1, 1) );
            float alpha = 1.0f;
            float beta = 0.0f;
            checkCUDNN( cudnnTransformTensor(cudnnHandle, &alpha,
                                             srcTensorDesc,
                                             data_tmp_d, &beta,
                                             dstTensorDesc,
                                             *data_d) );
            checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
            checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
            checkCUDNN( cudnnDestroy(cudnnHandle) );
            checkCudaErrors( cudaFree(data_tmp_d) );
            break;
        }
    }
}

// demonstrate different ways of setting tensor descriptor
//#define SIMPLE_TENSOR_DESCRIPTOR
#define ND_TENSOR_DESCRIPTOR
void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
                    cudnnTensorFormat_t& tensorFormat,
                    cudnnDataType_t& dataType,
                    int n,
                    int c,
                    int h,
                    int w)
{
#if SIMPLE_TENSOR_DESCRIPTOR
    checkCUDNN( cudnnSetTensor4dDescriptor(tensorDesc,
                                            tensorFormat,
                                            dataType,
                                            n, c,
                                            h,
                                            w ) );
#elif defined(ND_TENSOR_DESCRIPTOR)
    const int nDims = 4;
    int dimA[nDims] = {n,c,h,w};
    int strideA[nDims] = {c*h*w, h*w, w, 1};
    checkCUDNN( cudnnSetTensorNdDescriptor(tensorDesc,
                                            dataType,
                                            4,
                                            dimA,
                                            strideA ) ); 
#else
    checkCUDNN( cudnnSetTensor4dDescriptorEx(tensorDesc,
                                            dataType,
                                            n, c,
                                            h, w,
                                            c*h*w, h*w, w, 1) );
#endif
}

template <class value_type>
class network_t
{
    typedef typename ScaleFactorTypeMap<value_type>::Type scaling_type;
    struct ConvPlan
    {
        bool valid;
        int in_n;
        int in_c;
        int in_h;
        int in_w;
        int inputs;
        int outputs;
        int kernel_dim;
        cudnnConvolutionFwdAlgo_t algo;
        size_t workspace_bytes;

        ConvPlan()
            : valid(false),
              in_n(0),
              in_c(0),
              in_h(0),
              in_w(0),
              inputs(0),
              outputs(0),
              kernel_dim(0),
              algo(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM),
              workspace_bytes(0)
        {
        }
    };

    int convAlgorithm;
    ConvAlgoMode convAlgoMode;
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnPoolingDescriptor_t     poolingDesc;
    cudnnActivationDescriptor_t  activDesc;
    cudnnLRNDescriptor_t   normDesc;
    cublasHandle_t cublasHandle;
    MnistProfiler* profiler;
    ImageTiming* currentTiming;
    std::string currentImage;
    value_type* scratchA;
    value_type* scratchB;
    int scratchACapacity;
    int scratchBCapacity;
    void* workspace;
    size_t workspaceCapacity;
    std::vector<ConvPlan> convPlans;

    void createHandles()
    {
        checkCUDNN( cudnnCreate(&cudnnHandle) );
        checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
        checkCUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
        checkCUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );
        checkCUDNN( cudnnCreatePoolingDescriptor(&poolingDesc) );
        checkCUDNN( cudnnCreateActivationDescriptor(&activDesc) );
        checkCUDNN( cudnnCreateLRNDescriptor(&normDesc) );

        checkCublasErrors( cublasCreate(&cublasHandle) );
    }
    void releaseDeviceBuffer(value_type** data, int* capacity)
    {
        if (*data != NULL)
        {
            checkCudaErrors(cudaFree(*data));
            *data = NULL;
        }
        if (capacity != NULL)
        {
            *capacity = 0;
        }
    }
    void releaseWorkspace()
    {
        if (workspace != NULL)
        {
            checkCudaErrors(cudaFree(workspace));
            workspace = NULL;
        }
        workspaceCapacity = 0;
    }
    void destroyHandles()
    {
        releaseDeviceBuffer(&scratchA, &scratchACapacity);
        releaseDeviceBuffer(&scratchB, &scratchBCapacity);
        releaseWorkspace();
        checkCUDNN( cudnnDestroyLRNDescriptor(normDesc) );
        checkCUDNN( cudnnDestroyPoolingDescriptor(poolingDesc) );
        checkCUDNN( cudnnDestroyActivationDescriptor(activDesc) );
        checkCUDNN( cudnnDestroyConvolutionDescriptor(convDesc) );
        checkCUDNN( cudnnDestroyFilterDescriptor(filterDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
        checkCUDNN( cudnnDestroy(cudnnHandle) );

        checkCublasErrors( cublasDestroy(cublasHandle) );
    }
    void reserveWorkspace(size_t sizeInBytes)
    {
        if (sizeInBytes == 0)
        {
            return;
        }
        if (workspace != NULL && workspaceCapacity >= sizeInBytes)
        {
            return;
        }
        releaseWorkspace();
        checkCudaErrors(cudaMalloc(&workspace, sizeInBytes));
        workspaceCapacity = sizeInBytes;
        if (currentTiming != NULL)
        {
            currentTiming->cuda_malloc_count += 1;
        }
    }
    bool convPlanMatches(
        const ConvPlan& plan,
        const Layer_t<value_type>& conv,
        int in_n,
        int in_c,
        int in_h,
        int in_w) const
    {
        return plan.valid
            && plan.in_n == in_n
            && plan.in_c == in_c
            && plan.in_h == in_h
            && plan.in_w == in_w
            && plan.inputs == conv.inputs
            && plan.outputs == conv.outputs
            && plan.kernel_dim == conv.kernel_dim;
    }
    ConvPlan* findConvPlan(const Layer_t<value_type>& conv, int in_n, int in_c, int in_h, int in_w)
    {
        for (size_t i = 0; i < convPlans.size(); ++i)
        {
            if (convPlanMatches(convPlans[i], conv, in_n, in_c, in_h, in_w))
            {
                return &convPlans[i];
            }
        }
        return NULL;
    }
    void recordCudaOp(const char* opName, double elapsed_ms)
    {
        if (profiler != NULL && profiler->enabled())
        {
            profiler->writeOp(currentImage, opName, elapsed_ms);
        }
    }
    cudnnConvolutionFwdAlgo_t forcedConvAlgo() const
    {
        if (convAlgoMode == CONV_ALGO_FORCE_IMPLICIT_PRECOMP_GEMM)
        {
            return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        }
        return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    }
    bool benchmarkConvolutionAlgorithm(
        cudnnConvolutionFwdAlgo_t candidate,
        const value_type* filterData,
        value_type* srcData,
        value_type* dstData,
        double* elapsed_ms,
        size_t* workspace_bytes)
    {
        cudnnStatus_t status = cudnnGetConvolutionForwardWorkspaceSize(
            cudnnHandle,
            srcTensorDesc,
            filterDesc,
            convDesc,
            dstTensorDesc,
            candidate,
            workspace_bytes);
        if (status != CUDNN_STATUS_SUCCESS)
        {
            return false;
        }

        reserveWorkspace(*workspace_bytes);
        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);

        checkCudaErrors(cudaDeviceSynchronize());
        CudaEventTimer timer(true);
        timer.start();
        status = cudnnConvolutionForward(
            cudnnHandle,
            &alpha,
            srcTensorDesc,
            srcData,
            filterDesc,
            filterData,
            convDesc,
            candidate,
            workspace,
            *workspace_bytes,
            &beta,
            dstTensorDesc,
            dstData);

        if (status != CUDNN_STATUS_SUCCESS)
        {
            return false;
        }

        *elapsed_ms = timer.stop();
        return true;
    }
  public:
    network_t(
        MnistProfiler* profiler_arg = NULL,
        ConvAlgoMode conv_algo_mode_arg = CONV_ALGO_AUTO_IMPLICIT)
        : profiler(profiler_arg),
          currentTiming(NULL),
          scratchA(NULL),
          scratchB(NULL),
          scratchACapacity(0),
          scratchBCapacity(0),
          workspace(NULL),
          workspaceCapacity(0)
    {
        convAlgorithm = -1;
        convAlgoMode = conv_algo_mode_arg;
        switch (sizeof(value_type))
        {
            case 2 : dataType = CUDNN_DATA_HALF; break;
            case 4 : dataType = CUDNN_DATA_FLOAT; break;
            case 8 : dataType = CUDNN_DATA_DOUBLE; break;
            default : FatalError("Unsupported data type");
        }
        tensorFormat = CUDNN_TENSOR_NCHW;
        createHandles();    
    };
    ~network_t()
    {
        destroyHandles();
    }
    void resize(int size, value_type **data, int* capacity = NULL)
    {
        if (capacity != NULL)
        {
            if (*data != NULL && *capacity >= size)
            {
                return;
            }
            if (*data != NULL)
            {
                checkCudaErrors( cudaFree(*data) );
                if (currentTiming != NULL)
                {
                    currentTiming->cuda_free_count += 1;
                }
            }
            checkCudaErrors( cudaMalloc((void**)data, size*sizeof(value_type)) );
            *capacity = size;
            if (currentTiming != NULL)
            {
                currentTiming->cuda_malloc_count += 1;
            }
            return;
        }
        if (*data != NULL)
        {
            checkCudaErrors( cudaFree(*data) );
            if (currentTiming != NULL)
            {
                currentTiming->cuda_free_count += 1;
            }
        }
        checkCudaErrors( cudaMalloc((void**)data, size*sizeof(value_type)) );
        if (currentTiming != NULL)
        {
            currentTiming->cuda_malloc_count += 1;
        }
    }
    void setConvolutionAlgorithm(const cudnnConvolutionFwdAlgo_t& algo)
    {
        convAlgorithm = (int) algo;
    }
    void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer_t<value_type>& layer, int c, value_type *data)
    {
        setTensorDesc(biasTensorDesc, tensorFormat, dataType, 1, c, 1, 1);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(1);
        checkCUDNN( cudnnAddTensor( cudnnHandle, 
                                    &alpha, biasTensorDesc,
                                    layer.bias_d,
                                    &beta,
                                    dstTensorDesc,
                                    data) );
    }
    void fullyConnectedForward(
        const Layer_t<value_type>& ip,
        int& n,
        int& c,
        int& h,
        int& w,
        value_type* srcData,
        value_type** dstData,
        int* dstCapacity = NULL,
        const char* opName = "fully_connected")
    {
        if (n != 1)
        {
            FatalError("Not Implemented"); 
        }
        int dim_x = c*h*w;
        int dim_y = ip.outputs;
        resize(dim_y, dstData, dstCapacity);

        scaling_type alpha = scaling_type(1), beta = scaling_type(1);
        CudaEventTimer timer(profiler != NULL && profiler->enabled());
        timer.start();
        // place bias into dstData
        checkCudaErrors( cudaMemcpy(*dstData, ip.bias_d, dim_y*sizeof(value_type), cudaMemcpyDeviceToDevice) );
        
        gemv(cublasHandle, dim_x, dim_y, alpha,
                ip.data_d, srcData, beta,*dstData);
        recordCudaOp(opName, timer.stop());

        h = 1; w = 1; c = dim_y;
    }
    void convoluteForward(
        const Layer_t<value_type>& conv,
        int& n,
        int& c,
        int& h,
        int& w,
        value_type* srcData,
        value_type** dstData,
        int* dstCapacity = NULL,
        const char* opName = "convolution")
    {
        cudnnConvolutionFwdAlgo_t algo;
        const int in_n = n;
        const int in_c = c;
        const int in_h = h;
        const int in_w = w;

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);

        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        const int filterDimA[tensorDims] = {conv.outputs, conv.inputs, conv.kernel_dim, conv.kernel_dim};
                                       
        checkCUDNN( cudnnSetFilterNdDescriptor(filterDesc, dataType, CUDNN_TENSOR_NCHW,tensorDims, filterDimA) );
 
        const int convDims = 2;
        int padA[convDims] = {0,0};
        int filterStrideA[convDims] = {1,1};
        int upscaleA[convDims] = {1,1};
        cudnnDataType_t  convDataType = dataType;
        if (dataType == CUDNN_DATA_HALF) {
            convDataType = CUDNN_DATA_FLOAT; //Math are done in FP32 when tensor are in FP16
        }
        checkCUDNN( cudnnSetConvolutionNdDescriptor(convDesc, convDims, padA, filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION,convDataType) );
        // find dimension of convolution output
        checkCUDNN( cudnnGetConvolutionNdForwardOutputDim(convDesc, srcTensorDesc, filterDesc, tensorDims,tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        resize(n*c*h*w, dstData, dstCapacity);

        size_t sizeInBytes=0;
        if (convAlgorithm < 0)
        {
            ConvPlan* plan = findConvPlan(conv, in_n, in_c, in_h, in_w);
            if (plan == NULL)
            {
                checkCudaErrors(cudaDeviceSynchronize());
                HostTimePoint algo_start = HostClock::now();

                if (convAlgoMode == CONV_ALGO_AUTO_IMPLICIT)
                {
                    std::cout << "Benchmarking implicit convolution candidates for " << opName << " ...\n";
                    cudnnConvolutionFwdAlgo_t candidates[2] = {
                        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                    };

                    bool foundUsableAlgo = false;
                    double bestElapsedMs = 0.0;
                    size_t bestWorkspaceBytes = 0;
                    cudnnConvolutionFwdAlgo_t bestAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

                    for (int candidateIndex = 0; candidateIndex < 2; ++candidateIndex)
                    {
                        double candidateElapsedMs = 0.0;
                        size_t candidateWorkspaceBytes = 0;
                        cudnnConvolutionFwdAlgo_t candidate = candidates[candidateIndex];
                        bool usable = benchmarkConvolutionAlgorithm(
                            candidate,
                            conv.data_d,
                            srcData,
                            *dstData,
                            &candidateElapsedMs,
                            &candidateWorkspaceBytes);

                        std::cout << "^^^^ " << convAlgoName(candidate)
                                  << " usable=" << (usable ? "yes" : "no")
                                  << " time=" << candidateElapsedMs
                                  << " ms workspace=" << candidateWorkspaceBytes
                                  << " bytes\n";

                        if (profiler != NULL && profiler->enabled())
                        {
                            std::stringstream section;
                            section << opName << "_candidate_" << convAlgoName(candidate);
                            std::stringstream detail;
                            detail << "usable=" << (usable ? "yes" : "no")
                                   << ";algo=" << static_cast<int>(candidate)
                                   << ";algo_name=" << convAlgoName(candidate)
                                   << ";workspace_bytes=" << candidateWorkspaceBytes;
                            profiler->writeOp(currentImage, section.str(), candidateElapsedMs, detail.str());
                        }

                        if (usable && (!foundUsableAlgo || candidateElapsedMs < bestElapsedMs))
                        {
                            foundUsableAlgo = true;
                            bestElapsedMs = candidateElapsedMs;
                            bestWorkspaceBytes = candidateWorkspaceBytes;
                            bestAlgo = candidate;
                        }
                    }

                    if (!foundUsableAlgo)
                    {
                        FatalError("No usable implicit cuDNN convolution algorithm found");
                    }

                    algo = bestAlgo;
                    sizeInBytes = bestWorkspaceBytes;
                }
                else
                {
                    algo = forcedConvAlgo();
                    cudnnStatus_t status = cudnnGetConvolutionForwardWorkspaceSize(
                        cudnnHandle,
                        srcTensorDesc,
                        filterDesc,
                        convDesc,
                        dstTensorDesc,
                        algo,
                        &sizeInBytes);
                    if (status != CUDNN_STATUS_SUCCESS)
                    {
                        std::stringstream error_s;
                        error_s << "Forced cuDNN convolution algorithm is not usable: "
                                << convAlgoName(algo);
                        FatalError(error_s.str());
                    }
                    std::cout << "Using forced convolution algorithm for " << opName
                              << ": " << convAlgoName(algo)
                              << " workspace=" << sizeInBytes
                              << " bytes\n";
                }

                checkCudaErrors(cudaDeviceSynchronize());
                double algo_select_ms = elapsedMs(algo_start, HostClock::now());

                ConvPlan newPlan;
                newPlan.valid = true;
                newPlan.in_n = in_n;
                newPlan.in_c = in_c;
                newPlan.in_h = in_h;
                newPlan.in_w = in_w;
                newPlan.inputs = conv.inputs;
                newPlan.outputs = conv.outputs;
                newPlan.kernel_dim = conv.kernel_dim;
                newPlan.algo = algo;
                newPlan.workspace_bytes = sizeInBytes;
                convPlans.push_back(newPlan);

                if (profiler != NULL && profiler->enabled())
                {
                    std::stringstream section;
                    section << opName << "_algorithm_select";
                    std::stringstream detail;
                    detail << "algo=" << static_cast<int>(algo)
                           << ";algo_name=" << convAlgoName(algo)
                           << ";mode=" << convAlgoModeName(convAlgoMode)
                           << ";workspace_bytes=" << sizeInBytes;
                    profiler->writeOp(currentImage, section.str(), algo_select_ms, detail.str());
                }
            }
            else
            {
                algo = plan->algo;
                sizeInBytes = plan->workspace_bytes;
            }
        }
        // MY CODE END
        else
        {
            algo = (cudnnConvolutionFwdAlgo_t)convAlgorithm;
            std::cout << "Using explicitly set convolution algorithm for " << opName
                      << ": " << convAlgoName(algo) << "\n";
            checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,srcTensorDesc,filterDesc,convDesc,dstTensorDesc,algo,&sizeInBytes) );
        }

        reserveWorkspace(sizeInBytes);
        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);

        CudaEventTimer convTimer(profiler != NULL && profiler->enabled());
        convTimer.start();
        checkCUDNN( cudnnConvolutionForward(cudnnHandle,&alpha,srcTensorDesc,srcData,filterDesc,conv.data_d,convDesc,algo,workspace,sizeInBytes,&beta,dstTensorDesc,*dstData) );
        recordCudaOp(opName, convTimer.stop());

        std::string biasOpName = std::string(opName) + "_bias";
        CudaEventTimer biasTimer(profiler != NULL && profiler->enabled());
        biasTimer.start();
        addBias(dstTensorDesc, conv, c, *dstData);
        recordCudaOp(biasOpName.c_str(), biasTimer.stop());
    }

    void poolForward( int& n, int& c, int& h, int& w,
                      value_type* srcData, value_type** dstData,
                      int* dstCapacity = NULL,
                      const char* opName = "pool")
    {
        const int poolDims = 2;
        int windowDimA[poolDims] = {2,2};
        int paddingA[poolDims] = {0,0};
        int strideA[poolDims] = {2,2};
        checkCUDNN( cudnnSetPoolingNdDescriptor(poolingDesc,CUDNN_POOLING_MAX,CUDNN_PROPAGATE_NAN,poolDims,windowDimA,paddingA,strideA ) );

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);        

        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        checkCUDNN( cudnnGetPoolingNdForwardOutputDim(poolingDesc,srcTensorDesc,tensorDims,tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);  
     
        resize(n*c*h*w, dstData, dstCapacity);
        scaling_type alpha = scaling_type(1);
        scaling_type beta = scaling_type(0);
        CudaEventTimer timer(profiler != NULL && profiler->enabled());
        timer.start();
        checkCUDNN( cudnnPoolingForward(cudnnHandle,poolingDesc,&alpha,srcTensorDesc,srcData,&beta,dstTensorDesc,*dstData) );
        recordCudaOp(opName, timer.stop());
    }
    void softmaxForward(
        int n,
        int c,
        int h,
        int w,
        value_type* srcData,
        value_type** dstData,
        int* dstCapacity = NULL,
        const char* opName = "softmax")
    {
        resize(n*c*h*w, dstData, dstCapacity);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        CudaEventTimer timer(profiler != NULL && profiler->enabled());
        timer.start();
        checkCUDNN( cudnnSoftmaxForward(cudnnHandle,CUDNN_SOFTMAX_ACCURATE ,CUDNN_SOFTMAX_MODE_CHANNEL,&alpha,srcTensorDesc,srcData,&beta,dstTensorDesc,*dstData) );
        recordCudaOp(opName, timer.stop());
    }
    void lrnForward(
        int n,
        int c,
        int h,
        int w,
        value_type* srcData,
        value_type** dstData,
        int* dstCapacity = NULL,
        const char* opName = "lrn")
    {
        unsigned lrnN = 5;
        double lrnAlpha, lrnBeta, lrnK;
        lrnAlpha = 0.0001; lrnBeta = 0.75; lrnK = 1.0;
        checkCUDNN( cudnnSetLRNDescriptor(normDesc,lrnN,lrnAlpha,lrnBeta,lrnK) );

        resize(n*c*h*w, dstData, dstCapacity);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        CudaEventTimer timer(profiler != NULL && profiler->enabled());
        timer.start();
        checkCUDNN( cudnnLRNCrossChannelForward(cudnnHandle,normDesc,CUDNN_LRN_CROSS_CHANNEL_DIM1,&alpha,srcTensorDesc,srcData,&beta,dstTensorDesc,*dstData) );
        recordCudaOp(opName, timer.stop());
    }
    void activationForward(
        int n,
        int c,
        int h,
        int w,
        value_type* srcData,
        value_type** dstData,
        int* dstCapacity = NULL,
        const char* opName = "relu")
    {
        checkCUDNN( cudnnSetActivationDescriptor(activDesc,CUDNN_ACTIVATION_RELU,CUDNN_PROPAGATE_NAN,0.0) );
    
        resize(n*c*h*w, dstData, dstCapacity);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        CudaEventTimer timer(profiler != NULL && profiler->enabled());
        timer.start();
        checkCUDNN( cudnnActivationForward(cudnnHandle, activDesc, &alpha, srcTensorDesc, srcData, &beta, dstTensorDesc, *dstData) );    
        recordCudaOp(opName, timer.stop());
    }

    int classify_example(const char* fname, const Layer_t<value_type>& conv1,
                          const Layer_t<value_type>& conv2,
                          const Layer_t<value_type>& ip1,
                          const Layer_t<value_type>& ip2,
                          bool verbose = true,
                          ImageTiming* outTiming = NULL)
    {
        int n,c,h,w;
        value_type *srcData = NULL, *dstData = NULL;
        value_type imgData_h[IMAGE_H*IMAGE_W];
        ImageTiming timing;
        timing.image = fname;
        currentTiming = &timing;
        currentImage = fname;

        HostTimePoint io_start = HostClock::now();
        readImage(fname, imgData_h, verbose);
        timing.file_io_preprocess_ms = elapsedMs(io_start, HostClock::now());

        if (verbose)
        {
            std::cout << "Performing forward propagation ...\n";
        }

        resize(IMAGE_H*IMAGE_W, &scratchA, &scratchACapacity);
        CudaEventTimer h2dTimer(profiler != NULL && profiler->enabled());
        h2dTimer.start();
        checkCudaErrors( cudaMemcpy(scratchA, imgData_h,
                                    IMAGE_H*IMAGE_W*sizeof(value_type),
                                    cudaMemcpyHostToDevice) );
        timing.h2d_ms = h2dTimer.stop();
        if (profiler != NULL && profiler->enabled())
        {
            profiler->writeOp(currentImage, "host_to_device_image", timing.h2d_ms);
        }

        srcData = scratchA;
        n = c = 1; h = IMAGE_H; w = IMAGE_W;
        HostTimePoint infer_start = HostClock::now();
        convoluteForward(conv1, n, c, h, w, srcData, &scratchB, &scratchBCapacity, "conv1");
        dstData = scratchB;
        poolForward(n, c, h, w, dstData, &scratchA, &scratchACapacity, "pool1");
        srcData = scratchA;

        convoluteForward(conv2, n, c, h, w, srcData, &scratchB, &scratchBCapacity, "conv2");
        dstData = scratchB;
        poolForward(n, c, h, w, dstData, &scratchA, &scratchACapacity, "pool2");
        srcData = scratchA;

        fullyConnectedForward(ip1, n, c, h, w, srcData, &scratchB, &scratchBCapacity, "fc1");
        dstData = scratchB;
        activationForward(n, c, h, w, dstData, &scratchA, &scratchACapacity, "relu");
        srcData = scratchA;
        lrnForward(n, c, h, w, srcData, &scratchB, &scratchBCapacity, "lrn");
        dstData = scratchB;

        fullyConnectedForward(ip2, n, c, h, w, dstData, &scratchA, &scratchACapacity, "fc2");
        srcData = scratchA;
        softmaxForward(n, c, h, w, srcData, &scratchB, &scratchBCapacity, "softmax");
        dstData = scratchB;
        checkCudaErrors(cudaDeviceSynchronize());
        HostTimePoint infer_end = HostClock::now();

        const int max_digits = 10;
        // Take care of half precision
        Convert<scaling_type> toReal;
        value_type result[max_digits];
        CudaEventTimer d2hTimer(profiler != NULL && profiler->enabled());
        d2hTimer.start();
        checkCudaErrors( cudaMemcpy(result, dstData, max_digits*sizeof(value_type), cudaMemcpyDeviceToHost) );
        timing.d2h_ms = d2hTimer.stop();
        if (profiler != NULL && profiler->enabled())
        {
            profiler->writeOp(currentImage, "device_to_host_result", timing.d2h_ms);
        }
        int id = 0;
        for (int i = 1; i < max_digits; i++)
        {
            if (toReal(result[id]) < toReal(result[i])) id = i;
        }

        if (verbose)
        {
            std::cout << "Resulting weights from Softmax:" << std::endl;
            printDeviceVector(n*c*h*w, dstData);
        }

        timing.prediction = id;
        timing.total_ms = elapsedMs(infer_start, infer_end);
        if (profiler != NULL && profiler->enabled())
        {
            profiler->writeImage(timing);
        }
        if (outTiming != NULL)
        {
            *outTiming = timing;
        }
        currentTiming = NULL;
        currentImage.clear();
        return id;
    }
};

#if !defined(CUDA_VERSION) || (CUDA_VERSION <= 7000)
// using 1x1 convolution to emulate gemv in half precision when cuBLAS version <= 7.0
template <>
void network_t<half1>::fullyConnectedForward(const Layer_t<half1>& ip,
                          int& n, int& c, int& h, int& w,
                          half1* srcData, half1** dstData,
                          int* dstCapacity,
                          const char* opName)
{
    c = c*h*w; h = 1; w = 1;
    network_t<half1>::convoluteForward(ip, n, c, h, w, srcData, dstData, dstCapacity, opName);
    c = ip.outputs;
}
#endif

void displayUsage()
{
    printf( "mnistCUDNN {<options>}\n");
	printf( "help                   : display this help\n");
	printf( "device=<int>           : set the device to run the sample\n");
	printf( "image=<name>           : classify specific image\n");
	printf( "images=<listfile>      : classify newline-separated images with one model load\n");
	printf( "profile=<csv>          : append timing details to CSV\n");
	printf( "phase=<name>           : label profile rows, e.g. baseline or optimized\n");
	printf( "conv_algo=<mode>       : auto, implicit_gemm, or implicit_precomp_gemm\n");
}
 
 void printExecutionTime(
    std::chrono::high_resolution_clock::time_point start)
{
    checkCudaErrors(cudaDeviceSynchronize());

    auto end =
        std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double,std::milli>(
            end-start).count();

    std::cout << "\n================================\n";
    std::cout << "Total execution time : "
              << elapsed_ms
              << " ms\n";

    std::cout << "Total execution time : "
              << elapsed_ms/1000.0
              << " sec\n";

    std::cout << "================================\n";
}
 
	int main(int argc, char *argv[])
	{
	    auto program_start =
	        std::chrono::high_resolution_clock::now();
	    std::string image_path;
	    int i1,i2,i3;
	    char* profile_name = NULL;
	    char* phase_name = NULL;
	    char* conv_algo_name = NULL;
	    if (checkCmdLineFlag(argc, (const char **)argv, "profile"))
	    {
	        getCmdLineArgumentString(argc, (const char **)argv,
	                                 "profile", (char **) &profile_name);
	    }
	    if (checkCmdLineFlag(argc, (const char **)argv, "phase"))
	    {
	        getCmdLineArgumentString(argc, (const char **)argv,
	                                 "phase", (char **) &phase_name);
	    }
	    if (checkCmdLineFlag(argc, (const char **)argv, "conv_algo"))
	    {
	        getCmdLineArgumentString(argc, (const char **)argv,
	                                 "conv_algo", (char **) &conv_algo_name);
	    }
	    const char* profile_env = getenv("MNIST_PROFILE_CSV");
	    const char* phase_env = getenv("MNIST_PROFILE_PHASE");
	    const char* conv_algo_env = getenv("MNIST_CONV_ALGO");
	    ConvAlgoMode convAlgoMode = parseConvAlgoMode(
	        conv_algo_name != NULL ? conv_algo_name : conv_algo_env);
	    MnistProfiler profiler;

	    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
	    {
        displayUsage();
        exit(EXIT_WAIVED); 
    }

    int version = (int)cudnnGetVersion();
    printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d (%s)\n", version, CUDNN_VERSION, CUDNN_VERSION_STR);
    printf("Host compiler version : %s %s\r", COMPILER_NAME, COMPILER_VER);
    showDevices();

    int device = 0;
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        device = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        checkCudaErrors( cudaSetDevice(device) );
	    }
	    std::cout << "Using device " << device << std::endl;
	    profiler.open(profile_name != NULL ? profile_name : profile_env,
	                  phase_name != NULL ? phase_name : phase_env);
	    profiler.writeEnvironment(device, argc, argv);

	    if (checkCmdLineFlag(argc, (const char **)argv, "image"))
	    {
	        char* image_name;
	        getCmdLineArgumentString(argc, (const char **)argv,
	                                 "image", (char **) &image_name);

	        HostTimePoint model_start = HostClock::now();
	        network_t<float> mnist(&profiler, convAlgoMode);
	        double handle_init_ms = elapsedMs(model_start, HostClock::now());
	        HostTimePoint weights_start = HostClock::now();
	        Layer_t<float> conv1(1,20,5,conv1_bin,conv1_bias_bin,argv[0]);
	        Layer_t<float> conv2(20,50,5,conv2_bin,conv2_bias_bin,argv[0]);
	        Layer_t<float>   ip1(800,500,1,ip1_bin,ip1_bias_bin,argv[0]);
	        Layer_t<float>   ip2(500,10,1,ip2_bin,ip2_bias_bin,argv[0]);
	        double weights_load_ms = elapsedMs(weights_start, HostClock::now());
	        profiler.writeProcess("model_handle_init", handle_init_ms);
	        profiler.writeProcess("weights_load_and_h2d", weights_load_ms);
	        profiler.writeProcess("model_init_and_weights", handle_init_ms + weights_load_ms);

	        ImageTiming timing;
	        int i1 = mnist.classify_example(image_name, conv1, conv2, ip1, ip2, true, &timing);
	        std::cout << "\nResult of classification: " << i1 << std::endl;
	        std::cout << "Inference latency: " << std::fixed << std::setprecision(6)
	                  << timing.total_ms << " ms" << std::endl;

	        checkCudaErrors(cudaDeviceSynchronize());
	        profiler.writeProcess("overall_execution", elapsedMs(program_start, HostClock::now()));
	        printExecutionTime(program_start);
	        cudaDeviceReset();
	        exit(EXIT_SUCCESS);
	    }

	    if (checkCmdLineFlag(argc, (const char **)argv, "images"))
	    {
	        char* images_name;
	        getCmdLineArgumentString(argc, (const char **)argv,
	                                 "images", (char **) &images_name);
	        std::vector<std::string> images = readImageList(images_name);

	        HostTimePoint model_start = HostClock::now();
	        network_t<float> mnist(&profiler, convAlgoMode);
	        double handle_init_ms = elapsedMs(model_start, HostClock::now());
	        HostTimePoint weights_start = HostClock::now();
	        Layer_t<float> conv1(1,20,5,conv1_bin,conv1_bias_bin,argv[0]);
	        Layer_t<float> conv2(20,50,5,conv2_bin,conv2_bias_bin,argv[0]);
	        Layer_t<float>   ip1(800,500,1,ip1_bin,ip1_bias_bin,argv[0]);
	        Layer_t<float>   ip2(500,10,1,ip2_bin,ip2_bias_bin,argv[0]);
	        double weights_load_ms = elapsedMs(weights_start, HostClock::now());
	        profiler.writeProcess("model_handle_init", handle_init_ms);
	        profiler.writeProcess("weights_load_and_h2d", weights_load_ms);
	        profiler.writeProcess("model_init_and_weights", handle_init_ms + weights_load_ms);

	        HostTimePoint batch_start = HostClock::now();
	        double total_latency_ms = 0.0;
	        for (size_t i = 0; i < images.size(); ++i)
	        {
	            ImageTiming timing;
	            int prediction = mnist.classify_example(images[i].c_str(), conv1, conv2, ip1, ip2, false, &timing);
	            total_latency_ms += timing.total_ms;
	            std::cout << "MNIST_RESULT input=" << images[i]
	                      << " prediction=" << prediction
	                      << " latency_ms=" << std::fixed << std::setprecision(6)
	                      << timing.total_ms << std::endl;
	        }
	        double batch_ms = elapsedMs(batch_start, HostClock::now());
	        std::stringstream detail;
	        detail << "images=" << images.size()
	               << ";sum_image_latency_ms=" << total_latency_ms
	               << ";avg_image_latency_ms="
	               << (images.empty() ? 0.0 : total_latency_ms / static_cast<double>(images.size()));
	        profiler.writeProcess("batch_classification_loop", batch_ms, detail.str());
	        checkCudaErrors(cudaDeviceSynchronize());
	        profiler.writeProcess("overall_execution", elapsedMs(program_start, HostClock::now()));

	        printExecutionTime(program_start);
	        cudaDeviceReset();
	        exit(EXIT_SUCCESS);
	    }

    // default behaviour
    if (argc == 1 || (argc == 2) && checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        // check available memory
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties( &prop, device ));
        double globalMem = prop.totalGlobalMem/double(1024*1024);
        bool low_memory = false;
        if (globalMem < 1536) 
        {
        // takes care of 1x1 convolution workaround for fully connected layers
        // when CUDNN_CONVOLUTION_FWD_ALGO_FFT is used
#if !defined(CUDA_VERSION) || (CUDA_VERSION <= 7000)
            low_memory = true;
#endif
        }
        {
            std::cout << "\nTesting single precision\n";
            network_t<float> mnist(NULL, convAlgoMode);
            Layer_t<float> conv1(1,20,5,conv1_bin,conv1_bias_bin,argv[0]);
            Layer_t<float> conv2(20,50,5,conv2_bin,conv2_bias_bin,argv[0]);
            Layer_t<float>   ip1(800,500,1,ip1_bin,ip1_bias_bin,argv[0]);
            Layer_t<float>   ip2(500,10,1,ip2_bin,ip2_bias_bin,argv[0]);
            get_path(image_path, first_image, argv[0]);
            i1 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);
            
            get_path(image_path, second_image, argv[0]);
            i2 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);
            
            get_path(image_path, third_image, argv[0]);
            // New feature in cuDNN v3: FFT for convolution
            //mnist.setConvolutionAlgorithm(CUDNN_CONVOLUTION_FWD_ALGO_FFT);
            i3 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);

            std::cout << "\nResult of classification: " << i1 << " " << i2 << " " << i3 << std::endl;
            if (i1 != 1 || i2 != 3 || i3 != 5)
            {
                std::cout << "\nTest failed!\n";
                FatalError("Prediction mismatch");
            }
            else
            {
                std::cout << "\nTest passed!\n";
            }
        }

        {
            std::cout << "\nTesting half precision (math in single precision)\n";
            network_t<half1> mnist(NULL, convAlgoMode);
            // Conversion of input weights to half precision is done
            // on host using tools from fp16_emu.cpp
            Layer_t<half1> conv1(1,20,5,conv1_bin,conv1_bias_bin,argv[0],FP16_HOST);
            Layer_t<half1> conv2(20,50,5,conv2_bin,conv2_bias_bin,argv[0],FP16_HOST);
            // Conversion of input weights to half precision is done
            // on device using cudnnTransformTensor
            Layer_t<half1>   ip1(800,500,1,ip1_bin,ip1_bias_bin,argv[0], FP16_CUDNN);
            // Conversion of input weights to half precision is done
            // on device using CUDA kernel from fp16_dev.cu
            Layer_t<half1>   ip2(500,10,1,ip2_bin,ip2_bias_bin,argv[0], FP16_CUDA);
            get_path(image_path, first_image, argv[0]);
            i1 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);
            
            get_path(image_path, second_image, argv[0]);
            i2 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);
            
            get_path(image_path, third_image, argv[0]);
            // New feature in cuDNN v3: FFT for convolution
            if (!low_memory)
            {
                //mnist.setConvolutionAlgorithm(CUDNN_CONVOLUTION_FWD_ALGO_FFT);
            }
            i3 = mnist.classify_example(image_path.c_str(), conv1, conv2, ip1, ip2);

            std::cout << "\nResult of classification: " << i1 << " " << i2 << " " << i3 << std::endl;
            if (i1 != 1 || i2 != 3 || i3 != 5)
            {
                std::cout << "\nTest failed!\n";
                FatalError("Prediction mismatch");
            }
            else
            {
                std::cout << "\nTest passed!\n";
            }
        }

        printExecutionTime(program_start);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);        
    }

    displayUsage();
    printExecutionTime(program_start);
    cudaDeviceReset();
    exit(EXIT_WAIVED);
}
