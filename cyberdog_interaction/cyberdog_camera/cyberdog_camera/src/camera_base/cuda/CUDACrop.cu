// Copyright (c) 2021  Beijing Xiaomi Mobile Software Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "camera_base/cuda/CUDACrop.hpp"

#define CUDA(x)	cudaCheckError((x), #x, __FILE__, __LINE__)

inline __device__ __host__ int iDivUp(int a, int b ) { return (a % b != 0) ? (a / b + 1) : (a / b); }

template<typename T>
__global__ void gpuCrop( T* input, T* output, int offsetX, int offsetY,
					int inWidth, int outWidth, int outHeight )
{
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if( out_x >= outWidth || out_y >= outHeight )
        return;

    const int in_x = out_x + offsetX;
    const int in_y = out_y + offsetY;

    output[out_y * outWidth + out_x] = input[in_y * inWidth + in_x];
}

template<typename T>
static cudaError_t launchCrop( T* input, T* output, const int4& roi, size_t inputWidth, size_t inputHeight )
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( inputWidth == 0 || inputHeight == 0 )
        return cudaErrorInvalidValue;

    // get the ROI/output dimensions
    const int outputWidth = roi.z - roi.x;
    const int outputHeight = roi.w - roi.y;

    // validate the requested ROI
    if( outputWidth <= 0 || outputHeight <= 0 )
        return cudaErrorInvalidValue;

    if( outputWidth > inputWidth || outputHeight > inputHeight )
        return cudaErrorInvalidValue;

    if( roi.x < 0 || roi.y < 0 || roi.z < 0 || roi.w < 0 )
        return cudaErrorInvalidValue;

    if( roi.z >= inputWidth || roi.w >= inputHeight )
        return cudaErrorInvalidValue;

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

    gpuCrop<T><<<gridDim, blockDim>>>(input, output, roi.x, roi.y, inputWidth, outputWidth, outputHeight);

    cudaDeviceSynchronize();

    return cudaGetLastError();
}

cudaError_t cudaCrop(uchar3* input, uchar3* output, const int4& roi, size_t inputWidth, size_t inputHeight )
{
    return launchCrop<uchar3>(input, output, roi, inputWidth, inputHeight);
}
