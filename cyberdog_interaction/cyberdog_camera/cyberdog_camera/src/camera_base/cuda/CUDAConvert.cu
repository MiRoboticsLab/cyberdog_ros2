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

#include <cuda.h>
#include <cuda_runtime.h>
#include "camera_base/cuda/cuda_convert.hpp"

static __global__ void
convertIntRGBAPackedToFloatBGRPlanar(void *pDevPtr, int width, int height,
                                     void* cudaBuf, int pitch)
{
    uint8_t *pData = (uint8_t *)cudaBuf;
    char *pSrcData = (char *)pDevPtr;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        // For V4L2_PIX_FMT_ABGR32 --> RGBA-8-8-8-8
        for (int k = 0; k < 3; k++)
        {
            pData[(width * row + col) * 3 + k] =
                (uint8_t)(*(pSrcData + row * pitch + col * 4 + (3 - 1 - k)));
        }
    }
}

static __global__ void
convertIntRGBAPackedToFloatRGBPlanar(void *pDevPtr, int width, int height,
                void* cudaBuf, int pitch)
{
    float *pData = (float *)cudaBuf;
    char *pSrcData = (char *)pDevPtr;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height)
    {
        // For V4L2_PIX_FMT_ABGR32 --> RGBA-8-8-8-8
        for (int k = 0; k < 3; k++)
        {
            pData[(width * row + col) * 3 + k] =
                (uint8_t)(*(pSrcData + row * pitch + col * 4 + k));
        }
    }
}

static int convertIntPackedToFloatPlanar(void *pDevPtr,
                      int width,
                      int height,
                      int pitch,
                      COLOR_FORMAT colorFormat,
                      void* cudaBuf, void* pStream)
{
    dim3 threadsPerBlock(32, 32);
    dim3 blocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
    cudaStream_t stream;

    if (pStream != NULL)
    {
        stream = *(cudaStream_t*)pStream;
    }
    else
    {
        fprintf(stderr, "better not to run on default CUDA stream!\n");
        stream = 0;
    }

    if (colorFormat == COLOR_FORMAT_RGB)
    {
        convertIntRGBAPackedToFloatRGBPlanar<<<blocks, threadsPerBlock, 0, stream>>>(pDevPtr, width,
                height, cudaBuf, pitch);
    }
    else if (colorFormat == COLOR_FORMAT_BGR)
    {
        convertIntRGBAPackedToFloatBGRPlanar<<<blocks, threadsPerBlock, 0, stream>>>(pDevPtr, width,
                height, cudaBuf, pitch);
    }

    return 0;
}

//!
//! \details call the CUDA kernel to convert one BGRA packed frame to
//!          RGB or BGR planar frame
//!
//! \param eglFrame eglImage that is mapping to the BGRA packed frame
//!
//! \param width width of the frame
//!
//! \param height height of the frame
//!
//! \param colorFormat format of output frame, i.e. RGB or BGR
//!
//! \param cudaBuf CUDA buffer for the output frame
//!
//! \param offsets mean value from inference
//!
//! \param scales scale the float for following inference
//!
void cudaConvertIntPackedToFloatPlanar(cudaEglFrame eglFrame, int width, int height,
                            COLOR_FORMAT colorFormat,
                            void* cudaBuf,
                            cudaStream_t stream)
{
    if (eglFrame.frameType == cudaEglFrameTypePitch)
    {
        convertIntPackedToFloatPlanar((void *) eglFrame.frame.pPitch[0].ptr,
                          width,
                          height,
                          eglFrame.frame.pPitch[0].pitch,
                          colorFormat,
                          cudaBuf,
                          &stream);
    }
}
