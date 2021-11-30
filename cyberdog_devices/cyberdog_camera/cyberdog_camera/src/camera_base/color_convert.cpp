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

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <nvbuf_utils.h>
#include <iostream>
#include "camera_base/color_convert.hpp"

namespace cyberdog_camera
{

ColorConvert::ColorConvert(int width, int height)
: m_width(width),
  m_height(height)
{
  m_size = width * height * 3;
}

ColorConvert::~ColorConvert()
{
}

int ColorConvert::initialze(int fd)
{
  int ret = 0;

  ret = mapEglFrame(fd);
  if (ret) {
    printf("Failed to map egl frame\n");
    goto end;
  }

  ret = createCudaContext();
  if (ret) {
    printf("Failed to create cuda buffer and stream\n");
    goto unmap;
  }

  return ret;

unmap:
  unmapEglFrame();
end:
  return ret;
}

void ColorConvert::release()
{
  unmapEglFrame();
  destroyCudaContext();
}

int ColorConvert::convertRGBAToBGR(void * out)
{
  int ret = 0;
  struct timeval start, end;

  gettimeofday(&start, NULL);

  if (m_eglCtx.eglFrame.frameType == cudaEglFrameTypePitch) {
    cudaConvertIntPackedToFloatPlanar(
      m_eglCtx.eglFrame,
      m_width,
      m_height,
      COLOR_FORMAT_BGR,           // can be COLOR_FORMAT_RGB
      m_cudaBuf,            // cuda buffer pointer
      m_cudaStream);            // CUDA stream
    cudaStreamSynchronize(m_cudaStream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA err %d (%s)\n", err, cudaGetErrorString(err) );
      ret = -1;
    }
  }

  cudaMemcpy(out, m_cudaBuf, m_size, cudaMemcpyDeviceToHost);

  gettimeofday(&end, NULL);

  return ret;
}

int ColorConvert::mapEglFrame(int fd)
{
  cudaError_t status;

  m_eglCtx.egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (m_eglCtx.egl_display == EGL_NO_DISPLAY) {
    std::cerr << "Error while get EGL display connection" << std::endl;
    return -1;
  }

  if (!eglInitialize(m_eglCtx.egl_display, nullptr, nullptr)) {
    std::cerr << "Erro while initialize EGL display connection" << std::endl;
    return -1;
  }

  m_eglCtx.egl_khr_image = NvEGLImageFromFd(m_eglCtx.egl_display, fd);
  if (m_eglCtx.egl_khr_image == nullptr) {
    std::cerr << "Error while mapping dmabuf fd (" << fd << ") to EGLImage" << std::endl;
    return -1;
  }

  status = cudaGraphicsEGLRegisterImage(
    &m_eglCtx.pResource, m_eglCtx.egl_khr_image,
    cudaGraphicsRegisterFlagsNone);
  if (status != cudaSuccess) {
    std::cerr << "cuGraphicsEGLRegisterImage failed: " << status <<
      " cuda process stop" << std::endl;
    return -1;
  }

  status = cudaGraphicsResourceGetMappedEglFrame(&m_eglCtx.eglFrame, m_eglCtx.pResource, 0, 0);
  if (status != cudaSuccess) {
    std::cerr << "cuGraphicsSubResourceGetMappedArray failed" << std::endl;
    return -1;
  }

  cudaDeviceSynchronize();

  return 0;
}

void ColorConvert::unmapEglFrame()
{
  NvDestroyEGLImage(m_eglCtx.egl_display, m_eglCtx.egl_khr_image);
  cudaGraphicsUnregisterResource(m_eglCtx.pResource);
}

int ColorConvert::createCudaContext()
{
  cudaError_t err;

  err = cudaStreamCreate(&m_cudaStream);
  if (err != cudaSuccess) {
    fprintf(
      stderr, "failed to create cuda stream, err = %d (%s)!\n",
      err, cudaGetErrorString(err));
    return err;
  }

  err = cudaMalloc(reinterpret_cast<void **>(&m_cudaBuf), m_size);
  if (err != cudaSuccess) {
    fprintf(
      stderr, "failed to allocate cuda buffer, err = %d (%s)!\n",
      err, cudaGetErrorString(err));
    return err;
  }

  return 0;
}

void ColorConvert::destroyCudaContext()
{
  if (m_cudaBuf) {
    cudaFree(m_cudaBuf);
  }
  cudaStreamDestroy(m_cudaStream);
}

}  // namespace cyberdog_camera
