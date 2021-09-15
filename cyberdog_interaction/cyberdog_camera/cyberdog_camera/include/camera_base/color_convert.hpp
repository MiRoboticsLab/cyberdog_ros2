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

#ifndef CAMERA_BASE__COLOR_CONVERT_HPP_
#define CAMERA_BASE__COLOR_CONVERT_HPP_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>
#include <EGL/egl.h>
#include <cuda_egl_interop.h>
#include "./cuda/cuda_convert.hpp"

namespace cyberdog_camera
{

typedef struct
{
  EGLDisplay egl_display;
  cudaEglFrame eglFrame;
  EGLImageKHR egl_khr_image;
  cudaGraphicsResource_t pResource;
} egl_context;

class ColorConvert
{
public:
  ColorConvert(int width, int height);
  ~ColorConvert();

  int initialze(int fd);
  void release();
  int convertRGBAToBGR(void * out);

private:
  int mapEglFrame(int fd);
  void unmapEglFrame();
  int createCudaContext();
  void destroyCudaContext();

  int m_width;
  int m_height;
  size_t m_size;
  void * m_cudaBuf;
  cudaStream_t m_cudaStream;
  egl_context m_eglCtx;
};

}  // namespace cyberdog_camera

#endif  // CAMERA_BASE__COLOR_CONVERT_HPP_
