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

#ifndef CAMERA_BASE__CUDA__CUDACROP_HPP_
#define CAMERA_BASE__CUDA__CUDACROP_HPP_

#include <cuda_runtime.h>
#include <cuda.h>

cudaError_t cudaCrop(
  uchar3 * input, uchar3 * output, const int4 & roi, size_t inputWidth,
  size_t inputHeight);

#endif  // CAMERA_BASE__CUDA__CUDACROP_HPP_
