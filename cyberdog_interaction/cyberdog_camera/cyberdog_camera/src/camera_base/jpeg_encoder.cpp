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

#define LOG_TAG "JpegEncoder"
#include "camera_base/jpeg_encoder.hpp"
#include "camera_utils/log.hpp"

namespace cyberdog_camera
{

JpegEncoder::JpegEncoder(int width, int height, const char * name)
{
  m_jpegEncoder = NvJPEGEncoder::createJPEGEncoder(name);
  if (!m_jpegEncoder) {
    CAM_ERR("Failed to create JPEGEncoder.");
  }

  // yuv420 image size
  m_bufferSize = width * height * 3 / 2;
  m_outputBuffer = new unsigned char[m_bufferSize];
  if (!m_outputBuffer) {
    CAM_ERR("Failed to malloc output buffer.");
  }
}

JpegEncoder::~JpegEncoder()
{
  if (m_jpegEncoder) {
    delete m_jpegEncoder;
  }

  if (m_outputBuffer) {
    delete[] m_outputBuffer;
  }
}

void JpegEncoder::setCropRect(
  uint32_t left,
  uint32_t top, uint32_t width, uint32_t height)
{
  m_jpegEncoder->setCropRect(left, top, width, height);
}

size_t JpegEncoder::encodeFromFd(int fd, unsigned char ** data, int quality)
{
  if (!m_jpegEncoder || !m_outputBuffer) {
    CAM_ERR("Encoder not initialized.");
    return 0;
  }

  size_t size = m_bufferSize;
  unsigned char * buffer = m_outputBuffer;
  m_jpegEncoder->encodeFromFd(fd, JCS_YCbCr, &buffer, size, quality);
  *data = buffer;

  return size;
}

}  // namespace cyberdog_camera
