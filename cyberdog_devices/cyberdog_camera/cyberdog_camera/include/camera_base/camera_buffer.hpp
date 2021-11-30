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

#ifndef CAMERA_BASE__CAMERA_BUFFER_HPP_
#define CAMERA_BASE__CAMERA_BUFFER_HPP_

#include <nvmmapi/NvNativeBuffer.h>
#include <NvBuffer.h>

namespace cyberdog_camera
{

using Argus::Buffer;
using Argus::IBuffer;
using ArgusSamples::NvNativeBuffer;

/*
   Helper class to map NvNativeBuffer to Argus::Buffer and vice versa.
   A reference to DmaBuffer will be saved as client data in each Argus::Buffer.
   Also DmaBuffer will keep a reference to corresponding Argus::Buffer.
   This class also extends NvBuffer to act as a share buffer between Argus and V4L2 encoder.
*/
class DmaBuffer : public NvNativeBuffer, public NvBuffer
{
public:
  /* Always use this static method to create DmaBuffer */
  static DmaBuffer * create(
    const Argus::Size2D<uint32_t> & size,
    NvBufferColorFormat colorFormat,
    NvBufferLayout layout = NvBufferLayout_Pitch)
  {
    DmaBuffer * buffer = new DmaBuffer(size);
    if (!buffer) {
      return NULL;
    }

    if (NvBufferCreate(&buffer->m_fd, size.width(), size.height(), layout, colorFormat)) {
      delete buffer;
      return NULL;
    }

    /* save the DMABUF fd in NvBuffer structure */
    buffer->planes[0].fd = buffer->m_fd;
    /* byteused must be non-zero for a valid buffer */
    buffer->planes[0].bytesused = 1;

    return buffer;
  }

  /* Help function to convert Argus Buffer to DmaBuffer */
  static DmaBuffer * fromArgusBuffer(Buffer * buffer)
  {
    IBuffer * iBuffer = Argus::interface_cast<IBuffer>(buffer);
    const DmaBuffer * dmabuf = static_cast<const DmaBuffer *>(iBuffer->getClientData());

    return const_cast<DmaBuffer *>(dmabuf);
  }

  /* Return DMA buffer handle */
  int getFd() const {return m_fd;}

  /* Get and set reference to Argus buffer */
  void setArgusBuffer(Buffer * buffer) {m_buffer = buffer;}
  Buffer * getArgusBuffer() const {return m_buffer;}

private:
  explicit DmaBuffer(const Argus::Size2D<uint32_t> & size)
  : NvNativeBuffer(size),
    NvBuffer(0, 0),
    m_buffer(NULL)
  {
  }

  Buffer * m_buffer;     /* Reference to Argus::Buffer */
};

struct ImageBuffer
{
  Argus::Size2D<uint32_t> res;
  void * data;
  int fd;
  size_t size;
  struct timespec timestamp;
  void * priv;
};

}  // namespace cyberdog_camera

#endif  // CAMERA_BASE__CAMERA_BUFFER_HPP_
