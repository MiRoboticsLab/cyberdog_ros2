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

#define LOG_TAG "VideoEncoder"
#include <set>
#include "camera_base/video_encoder.hpp"
#include "camera_utils/log.hpp"

#define MAX_QUEUED_BUFFERS (2)

namespace cyberdog_camera
{

VideoEncoder::VideoEncoder(
  const char * name, int width,
  int height, uint32_t pixfmt)
: m_name(name),
  m_width(width),
  m_height(height),
  m_pixfmt(pixfmt)
{
  m_VideoEncoder = NULL;
  m_outputDoneCb = NULL;
}

VideoEncoder::~VideoEncoder()
{
  if (m_VideoEncoder) {
    delete m_VideoEncoder;
  }
}

bool VideoEncoder::initialize(video_encoder_settings & settings)
{
  // Create encoder
  if (!createVideoEncoder(settings)) {
    CAM_ERR("Could not create encoder.");
    return false;
  }

  // Stream on
  if (m_VideoEncoder->output_plane.setStreamStatus(true) < 0) {
    CAM_ERR("Failed to stream on output plane");
    return false;
  }
  if (m_VideoEncoder->capture_plane.setStreamStatus(true) < 0) {
    CAM_ERR("Failed to stream on capture plane");
    return false;
  }

  // Set DQ callback
  m_VideoEncoder->capture_plane.setDQThreadCallback(encoderCapturePlaneDqCallback);

  m_VideoEncoder->capture_plane.startDQThread(this);

  // Enqueue all the empty capture plane buffers
  for (uint32_t i = 0; i < m_VideoEncoder->capture_plane.getNumBuffers(); i++) {
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];

    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

    v4l2_buf.index = i;
    v4l2_buf.m.planes = planes;

    m_VideoEncoder->capture_plane.qBuffer(v4l2_buf, NULL);
  }

  return true;
}

bool VideoEncoder::encodeFromFd(int dmabuf_fd)
{
  struct v4l2_buffer v4l2_buf;
  struct v4l2_plane planes[MAX_PLANES];

  memset(&v4l2_buf, 0, sizeof(v4l2_buf));
  memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));
  v4l2_buf.m.planes = planes;

  if (m_VideoEncoder->output_plane.getNumQueuedBuffers() < MAX_QUEUED_BUFFERS) {
    v4l2_buf.index = m_VideoEncoder->output_plane.getNumQueuedBuffers();
    v4l2_buf.m.planes[0].m.fd = dmabuf_fd;
    v4l2_buf.m.planes[0].bytesused = 1;     // byteused must be non-zero
    if (m_VideoEncoder->output_plane.qBuffer(v4l2_buf, NULL) < 0) {
      m_VideoEncoder->abort();
      CAM_ERR("Failed to qBuffer.");
      return false;
    }
    m_dmabufFdSet.insert(dmabuf_fd);
  } else {
    if (m_VideoEncoder->output_plane.dqBuffer(v4l2_buf, NULL, NULL, 10) < 0) {
      m_VideoEncoder->abort();
      CAM_ERR("Failed to dqBuffer.");
      return false;
    }
    // Buffer done, execute callback
    m_inputDonecb(v4l2_buf.m.planes[0].m.fd, m_inputCbArg);
    m_dmabufFdSet.erase(v4l2_buf.m.planes[0].m.fd);

    if (dmabuf_fd < 0) {
      // Send EOS
      v4l2_buf.m.planes[0].bytesused = 0;
    } else {
      v4l2_buf.m.planes[0].m.fd = dmabuf_fd;
      v4l2_buf.m.planes[0].bytesused = 1;       // byteused must be non-zero
      m_dmabufFdSet.insert(dmabuf_fd);
    }

    if (m_VideoEncoder->output_plane.qBuffer(v4l2_buf, NULL) < 0) {
      m_VideoEncoder->abort();
      CAM_ERR("Failed to qBuffer.");
      return false;
    }
  }

  return true;
}

bool VideoEncoder::shutdown()
{
  // Wait till capture plane DQ Thread finishes
  // i.e. all the capture plane buffers are dequeued
  m_VideoEncoder->capture_plane.waitForDQThread(2000);

  // Return all queued buffers in output plane
  assert(m_dmabufFdSet.size() == MAX_QUEUED_BUFFERS - 1);   // EOS buffer
  // is not in the set
  for (std::set<int>::iterator it = m_dmabufFdSet.begin();
    it != m_dmabufFdSet.end(); it++)
  {
    m_inputDonecb(*it, m_inputCbArg);
  }
  m_dmabufFdSet.clear();

  if (m_VideoEncoder) {
    delete m_VideoEncoder;
    m_VideoEncoder = NULL;
  }

  return false;
}

bool VideoEncoder::createVideoEncoder(video_encoder_settings & settings)
{
  int ret = 0;

  m_VideoEncoder = NvVideoEncoder::createVideoEncoder(m_name.c_str());
  if (!m_VideoEncoder) {
    CAM_ERR("Could not create m_VideoEncoderoder");
    return false;
  }

  ret = m_VideoEncoder->setCapturePlaneFormat(
    m_pixfmt, m_width,
    m_height, 2 * 1024 * 1024);
  if (ret < 0) {
    CAM_ERR("Could not set capture plane format");
  }

  ret = m_VideoEncoder->setOutputPlaneFormat(
    V4L2_PIX_FMT_YUV420M, m_width,
    m_height);
  if (ret < 0) {
    CAM_ERR("Could not set output plane format");
  }

  ret = m_VideoEncoder->setBitrate(settings.bitrate);
  if (ret < 0) {
    CAM_ERR("Could not set bitrate");
  }

  m_VideoEncoder->setMaxPerfMode(1);

  if (m_pixfmt == V4L2_PIX_FMT_H264) {
    ret = m_VideoEncoder->setProfile(settings.profile);
  } else {
    ret = m_VideoEncoder->setProfile(V4L2_MPEG_VIDEO_H265_PROFILE_MAIN);
  }
  if (ret < 0) {
    CAM_ERR("Could not set m_VideoEncoderoder profile");
  }

  if (m_pixfmt == V4L2_PIX_FMT_H264) {
    ret = m_VideoEncoder->setLevel(V4L2_MPEG_VIDEO_H264_LEVEL_5_0);
    if (ret < 0) {
      CAM_ERR("Could not set m_VideoEncoderoder level");
    }
  }

  ret = m_VideoEncoder->setRateControlMode(V4L2_MPEG_VIDEO_BITRATE_MODE_CBR);
  if (ret < 0) {
    CAM_ERR("Could not set rate control mode");
  }

  ret = m_VideoEncoder->setIFrameInterval(30);
  if (ret < 0) {
    CAM_ERR("Could not set I-frame interval");
  }

  ret = m_VideoEncoder->setFrameRate(30, 1);
  if (ret < 0) {
    CAM_ERR("Could not set m_VideoEncoderoder framerate");
  }

  ret = m_VideoEncoder->setInsertSpsPpsAtIdrEnabled(settings.insert_sps);
  if (ret < 0) {
    CAM_ERR("Could not set InsertSpsPpsAtIdr");
  }

  // Query, Export and Map the output plane buffers so that we can read
  // raw data into the buffers
  ret = m_VideoEncoder->output_plane.setupPlane(V4L2_MEMORY_DMABUF, 10, true, false);
  if (ret < 0) {
    CAM_ERR("Could not setup output plane");
  }

  // Query, Export and Map the capture plane buffers so that we can write
  // encoded data from the buffers
  ret = m_VideoEncoder->capture_plane.setupPlane(V4L2_MEMORY_MMAP, 6, true, false);
  if (ret < 0) {
    CAM_ERR("Could not setup capture plane");
  }

  return true;
}

bool VideoEncoder::encoderCapturePlaneDqCallback(
  struct v4l2_buffer * v4l2_buf,
  NvBuffer * buffer,
  NvBuffer * shared_buffer)
{
  (void)shared_buffer;
  if (!v4l2_buf) {
    m_VideoEncoder->abort();
    CAM_ERR("Failed to dequeue buffer from capture plane");
    return false;
  }

  struct timeval tv;
  gettimeofday(&tv, NULL);
  int64_t timestamp = (int64_t)tv.tv_sec * 1000 * 1000 + tv.tv_usec;

  if (m_outputDoneCb) {
    m_outputDoneCb(
      reinterpret_cast<uint8_t *>(buffer->planes[0].data), buffer->planes[0].bytesused, timestamp,
      m_outputCbArg);
  }

  m_VideoEncoder->capture_plane.qBuffer(*v4l2_buf, NULL);

  // GOT EOS from encoder. Stop dqthread.
  if (buffer->planes[0].bytesused == 0) {
    return false;
  }

  return true;
}

}  // namespace cyberdog_camera
