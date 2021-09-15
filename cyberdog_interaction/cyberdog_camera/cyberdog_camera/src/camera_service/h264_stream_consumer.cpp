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

#define LOG_TAG "H264Stream"
#include <string>
#include <memory>
#include <vector>
#include <utility>
#include "camera_service/h264_stream_consumer.hpp"
#include "camera_service/camera_manager.hpp"
#include "camera_service/ncs_client.hpp"
#include "camera_algo/algo_dispatcher.hpp"
#include "camera_utils/log.hpp"

namespace cyberdog_camera
{

H264StreamConsumer::H264StreamConsumer(Size2D<uint32_t> size)
: StreamConsumer(size),
  m_videoEncoder(NULL)
{
  auto qos = rclcpp::QoS(rclcpp::QoSInitialization(RMW_QOS_POLICY_HISTORY_KEEP_LAST, 10));
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  m_h264Publisher =
    CameraManager::getInstance()->create_publisher<sensor_msgs::msg::CompressedImage>(
    "h264_video", qos);

  m_bodySub = CameraManager::getInstance()->create_subscription<BodyInfoT>(
    "body", qos,
    std::bind(&H264StreamConsumer::bodySubCallback, this, std::placeholders::_1));

  // for video
  m_video_stream = std::make_shared<live_stream::VideoStream>(
    std::string("rtmp://127.0.0.1:1935/live"));
}

H264StreamConsumer::~H264StreamConsumer()
{
}

bool H264StreamConsumer::threadInitialize()
{
  if (!StreamConsumer::threadInitialize()) {
    return false;
  }

  m_videoEncoder = new VideoEncoder(
    "enc",
    m_size.width(), m_size.height(), V4L2_PIX_FMT_H264);

  if (!m_videoEncoder) {
    CAM_ERR("failed to create VideoEncoder\n");
    return false;
  }

  m_nvosdContext = nvosd_create_context();
  if (!m_nvosdContext) {
    CAM_ERR("Failed to create nvosd context");
  }

  // for video
  m_video_stream->configure(m_size.width(), m_size.height());
  if ((m_video_stream->initDecoder()) < 0) {
    CAM_ERR("Failed to init video decoder");
  }

  m_videoEncoder->setInputDoneCallback(inputDoneCallback, this);
  m_videoEncoder->setOutputDoneCallback(outputDoneCallback, this);

  video_encoder_settings settings;
  settings.bitrate = 1 * 1024 * 1024;
  settings.profile = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
  settings.insert_sps = true;

  m_videoEncoder->initialize(settings);
  startSoundTimer();

#ifdef USE_SOFT_ENC
  m_video_stream->startWork();
#endif
  return true;
}

bool H264StreamConsumer::threadShutdown()
{
#ifdef USE_SOFT_ENC
  m_video_stream->reset();
#endif
  stopSoundTimer();
  m_videoEncoder->encodeFromFd(-1);
  m_videoEncoder->shutdown();

  if (m_nvosdContext) {
    nvosd_destroy_context(m_nvosdContext);
  }

  delete m_videoEncoder;
  m_videoEncoder = NULL;

#ifndef USE_SOFT_ENC
  m_video_stream->reset();
#endif
  return StreamConsumer::threadShutdown();
}

bool H264StreamConsumer::processBuffer(Buffer * buffer)
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);

  if (!buffer) {
    return false;
  }

  float frameRate;
  if (getFrameRate(frameRate)) {
    CAM_INFO("%.2f frames per second", frameRate);
  }

  DmaBuffer * dma_buf = DmaBuffer::fromArgusBuffer(buffer);
  int fd = dma_buf->getFd();

  if (m_videoEncoder) {
    m_videoEncoder->encodeFromFd(fd);
  }

#ifdef USE_SOFT_ENC
  ImageBuffer buf;
  memset(&buf, 0, sizeof(buf));
  buf.res = m_size;
  buf.fd = fd;
  buf.timestamp = ts;

  publishImage(m_frameCount, buf);
#endif

  m_frameCount++;

  return true;
}

void H264StreamConsumer::drawBodyRectangles(int fd)
{
  m_rectLock.lock();
  std::vector<NvOSD_RectParams> rectParams;
  for (size_t i = 0; i < m_currentBodyRects.size(); i++) {
    NvOSD_RectParams param;
    param.left = m_currentBodyRects[i].roi.x_offset;
    param.top = m_currentBodyRects[i].roi.y_offset;
    param.width = m_currentBodyRects[i].roi.width;
    param.height = m_currentBodyRects[i].roi.height;
    param.border_width = 7;
    if (m_currentBodyRects[i].reid.length() > 0) {
      param.border_color.red = 1.0f;
      param.border_color.green = 0.0;
      param.border_color.blue = 0.0;
    } else {
      param.border_color.red = 0.0;
      param.border_color.green = 0.0;
      param.border_color.blue = 1.0f;
    }
    rectParams.push_back(param);
  }
  m_rectLock.unlock();

  if (rectParams.size() > 0) {
    nvosd_draw_rectangles(m_nvosdContext, MODE_HW, fd, rectParams.size(), rectParams.data());
  }
}

void H264StreamConsumer::inputDoneCallback(int fd)
{
  Argus::Buffer * buffer = getBufferFromFd(fd);

  if (buffer) {
    bufferDone(buffer);
  }
}

void H264StreamConsumer::outputDoneCallback(uint8_t * data, size_t size, int64_t ts)
{
  publishH264Image(data, size, ts);
}

void H264StreamConsumer::publishImage(uint64_t frame_id, ImageBuffer & buf)
{
  (void)frame_id;
  (void)buf;
#ifdef USE_SOFT_ENC
  int height = buf.res.height();
  int width = buf.res.width();
  int64_t size = height * width * 3 / 2;
  uint8_t * data = new uint8_t[size];
  memset(data, 0, size * sizeof(uint8_t));
  NvBuffer2Raw(buf.fd, 0, width, height, &data[0]);
  NvBuffer2Raw(
    buf.fd, 1, width / 2, height / 2,
    &data[width * height]);
  if ((m_video_stream->emptyThisBuffer(data, size)) < 0) {
    CAM_ERR("%s,send buffer to video", __func__);
  }
  delete[] data;
#endif
}

void H264StreamConsumer::publishH264Image(uint8_t * data, size_t size, int64_t timestamp)
{
  auto msg = std::make_unique<sensor_msgs::msg::CompressedImage>();
  msg->header.stamp.sec = timestamp / (1000 * 1000);
  msg->header.stamp.nanosec = (timestamp % (1000 * 1000)) * 1000;
  msg->format = "h264";
  msg->data.resize(size);
  memcpy(&msg->data[0], data, size);

#ifndef USE_SOFT_ENC
  if ((m_video_stream->emptyThisBuffer(data, size, timestamp)) < 0) {
    CAM_ERR("%s,send buffer to video ", __func__);
  }
#endif
  m_h264Publisher->publish(std::move(msg));
}

void H264StreamConsumer::bodySubCallback(const BodyInfoT::SharedPtr msg)
{
  bool has_reid = false;
  int reid_index = 0;
  for (size_t i = 0; i < msg->infos.size(); i++) {
    if (msg->infos[i].reid.length() > 0) {
      has_reid = true;
      reid_index = i;
      break;
    }
  }

  m_rectLock.lock();
  m_currentBodyRects.clear();
  if (has_reid) {
    m_currentBodyRects.push_back(msg->infos[reid_index]);
  } else if (!AlgoDispatcher::getInstance().isBodyTracked()) {
    m_currentBodyRects = msg->infos;
  }
  m_rectLock.unlock();
}

bool H264StreamConsumer::startSoundTimer()
{
  struct sigevent evp;
  struct itimerspec ts;
  memset(&evp, 0, sizeof(struct sigevent));
  memset(&ts, 0, sizeof(struct itimerspec));

  evp.sigev_value.sival_ptr = this;
  evp.sigev_notify = SIGEV_THREAD;
  evp.sigev_notify_function = playVideoSound;
  if (timer_create(CLOCK_MONOTONIC, &evp, &m_timer) != 0) {
    CAM_ERR("create sound playing timer failed.");
    return false;
  }

  ts.it_value.tv_sec = 1;
  ts.it_value.tv_nsec = 0;
  ts.it_interval.tv_sec = 120;
  ts.it_interval.tv_nsec = 0;
  if (timer_settime(m_timer, 0, &ts, NULL) != 0) {
    CAM_ERR("start sound playing timer failed.");
    return false;
  }
  CAM_INFO("Start preview sound timer.");

  return true;
}

void H264StreamConsumer::stopSoundTimer()
{
  CAM_INFO("Stop preview sound timer.");
  timer_delete(m_timer);
}

void H264StreamConsumer::playVideoSound(union sigval val)
{
  (void)val;
  NCSClient::getInstance().play(SoundRecording);
}

}  // namespace cyberdog_camera
