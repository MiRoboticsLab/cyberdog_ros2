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

#define LOG_TAG "VideoStream"
#include <string>
#include "camera_service/video_stream_consumer.hpp"
#include "camera_service/ncs_client.hpp"
#include "camera_utils/log.hpp"
#include "camera_utils/utils.hpp"

namespace cyberdog_camera
{

VideoStreamConsumer::VideoStreamConsumer(Size2D<uint32_t> size, std::string & filename)
: StreamConsumer(size),
  m_videoEncoder(NULL),
  m_filename(filename),
  m_startTs(0)
{
}

VideoStreamConsumer::~VideoStreamConsumer()
{
}

bool VideoStreamConsumer::threadInitialize()
{
  if (!StreamConsumer::threadInitialize()) {
    return false;
  }

  std::string path = "/home/mi/Camera/" + m_filename;
  m_videoMuxer = VideoMuxer::create(m_size.width(), m_size.height(), path.c_str());

  m_videoEncoder = new VideoEncoder(
    "video-enc",
    m_size.width(), m_size.height(), V4L2_PIX_FMT_H264);
  if (!m_videoEncoder) {
    CAM_ERR("failed to create VideoEncoder\n");
    return false;
  }

  m_videoEncoder->setInputDoneCallback(inputDoneCallback, this);
  m_videoEncoder->setOutputDoneCallback(outputDoneCallback, this);

  video_encoder_settings settings;
  settings.bitrate = 4 * 1024 * 1024;
  settings.profile = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH;
  settings.insert_sps = false;

  m_videoEncoder->initialize(settings);

  return true;
}

bool VideoStreamConsumer::threadShutdown()
{
  m_videoEncoder->encodeFromFd(-1);
  m_videoEncoder->shutdown();

  m_videoMuxer->release();
  delete m_videoMuxer;
  m_videoMuxer = NULL;

  delete m_videoEncoder;
  m_videoEncoder = NULL;

  stopSoundTimer();

  return StreamConsumer::threadShutdown();
}

bool VideoStreamConsumer::processBuffer(Buffer * buffer)
{
  if (!buffer) {
    return false;
  }

  if (m_frameCount == 0) {
    struct timeval ts;
    gettimeofday(&ts, NULL);
    m_startTs = ts.tv_sec * 1000 + ts.tv_usec / 1000;

    startSoundTimer();
  }

  float frameRate;
  if (getFrameRate(frameRate)) {
    CAM_INFO("%.2f frames per second", frameRate);
  }

  DmaBuffer * dma_buf = DmaBuffer::fromArgusBuffer(buffer);
  int fd = dma_buf->getFd();

  m_videoEncoder->encodeFromFd(fd);

  m_frameCount++;

  return true;
}


uint64_t VideoStreamConsumer::getRecordingTime()
{
  struct timeval ts;

  if (m_startTs == 0) {
    return 0;
  }

  gettimeofday(&ts, NULL);
  return (ts.tv_sec * 1000 + ts.tv_usec / 1000) - m_startTs;
}

void VideoStreamConsumer::inputDoneCallback(int fd)
{
  Argus::Buffer * buffer = getBufferFromFd(fd);

  if (buffer) {
    bufferDone(buffer);
  }
}

void VideoStreamConsumer::outputDoneCallback(uint8_t * data, size_t size, int64_t ts)
{
  m_videoMuxer->processData(data, size, ts);
}

bool VideoStreamConsumer::startSoundTimer()
{
  struct sigevent evp;
  struct itimerspec ts;
  memset(&evp, 0, sizeof(struct sigevent));
  memset(&ts, 0, sizeof(struct itimerspec));

  evp.sigev_value.sival_ptr = this;
  evp.sigev_notify = SIGEV_THREAD;
  evp.sigev_notify_function = playVideoSound;
  if (timer_create(CLOCK_MONOTONIC, &evp, &m_videoTimer) != 0) {
    CAM_ERR("create sound playing timer failed.");
    return false;
  }

  ts.it_value.tv_sec = 1;
  ts.it_value.tv_nsec = 0;
  ts.it_interval.tv_sec = 30;
  ts.it_interval.tv_nsec = 0;
  if (timer_settime(m_videoTimer, 0, &ts, NULL) != 0) {
    CAM_ERR("start sound playing timer failed.");
    return false;
  }
  CAM_INFO("Start video sound timer.");

  return true;
}

void VideoStreamConsumer::stopSoundTimer()
{
  CAM_INFO("Stop video sound timer.");
  timer_delete(m_videoTimer);
}

void VideoStreamConsumer::playVideoSound(union sigval val)
{
  (void)val;
  NCSClient::getInstance().play(SoundRecording);
}

}  // namespace cyberdog_camera
