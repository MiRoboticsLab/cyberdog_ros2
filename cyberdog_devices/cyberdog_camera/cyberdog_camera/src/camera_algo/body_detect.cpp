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

#define LOG_TAG "BodyDetect"
#include <functional>
#include <algorithm>
#include <vector>
#include <memory>
#include <utility>
#include "camera_algo/body_detect.hpp"
#include "camera_service/camera_manager.hpp"
#include "camera_utils/utils.hpp"
#include "camera_base/cuda/CUDACrop.hpp"
#include "camera_utils/log.hpp"

namespace cyberdog_camera
{

#ifdef HAVE_VISION_LIB
BodyDetect::BodyDetect()
: AlgorithmBase(),
  m_reidEnabled(false),
  m_unMatchCount(0),
  m_id(0),
  m_isTracked(false),
  m_inProcess(false)
{
  CAM_INFO();
}

BodyDetect::~BodyDetect()
{
  CAM_INFO();
  TimePerf perf(__FUNCTION__);

  std::vector<SingleBodyInfo> res;
  ImageBuffer dummy;
  memset(&dummy, 0, sizeof(dummy));
  processBodyResult(dummy, res);

  delete m_bodyDetector;
  m_featDetector->destroy();
}

bool BodyDetect::init()
{
  TimePerf perf("body detect init");
  m_bodyDetector = BodyDetector::create();
  m_featDetector = FeatureDetector::create();

  rclcpp::ServicesQoS qos;
  pub_ = CameraManager::getInstance()->create_publisher<BodyInfoT>("body", qos);

  return true;
}

double get_iou(const SingleBodyInfo & b1, const SingleBodyInfo & b2)
{
  int w = std::max(
    std::min(
      (b1.rect.left + b1.rect.width),
      (b2.rect.left + b2.rect.width)) - std::max(
      b1.rect.left,
      b2.rect.left), (uint32_t)0);
  int h = std::max(
    std::min(
      (b1.rect.top + b1.rect.height),
      (b2.rect.top + b2.rect.height)) - std::max(b1.rect.top, b2.rect.top),
    (uint32_t)0);

  return w * h / static_cast<double>(b1.rect.width * b1.rect.height +
         b2.rect.width * b2.rect.height - w * h);
}

bool BodyDetect::getBodyFeatures(ImageBuffer & frame, std::vector<SingleBodyInfo> & bodys)
{
  if (frame.data == NULL) {
    return false;
  }
  if (bodys.size() == 0) {
    if (m_reidEnabled) {m_unMatchCount++;}
    return true;
  }

  cudaError_t err;
  void * input_host, * input_cuda;
  void * output_host, * output_cuda;

  err = cudaHostAlloc(&input_host, frame.res.width() * frame.res.height() * 3, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    CAM_ERR("Failed to malloc cuda input buffer");
    return false;
  }

  err = cudaHostGetDevicePointer(&input_cuda, input_host, 0);
  if (err != cudaSuccess) {
    CAM_ERR("Failed to map cuda input buffer");
    cudaFreeHost(input_host);
    return false;
  }
  memcpy(input_host, frame.data, frame.res.width() * frame.res.height() * 3);

  // Get match object according to body selected
  int matchedId = -1;
  if (m_reidEnabled && m_reidObject.feats.empty()) {
    double maxScore = 0.5;
    for (size_t i = 0; i < bodys.size(); ++i) {
      double score = get_iou(m_reidObject, bodys[i]);
      if (score > maxScore) {
        maxScore = score;
        matchedId = i;
      }
    }
    printf("Matched id according to body selected: %d \n", matchedId);
  }

  int reidBody = -1;
  double reidScore = 0.0;
  std::vector<float> reidFeat;
  for (size_t i = 0; i < bodys.size(); i++) {
    err = cudaHostAlloc(
      &output_host, bodys[i].rect.width * bodys[i].rect.height * 3,
      cudaHostAllocMapped);
    if (err != cudaSuccess) {
      CAM_ERR("Failed to malloc cuda output buffer");
      cudaFreeHost(input_host);
      return false;
    }
    err = cudaHostGetDevicePointer(&output_cuda, output_host, 0);
    if (err != cudaSuccess) {
      CAM_ERR("Failed to map cuda output buffer");
      cudaFreeHost(input_host);
      cudaFreeHost(output_host);
      return false;
    }

    const int4 crop_roi = make_int4(
      bodys[i].rect.left,
      bodys[i].rect.top,
      bodys[i].rect.left + bodys[i].rect.width,
      bodys[i].rect.top + bodys[i].rect.height);
    cudaCrop(
      reinterpret_cast<uchar3 *>(input_cuda),
      reinterpret_cast<uchar3 *>(output_cuda),
      crop_roi, frame.res.width(), frame.res.height());

    std::vector<float> feats;
    feats.resize(m_featDetector->getFeatureSize());
    ImageBuffer buf;
    buf.res = Size2D<uint32_t>(bodys[i].rect.width, bodys[i].rect.height);
    buf.data = output_host;
    buf.size = buf.res.width() * buf.res.height() * 3;

    {
      TimePerf perf("extract");
      m_featDetector->extract(buf, feats);
    }

    if (!m_reidObject.feats.empty()) {
      double score = m_featDetector->getMatchScore(
        m_reidObject.feats, feats,
        m_reidObject.feats.size() / 128);
      printf(
        "score = %lf, bbox: %d,%d,%d,%d\n", score,
        bodys[i].rect.left, bodys[i].rect.top,
        bodys[i].rect.width, bodys[i].rect.height);
      if (score > reidScore) {
        reidScore = score;
        reidBody = i;
        reidFeat.assign(feats.begin(), feats.end());
      }
    } else if (static_cast<int>(i) == matchedId) {
      printf("Find matched id in detection, get tracked feats.\n");
      m_reidObject.feats.insert(m_reidObject.feats.end(), feats.begin(), feats.end());
      m_reidObject.id = std::to_string(m_id);
    }

    cudaFreeHost(output_host);
  }

  if (m_reidEnabled) {
    if (reidScore > 0.8) {
      m_unMatchCount = 0;
      bodys[reidBody].id = m_reidObject.id;
      if (reidScore > 0.85) {
        if (m_reidObject.feats.size() / 128 > 15) {
          m_reidObject.feats.erase(m_reidObject.feats.begin(), m_reidObject.feats.begin() + 128);
        }
        m_reidObject.feats.insert(m_reidObject.feats.end(), reidFeat.begin(), reidFeat.end());
        printf("Find matched, update tracker feats.\n");
      }
      m_isTracked = true;
    } else {
      m_unMatchCount++;
      if (m_unMatchCount > 300) {
        printf("Timeout without match, delete tracker. \n");
        m_reidObject.feats.clear();
        m_reidEnabled = false;
        m_unMatchCount = 0;
        m_isTracked = false;
      }
    }
  }

  cudaFreeHost(input_host);

  return true;
}

bool BodyDetect::processImageBuffer(ImageBuffer & buffer)
{
  if (buffer.data == NULL) {
    return false;
  }

  std::vector<SingleBodyInfo> res;

  m_inProcess = true;
  {
    TimePerf perf("bodydetect");
    m_bodyDetector->detect(buffer, res);
  }

  for (size_t i = 0; i < res.size(); i++) {
    res[i].id = "";
  }

  if (m_reidEnabled) {
    getBodyFeatures(buffer, res);
  }

  processBodyResult(buffer, res);

  m_inProcess = false;
  bufferDone(buffer);

  return true;
}

void BodyDetect::processBodyResult(ImageBuffer & frame, std::vector<SingleBodyInfo> & bodys)
{
  auto msg = std::make_unique<BodyInfoT>();

  msg->header.frame_id = "0";
  msg->header.stamp.sec = frame.timestamp.tv_sec;
  msg->header.stamp.nanosec = frame.timestamp.tv_nsec;
  msg->count = bodys.size();
  msg->infos.resize(msg->count);

  for (size_t i = 0; i < msg->count; i++) {
    BodyT & info = msg->infos[i];
    info.roi.x_offset = bodys[i].rect.left;
    info.roi.y_offset = bodys[i].rect.top;
    info.roi.width = bodys[i].rect.width;
    info.roi.height = bodys[i].rect.height;
    info.reid = bodys[i].id;
    printf(
      "===%lu body %u, %u, %u, %u id: %s\n", i, bodys[i].rect.left, bodys[i].rect.top,
      bodys[i].rect.width, bodys[i].rect.height, bodys[i].id.c_str());
  }

  pub_->publish(std::move(msg));
}

#endif

}  // namespace cyberdog_camera
