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

#define LOG_TAG "FaceDetect"
#include <unistd.h>
#include <sys/stat.h>
#include <cmath>
#include <memory>
#include <vector>
#include <utility>
#include <string>
#include "camera_algo/face_detect.hpp"
#include "camera_service/camera_manager.hpp"
#include "camera_service/ncs_client.hpp"
#include "camera_utils/utils.hpp"
#include "camera_utils/log.hpp"

namespace cyberdog_camera
{

#ifdef HAVE_VISION_LIB
static const int FACE_ROI_X = 320;
static const int FACE_ROI_Y = 240;
static const int FACE_ROI_WIDTH = 640;
static const int FACE_ROI_HEIGHT = 480;

// key value to judge whether face is legal or not.
static const float FACE_NUMBER_STABLE_VAL = 0.0f;
static const float FACE_POSE_STABLE_THRES = 2.0f;
static const float FACE_POSE_YAW_LEGAL_THRES = 20.0f;
static const float FACE_POSE_PITCH_LEGAL_THRES = 10.0f;
static const float FACE_POSE_ROW_LEGAL_THRES = 20.0f;
static const float FACE_AREA_STABLE_THRES = 0.0005f;
static const float FACE_AREA_LEGAL_THRES = 0.005f;

static const float FACE_SCORE_THRES = 0.80f;

#define SCALE_RATIO 1 / 1

FaceDetect::FaceDetect()
: AlgorithmBase(),
  m_addFace(false),
  m_jpegEncoder(NULL),
  m_inProcess(false)
{
  m_msgSendFlags = new unsigned[msgTypeMax];
  memset(m_msgSendFlags, 0, msgTypeMax * sizeof(unsigned));
}

FaceDetect::~FaceDetect()
{
  CAM_INFO();
  if (m_detector) {
    delete m_detector;
  }
  if (m_msgSendFlags) {
    delete[] m_msgSendFlags;
  }
}

bool FaceDetect::init()
{
  TimePerf perf("face detect init");
  m_detector = FaceDetector::create();
  cyberdog_camera::FaceManager::getInstance();

  m_jpegEncoder = std::make_shared<cyberdog_camera::JpegEncoder>(
    FACE_ROI_WIDTH, FACE_ROI_HEIGHT, "face-enc");
  if (!m_jpegEncoder) {
    CAM_ERR("Failed to create JPEGEncoder.");
    return false;
  }

  rclcpp::ServicesQoS qos;
  pub_ = CameraManager::getInstance()->create_publisher<FaceInfoT>("face", qos);

  return true;
}

bool FaceDetect::processImageBuffer(ImageBuffer & buffer)
{
  if (buffer.data == NULL) {
    return false;
  }

  m_inProcess = true;
  if (m_addFace) {
    std::vector<SingleFaceInfo> faces_info;
    m_detector->detect(buffer, faces_info);

    std::vector<SingleFaceInfo> infos = filter_face_info(faces_info);
    bool legal = processFaceInfo(infos);

    if (legal) {
      if (checkFacePose(infos[0]) && infos[0].score > FACE_SCORE_THRES) {
        cv::Rect rect = getFaceImageRect(infos[0].rect);
        size_t size;
        unsigned char * jpeg_buf;
        cropFaceRegion(buffer.fd, rect, &jpeg_buf, size);
        FaceManager::getInstance()->addFaceData(
          m_faceId, infos[0].feats, jpeg_buf, size);

        NCSClient::getInstance().play(SoundFaceAddEnd);

        m_addFace = false;
      }
    }
  } else {
    std::vector<MatchedFaceInfo> faces_info;
    FeatureMap endlib_feats = cyberdog_camera::FaceManager::getInstance()->getFeatures();
    {
      TimePerf perf("facedetct");
      m_detector->match(buffer, endlib_feats, faces_info);
    }

    processFaceResult(buffer, faces_info);
  }
  m_inProcess = false;

  bufferDone(buffer);
  return true;
}

std::vector<SingleFaceInfo> FaceDetect::filter_face_info(std::vector<SingleFaceInfo> & infos)
{
  std::vector<SingleFaceInfo> result;

  for (size_t i = 0; i < infos.size(); i++) {
    if (infos[i].rect.left >= FACE_ROI_X &&
      infos[i].rect.top >= FACE_ROI_Y &&
      (infos[i].rect.left + infos[i].rect.width) <= FACE_ROI_X + FACE_ROI_WIDTH &&
      (infos[i].rect.top + infos[i].rect.height) <= FACE_ROI_Y + FACE_ROI_HEIGHT)
    {
      result.push_back(infos[i]);
    }
  }

  return result;
}

bool FaceDetect::processFaceInfo(std::vector<SingleFaceInfo> & infos)
{
  float mean[10];
  double stdev[10];

  m_faceStats[statsFaceNum].push_back(infos.size());
  if (infos.size() != 1) {
    m_faceStats[statsFaceYaw].push_back(0.0f);
    m_faceStats[statsFacePitch].push_back(0.0f);
    m_faceStats[statsFaceRow].push_back(0.0f);
    m_faceStats[statsFaceArea].push_back(0.0f);
  } else {
    m_faceStats[statsFaceYaw].push_back(infos[0].poses[0]);
    m_faceStats[statsFacePitch].push_back(infos[0].poses[1]);
    m_faceStats[statsFaceRow].push_back(infos[0].poses[2]);
    m_faceStats[statsFaceArea].push_back(
      static_cast<float>(infos[0].rect.width * infos[0].rect.height) / (1280 * 960));
  }

  // 1.face number should be exactly 1
  if (m_faceStats[statsFaceNum].full()) {
    get_mean_stdev(m_faceStats[statsFaceNum].vector(), mean[statsFaceNum], stdev[statsFaceNum]);
    if (stdev[statsFaceNum] == FACE_NUMBER_STABLE_VAL) {
      if (mean[statsFaceNum] == 1.0f) {
        CAM_INFO("Nice, only 1 face!!");
      } else if (mean[statsFaceNum] == 0.0f) {
        CAM_INFO("No face!!");
        return false;
      } else {
        CAM_INFO("More than 1 face!!");
        if (!m_msgSendFlags[msgNumError]) {
          NCSClient::getInstance().play(SoundFaceNumWarn);
          m_msgSendFlags[msgNumError] = 1;
        }
        return false;
      }
    } else {
      CAM_INFO("Face number not stable!!");
      return false;
    }
  } else {
    return false;
  }

  // 2.face pose
  if (m_faceStats[statsFaceYaw].full() &&
    m_faceStats[statsFacePitch].full() &&
    m_faceStats[statsFaceRow].full())
  {
    get_mean_stdev(m_faceStats[statsFaceYaw].vector(), mean[statsFaceYaw], stdev[statsFaceYaw]);
    get_mean_stdev(
      m_faceStats[statsFacePitch].vector(), mean[statsFacePitch],
      stdev[statsFacePitch]);
    get_mean_stdev(m_faceStats[statsFaceRow].vector(), mean[statsFaceRow], stdev[statsFaceRow]);
    CAM_INFO(
      "face num = %u, mean(%f, %f, %f), stdev(%f, %f, %f)",
      infos.size(),
      mean[1], mean[2], mean[3],
      stdev[1], stdev[2], stdev[3]);
    if (stdev[statsFaceYaw] < FACE_POSE_STABLE_THRES &&
      stdev[statsFacePitch] < FACE_POSE_STABLE_THRES &&
      stdev[statsFaceRow] < FACE_POSE_STABLE_THRES)
    {
      if (abs(mean[statsFaceYaw]) < FACE_POSE_YAW_LEGAL_THRES &&
        abs(mean[statsFacePitch]) < FACE_POSE_PITCH_LEGAL_THRES &&
        abs(mean[statsFaceRow]) < FACE_POSE_ROW_LEGAL_THRES)
      {
        CAM_INFO("Nice, degree is OK!!");
      } else {
        CAM_INFO("Degree is NOT OK!!");
        if (!m_msgSendFlags[msgPoseError]) {
          NCSClient::getInstance().play(SoundFacePoseWarn);
          m_msgSendFlags[msgPoseError] = 1;
        }
        return false;
      }
    } else {
      CAM_INFO("Degree is not stable!!");
      return false;
    }
  }

  // face distance
  if (m_faceStats[statsFaceArea].full()) {
    get_mean_stdev(m_faceStats[statsFaceArea].vector(), mean[statsFaceArea], stdev[statsFaceArea]);
    if (stdev[statsFaceArea] < FACE_AREA_STABLE_THRES) {
      if (mean[statsFaceArea] > FACE_AREA_LEGAL_THRES) {
        CAM_INFO("Nice, distance is OK!! %f", mean[statsFaceArea]);
        return true;
      } else {
        CAM_INFO("Distance is NOT OK!! %f", mean[statsFaceArea]);
        if (!m_msgSendFlags[msgDistError]) {
          NCSClient::getInstance().play(SoundFaceDistWarn);
          m_msgSendFlags[msgDistError] = 1;
        }
        return false;
      }
    } else {
      CAM_INFO("Distance is NOT stable!!");
      return false;
    }
  }

  return false;
}

void FaceDetect::clearFaceStats()
{
  for (int i = 0; i < statsFaceTypeMax; i++) {
    m_faceStats[i].clear();
  }
}

cv::Rect FaceDetect::getFaceImageRect(Rect & rect)
{
  cv::Rect ori_rect(rect.left,
    rect.top,
    rect.width,
    rect.height);
  cv::Rect square = square_rect(ori_rect);

  return scale_rect_center(square, square.size() * SCALE_RATIO);
}

bool FaceDetect::cropFaceRegion(
  int fd, const cv::Rect & rect, unsigned char ** jpeg_buf,
  size_t & size)
{
  m_jpegEncoder->setCropRect(rect.x, rect.y, rect.width, rect.height);
  size = m_jpegEncoder->encodeFromFd(fd, jpeg_buf, 90);
  CAM_INFO("face jpeg size = %ld", size);

  return true;
}

void FaceDetect::processFaceResult(ImageBuffer & frame, std::vector<MatchedFaceInfo> & faces)
{
  auto msg = std::make_unique<FaceInfoT>();

  msg->header.frame_id = "0";
  msg->header.stamp.sec = frame.timestamp.tv_sec;
  msg->header.stamp.nanosec = frame.timestamp.tv_nsec;
  msg->count = faces.size();
  msg->infos.resize(msg->count);

  for (size_t i = 0; i < msg->count; i++) {
    FaceT & info = msg->infos[i];
    info.roi.x_offset = faces[i].info.rect.left;
    info.roi.y_offset = faces[i].info.rect.top;
    info.roi.width = faces[i].info.rect.width;
    info.roi.height = faces[i].info.rect.height;
    info.id = faces[i].id;
    info.score = faces[i].info.score;
    info.match = faces[i].match_score;
    info.yaw = faces[i].info.poses[0];
    info.pitch = faces[i].info.poses[1];
    info.row = faces[i].info.poses[2];
    info.is_host = cyberdog_camera::FaceManager::getInstance()->isHost(info.id);

    CAM_INFO(
      "face %u: id(%s), host(%d), score(%f), poses(%f, %f, %f)",
      i, info.id.c_str(), info.is_host, info.score,
      info.yaw, info.pitch, info.row);
  }

  pub_->publish(std::move(msg));
}

bool FaceDetect::checkFacePose(SingleFaceInfo & info)
{
  float yaw = info.poses[0];
  float pitch = info.poses[1];
  float row = info.poses[2];

  return (std::abs(pitch) < 10.0) && (std::abs(yaw) < 20.0) && (std::abs(row) < 20.0);
}

int FaceDetect::addFace(const std::string & name, bool is_host)
{
  int ret = CAM_SUCCESS;

  m_addFace = true;
  m_faceId.name = name;
  m_faceId.is_host = is_host;

  NCSClient::getInstance().play(SoundFaceAddStart);

  return ret;
}

int FaceDetect::cancelAddFace()
{
  if (!m_addFace) {
    return CAM_INVALID_STATE;
  }

  m_addFace = false;
  clearFaceStats();
  memset(m_msgSendFlags, 0, msgTypeMax * sizeof(unsigned));

  return CAM_SUCCESS;
}

#endif

}  // namespace cyberdog_camera
