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

#ifndef CAMERA_SERVICE__FACE_MANAGER_HPP_
#define CAMERA_SERVICE__FACE_MANAGER_HPP_

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include "./ros2_service.hpp"
#include "camera_utils/utils.hpp"

#define FACE_CACHE_SIZE (128 * 1024)

namespace cyberdog_camera
{

struct FaceId
{
  std::string name;
  bool is_host;
};

class FaceManager
{
public:
  static FaceManager * getInstance();
  static const std::string getFaceDataPath();
  std::map<std::string, std::vector<float>> & getFeatures();
  bool addFaceData(
    const FaceId & face_id,
    std::vector<float> & feats, void * img, size_t size);
  int addFaceInfo(const std::string & name, bool is_host = false);
  int cancelAddFace(const std::string & args);
  int confirmFace();
  int updateFaceId(std::string & ori_name, std::string & new_name);
  int deleteFace(std::string & face_name);
  bool findFace(const std::string & face_name);
  bool isHost(const std::string & face_name);

private:
  FaceManager();
  ~FaceManager();
  void initialize();
  bool loadFeatures();
  void publishResult(
    int result,
    const std::string & name = "",
    void * img = NULL, size_t size = 0);
  int stopFaceDetect();
  static void * stopThreadFunc(void * args);

  std::map<std::string, std::vector<float>> m_features;
  std::map<std::string, bool> m_hostMap;
  std::mutex m_mutex;
  pthread_t m_thread;
  rclcpp::Publisher<FaceResultT>::SharedPtr m_resultPub;

  bool m_inFaceAdding;
  bool m_faceDetectEnabled;
  /* save last face info, wait for user confirm */
  FaceId m_faceIdCached;
  std::vector<float> m_faceFeatsCached;
  uint8_t m_faceImageCached[FACE_CACHE_SIZE];
  size_t m_faceCacheSize;
};

}  // namespace cyberdog_camera

#endif  // CAMERA_SERVICE__FACE_MANAGER_HPP_
