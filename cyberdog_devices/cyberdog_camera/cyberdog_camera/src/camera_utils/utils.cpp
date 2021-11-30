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

#include <unistd.h>
#include <numeric>
#include <string>
#include <vector>
#include <map>
#include "camera_utils/utils.hpp"

std::string get_time_string()
{
  char str_time[64];
  struct tm t;
  time_t now;
  time(&now);
  localtime_r(&now, &t);
  strftime(str_time, sizeof(str_time), TIME_STR, &t);

  return str_time;
}

void split_string(
  const std::string & s,
  std::vector<std::string> & v, const std::string & sp)
{
  std::string::size_type pos1, pos2;
  pos2 = s.find(sp);
  pos1 = 0;

  while (std::string::npos != pos2) {
    v.push_back(s.substr(pos1, pos2));
    pos1 = pos2 + sp.size();
    pos2 = s.find(sp, pos1);
  }
  if (pos1 != s.length()) {
    v.push_back(s.substr(pos1));
  }
}

std::map<std::string, std::string> parse_parameters(std::string & params)
{
  std::vector<std::string> key_values;
  std::map<std::string, std::string> params_map;

  split_string(params, key_values, ";");
  for (size_t i = 0; i < key_values.size(); i++) {
    size_t pos = key_values[i].find("=");
    if (std::string::npos != pos) {
      std::string key = key_values[i].substr(0, pos);
      std::string value = key_values[i].substr(pos + 1);
      params_map[key] = value;
    }
  }

  return params_map;
}

size_t get_file_size(const std::string & path)
{
  FILE * fp;
  size_t size;

  fp = fopen(path.c_str(), "rb");
  if (!fp) {
    return 0;
  }

  fseek(fp, 0, SEEK_END);
  size = ftell(fp);
  fclose(fp);

  if (size > 0) {
    return size;
  } else {
    return 0;
  }
}

int remove_file(const std::string & path)
{
  if (access(path.c_str(), 0) != 0) {
    CAM_ERR("File %s not exist.", path.c_str());
    return CAM_ERROR;
  }

  if (unlink(path.c_str()) != 0) {
    CAM_ERR("Remove file %s failed - %s.", path.c_str(), strerror(errno));
    return CAM_ERROR;
  }

  return CAM_SUCCESS;
}

int rename_file(const std::string & from, const std::string & to)
{
  if (access(from.c_str(), 0) != 0) {
    CAM_ERR("File %s not found", from.c_str());
    return CAM_ERROR;
  }

  if (rename(from.c_str(), to.c_str()) != 0) {
    CAM_ERR("Rename file %s failed - %s.", from.c_str(), strerror(errno));
    return CAM_ERROR;
  }

  return CAM_SUCCESS;
}

cv::Rect scale_rect_center(cv::Rect rect, cv::Size size)
{
  rect = rect + size;
  cv::Point pt;
  pt.x = cvRound(size.width / 2.0);
  pt.y = cvRound(size.height / 2.0);

  return rect - pt;
}

cv::Rect square_rect(cv::Rect rect)
{
  rect.width = rect.width + (rect.width & 0x01);
  rect.height = rect.height + (rect.height & 0x01);
  int delta = abs(rect.width - rect.height);

  return rect.width > rect.height ?
         scale_rect_center(rect, cv::Size(0, delta)) :
         scale_rect_center(rect, cv::Size(delta, 0));
}

void get_mean_stdev(std::vector<float> & vec, float & mean, double & stdev)
{
  size_t count = vec.size();
  float sum = std::accumulate(vec.begin(), vec.end(), 0.0);
  mean = sum / count;

  double accum = 0.0;
  for (size_t i = 0; i < count; i++) {
    accum += (vec[i] - mean) * (vec[i] - mean);
  }

  stdev = sqrt(accum / count);
}

TimePerf::TimePerf(const std::string & name)
{
  m_name = name;
  gettimeofday(&start, NULL);
}

TimePerf::~TimePerf()
{
  gettimeofday(&end, NULL);
  CAM_INFO(
    "%s cost time: %luus", m_name.c_str(),
    (end.tv_sec - start.tv_sec) * 1000000 +
    (end.tv_usec - start.tv_usec));
}

Condition::Condition()
{
  pthread_condattr_t attr;

  pthread_condattr_init(&attr);
  pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
  pthread_mutex_init(&m_mutex, NULL);
  pthread_cond_init(&m_cond, &attr);
}

Condition::~Condition()
{
  pthread_mutex_destroy(&m_mutex);
  pthread_cond_destroy(&m_cond);
}

int Condition::wait()
{
  int ret = 0;

  pthread_mutex_lock(&m_mutex);
  ret = pthread_cond_wait(&m_cond, &m_mutex);
  pthread_mutex_unlock(&m_mutex);

  return ret;
}

int Condition::timedwait(uint32_t ms)
{
  int ret = 0;
  struct timespec ts;

  clock_gettime(CLOCK_MONOTONIC, &ts);
  ts.tv_sec += (ms / 1000);

  pthread_mutex_lock(&m_mutex);
  ret = pthread_cond_timedwait(&m_cond, &m_mutex, &ts);
  pthread_mutex_unlock(&m_mutex);

  return ret;
}

int Condition::signal()
{
  int ret = 0;

  pthread_mutex_lock(&m_mutex);
  ret = pthread_cond_signal(&m_cond);
  pthread_mutex_unlock(&m_mutex);

  return ret;
}
