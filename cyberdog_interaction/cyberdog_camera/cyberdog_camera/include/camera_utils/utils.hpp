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

#ifndef CAMERA_UTILS__UTILS_HPP_
#define CAMERA_UTILS__UTILS_HPP_

#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>
#include "./log.hpp"

#define TIME_STR "%Y%m%d_%H%M%S"

#define PRINT_TIMESTAMP \
  struct timeval current; \
  gettimeofday(&current, NULL); \
  printf( \
    "current timestamp: %luus\n", \
    current.tv_sec * 1000000 + current.tv_usec);

#define TIME_START(name) \
  { \
    struct timeval start, end; \
    gettimeofday(&start, NULL);

#define TIME_END(name) \
  gettimeofday(&end, NULL); \
  printf( \
    #name " cost time: %luus\n", \
    (end.tv_sec - start.tv_sec) * 1000000 \
    + (end.tv_usec - start.tv_usec)); \
}

enum
{
  CAM_SUCCESS = 0,
  CAM_INVALID_ARG,
  CAM_UNSUPPORTED,
  CAM_TIMEOUT,
  CAM_BUSY,
  CAM_INVALID_STATE,
  CAM_ERROR,
  CAM_UNDEFINE_ERROR = 255
};

std::string get_time_string();

void split_string(
  const std::string & s,
  std::vector<std::string> & v, const std::string & sp);
std::map<std::string, std::string> parse_parameters(std::string & params);

/* file operation function */
size_t get_file_size(const std::string & path);
int remove_file(const std::string & path);
int rename_file(const std::string & from, const std::string & to);

/* rectangle operation function */
cv::Rect scale_rect_center(cv::Rect rect, cv::Size size);
cv::Rect square_rect(cv::Rect rect);

inline float get_rect_area(float x0, float y0, float x1, float y1)
{
  return (x1 - x0) * (y1 - y0);
}

void get_mean_stdev(std::vector<float> & vec, float & mean, double & stdev);

struct Condition
{
public:
  Condition();
  ~Condition();

  int wait();
  int timedwait(uint32_t ms);
  int signal();

private:
  pthread_mutex_t m_mutex;
  pthread_cond_t m_cond;
};

struct TimePerf
{
public:
  explicit TimePerf(const std::string & name = "");
  ~TimePerf();

private:
  struct timeval start, end;
  std::string m_name;
};

template<typename T, size_t COUNT>
struct SizedVector
{
public:
  SizedVector()
  {
    m_index = 0;
  }

  void push_back(const T & obj)
  {
    if (m_vector.size() >= COUNT) {
      m_index %= COUNT;
      m_vector[m_index] = obj;
      m_index++;
    } else {
      m_vector.push_back(obj);
    }
  }

  size_t size()
  {
    return m_vector.size();
  }

  std::vector<T> & vector()
  {
    return m_vector;
  }

  void clear()
  {
    m_vector.clear();
    m_index = 0;
  }

  bool full()
  {
    return m_vector.size() == COUNT;
  }

private:
  std::vector<T> m_vector;
  size_t m_index;
};

#endif  // CAMERA_UTILS__UTILS_HPP_
