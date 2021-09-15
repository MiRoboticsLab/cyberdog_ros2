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

#ifndef CAMERA_UTILS__QUEUE_HPP_
#define CAMERA_UTILS__QUEUE_HPP_

#include <pthread.h>
#include <queue>

template<typename T>
class Queue
{
public:
  Queue()
  {
    pthread_mutex_init(&m_mutex, NULL);
    pthread_cond_init(&m_cond, NULL);
  }
  ~Queue()
  {
    pthread_mutex_destroy(&m_mutex);
    pthread_cond_destroy(&m_cond);
  }

  void push(const T & obj)
  {
    pthread_mutex_lock(&m_mutex);
    m_queue.push(obj);
    pthread_cond_signal(&m_cond);
    pthread_mutex_unlock(&m_mutex);
  }

  T & pop()
  {
    pthread_mutex_lock(&m_mutex);
    while (m_queue.empty()) {
      pthread_cond_wait(&m_cond, &m_mutex);
    }
    T & obj = m_queue.front();
    m_queue.pop();
    pthread_mutex_unlock(&m_mutex);
    return obj;
  }

  size_t size()
  {
    pthread_mutex_lock(&m_mutex);
    size_t size = m_queue.size();
    pthread_mutex_unlock(&m_mutex);
    return size;
  }

private:
  std::queue<T> m_queue;
  pthread_mutex_t m_mutex;
  pthread_cond_t m_cond;
};

#endif  // CAMERA_UTILS__QUEUE_HPP_
