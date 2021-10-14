// Copyright (c) 2021 Xiaomi Corporation
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

#ifndef AUDIO_BASE__AUDIO_QUEUE_HPP_
#define AUDIO_BASE__AUDIO_QUEUE_HPP_

#include <list>
#include <mutex>
#include <condition_variable>

namespace athena_audio
{
template<typename T>
class MsgQueue
{
public:
  MsgQueue()
  {
    is_wait = false;
  }
  ~MsgQueue()
  {
    if (!data_list.empty()) {
      data_list.clear();
    }
    if (is_wait) {
      read_signal.notify_all();
    }
  }

  bool EnQueue(const T & t)
  {
    std::unique_lock<std::mutex> lk(data_lock);
    data_list.emplace_front(t);
    if (is_wait) {
      is_wait = false;
      read_signal.notify_one();
    }
    return true;
  }

  bool EnQueueOne(const T & t)
  {
    std::unique_lock<std::mutex> lk(data_lock);
    if (!Empty()) {
      return false;
    } else {
      data_list.emplace_front(t);
      if (is_wait) {
        is_wait = false;
        read_signal.notify_one();
      }
    }
    return true;
  }

  bool DeQueue(T & t)
  {
    std::unique_lock<std::mutex> lk(data_lock);
    if (data_list.empty()) {
      is_wait = true;
      read_signal.wait(lk);
    }
    if (data_list.empty()) {
      return false;
    } else {
      t = data_list.back();
      data_list.pop_back();
      return true;
    }
  }

  void Clear()
  {
    std::unique_lock<std::mutex> lk(data_lock);
    fprintf(stdout, "vc: msgqueue size: %d\n", static_cast<int>(data_list.size()));
    while (!data_list.empty()) {
      T t = data_list.front();
      t.free_mem();  // need a detect operation, make sure func calling is correct
      data_list.pop_front();
    }
    fprintf(stdout, "vc: msgqueue size: %d\n", static_cast<int>(data_list.size()));
  }

  int Size()
  {
    std::unique_lock<std::mutex> lk(data_lock);
    return static_cast<int>(data_list.size());
  }

  bool Empty()
  {
    return data_list.empty();
  }

private:
  std::condition_variable read_signal;
  std::mutex data_lock;
  std::list<T> data_list;
  bool is_wait;
};  // class MsgQueue
}  // namespace athena_audio

#endif  // AUDIO_BASE__AUDIO_QUEUE_HPP_
