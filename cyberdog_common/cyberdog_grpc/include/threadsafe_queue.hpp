// Copyright (c) 2021 Beijing Xiaomi Mobile Software Co., Ltd. All rights reserved.
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


#ifndef THREADSAFE_QUEUE_HPP_
#define THREADSAFE_QUEUE_HPP_
#include <queue>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <chrono>
#include <utility>

template<typename T>
class threadsafe_queue
{
public:
  threadsafe_queue() {time_out_seconds = 35;}
  ~threadsafe_queue() {}

  void push(T new_data)
  {
    std::lock_guard<std::mutex> lk(mut);
    data_queue.push(std::move(new_data));
    cond.notify_all();
  }
  void wait_and_pop(T & val)
  {
    std::unique_lock<std::mutex> ulk(mut);
    if (time_out_seconds != 0) {
      if (cond.wait_for(ulk, std::chrono::seconds(time_out_seconds)) == std::cv_status::timeout) {
        return;
      } else {
        cond.wait(ulk, [this] {return !this->data_queue.empty();});
      }
    }
    val = std::move(data_queue.front());
    data_queue.pop();
  }
  std::shared_ptr<T> wait_and_pop()
  {
    std::unique_lock<std::mutex> ulk(mut);
    if (time_out_seconds != 0) {
      if (cond.wait_for(ulk, std::chrono::seconds(time_out_seconds)) == std::cv_status::timeout) {
        return NULL;
      } else {
        cond.wait(ulk, [this] {return !this->data_queue.empty();});
      }
    }
    std::shared_ptr<T> val(std::make_shared<T>(std::move(data_queue.front())));
    data_queue.pop();
    return val;
  }
  bool try_pop(T & val)
  {
    std::lock_guard<std::mutex> lk(mut);
    if (data_queue.empty()) {
      return false;
    }
    val = std::move(data_queue.front());
    data_queue.pop();
    return true;
  }
  std::shared_ptr<T> try_pop()
  {
    std::shared_ptr<T> val;
    std::lock_guard<std::mutex> lk(mut);
    if (data_queue.empty()) {
      return val;
    }
    val = std::make_shared<T>(std::move(data_queue.front()));
    data_queue.pop();
    return val;
  }
  bool empty()
  {
    std::lock_guard<std::mutex> lk(mut);
    return data_queue.empty();
  }
  void set_time_out(int t)
  {
    time_out_seconds = t;
  }

  void clear()
  {
    std::lock_guard<std::mutex> lk(mut);
    while (!data_queue.empty()) {data_queue.pop();}
  }

private:
  std::queue<T> data_queue;
  std::mutex mut;
  std::condition_variable cond;
  int time_out_seconds;
};
#endif  // THREADSAFE_QUEUE_HPP_
