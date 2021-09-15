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


#ifndef MSGDISPATCHER_HPP_
#define MSGDISPATCHER_HPP_
#include <queue>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <chrono>
#include <utility>
template<typename MessageT>
class LatestMsgDispather
{
  using SharedPtrCallback = std::function<void (std::shared_ptr<MessageT>)>;

public:
  LatestMsgDispather()
  : callback_(nullptr), need_run_(true)
  {
  }
  ~LatestMsgDispather()
  {
    need_run_ = false;
    cond.notify_all();
    thread_->join();
  }
  void push(MessageT msg)
  {
    std::lock_guard<std::mutex> lk(mut);
    if (!need_run_) {
      return;
    }
    while (!queue_.empty()) {
      queue_.pop();
    }
    queue_.push(std::move(msg));
    cond.notify_all();
  }
  std::shared_ptr<MessageT> get()
  {
    std::unique_lock<std::mutex> ulk(mut);
    cond.wait(
      ulk, [this]
      {return !need_run_ || !this->queue_.empty();});
    if (!need_run_) {
      return NULL;
    }
    std::shared_ptr<MessageT> val(std::make_shared<MessageT>(std::move(queue_.front())));
    queue_.pop();
    return val;
  }
  template<typename CallbackT>
  void setCallback(CallbackT && callback)
  {
    callback_ = std::forward<CallbackT>(callback);
    run();
  }

  void run()
  {
    thread_ = std::make_shared<std::thread>(&LatestMsgDispather::process_thread, this);
  }
  void process_thread()
  {
    while (need_run_) {
      if (callback_ != nullptr) {
        auto msg = get();
        if (msg != NULL) {
          callback_(msg);
        }
      }
    }
  }

private:
  bool need_run_;
  std::queue<MessageT> queue_;
  std::mutex mut;
  std::condition_variable cond;
  SharedPtrCallback callback_;
  std::shared_ptr<std::thread> thread_;
};
#endif  // MSGDISPATCHER_HPP_
