// Copyright (c) 2021 Beijing Xiaomi Mobile Software Co., Ltd.
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


#ifndef AUDIO_BASE__MEMPOOL_HPP_
#define AUDIO_BASE__MEMPOOL_HPP_

#include <vector>
#include <map>
#include <utility>

namespace audio_base
{
class MemPool
{
public:
  MemPool(int size, int chunk)
  : m_iSize(size), m_iChunkSize(chunk)
  {}

  ~MemPool()
  {
    Clear();
    Reset();
  }

  bool Create(bool need_set = false)
  {
    /**
     * @brief malloc memory, prepare for wokring
     *
     * @return bool, 0 success
     */
    bool result = true;
    for (int i = 0; i < m_iSize; ++i) {
      char * p = reinterpret_cast<char *>(malloc(m_iChunkSize));
      if (p == nullptr) {
        Clear(i);
        result = false;
        break;
      } else {
        if (need_set) {
          memset(p, 0x0, m_iChunkSize);
        }
        m_vPointer.push_back(p);
      }
    }

    if (result) {
      for (int i = 0; i < m_iSize; ++i) {
        m_mRecorder.insert(std::make_pair(i, false));
      }
    }
    return true;
  }

  bool Clear(int position = -1)
  {
    /**
     * @brief free memory fron 0 to position, default free all
     *
     */
    bool result = true;
    int free_size = position == -1 ? m_iSize : position;
    free_size = free_size > m_iSize ? m_iSize : free_size;
    for (int i = 0; i < free_size; ++i) {
      if (m_vPointer[i] != nullptr) {
        free(m_vPointer[i]);
        m_vPointer[i] = nullptr;
      } else {
        result = false;
      }
    }
    if (result) {
      Reset();
    }
    return result;
  }

  void Reset()
  {
    for (auto iter = m_mRecorder.begin(); iter != m_mRecorder.end(); ++iter) {
      iter->second = false;
    }
  }

  int GetMemory(char ** src)
  {
    /**
     * @brief get an unused memory block, after using must call Release
     *
     */
    m_iWorker++;
    m_iWorker %= m_iSize;
    // fprintf(stdout, "vc:combiner mempool get memory worker: %d, sizeL %d\n", m_iWorker, m_iSize);
    // need  check safety in future, depend user using Release correctly
    *src = reinterpret_cast<char *>(m_vPointer[m_iWorker]);
    m_mRecorder.at(m_iWorker) = true;
    return m_iWorker;
  }

  int GetMemory(unsigned char ** src)
  {
    /**
     * @brief get an unused memory block, after using must call Release
     *
     */
    m_iWorker++;
    m_iWorker %= m_iSize;
    // fprintf(stdout, "vc:combiner mempool get memory worker: %d, sizeL %d\n", m_iWorker, m_iSize);
    // need  check safety in future, depend user using Release correctly
    *src = reinterpret_cast<unsigned char *>(m_vPointer[m_iWorker]);
    m_mRecorder.at(m_iWorker) = true;
    return m_iWorker;
  }

  bool Release(int index)
  {
    auto iter = m_mRecorder.find(index);
    if (iter != m_mRecorder.end()) {
      iter->second = false;
      return true;
    }
    return false;
  }

private:
  int m_iSize;
  int m_iChunkSize;
  int m_iWorker;
  std::vector<char *> m_vPointer;
  std::map<int, bool> m_mRecorder;
};  // MemPool
}  // namespace audio_base
#endif  // AUDIO_BASE__MEMPOOL_HPP_
