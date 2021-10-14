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

#ifndef AUDIO_BACKTRACE__BACK_TRACE_H_
#define AUDIO_BACKTRACE__BACK_TRACE_H_

#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <execinfo.h>

#include <string>
#include <cstring>
#include <iomanip>
#include <chrono>
#include <vector>

static void dump(void)
{
#define BACKTRACE_SIZE   16
  int j, nptrs;
  void * buffer[BACKTRACE_SIZE];
  char ** strings;

  nptrs = backtrace(buffer, BACKTRACE_SIZE);
  strings = backtrace_symbols(buffer, nptrs);
  if (strings == NULL) {
    fprintf(stderr, "backtrace_symbols error!");
    exit(EXIT_FAILURE);
  }

  for (j = 0; j < nptrs; j++) {
    fprintf(stderr, "  [%02d] %s\n", j, strings[j]);
  }

  free(strings);
}

static void signal_handler(int signo)
{
  fprintf(stderr, "=========>>>catch signal %d <<<=========\n", signo);
  fprintf(stderr, "backtrace start...\n");
  dump();
  fprintf(stderr, "backtrace end...\n");

  signal(signo, SIG_DFL);
  raise(signo);
}

#endif  // AUDIO_BACKTRACE__BACK_TRACE_H_
