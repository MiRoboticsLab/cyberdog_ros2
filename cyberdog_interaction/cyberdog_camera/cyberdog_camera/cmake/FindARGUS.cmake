# Copyright (c) 2021 Beijing Xiaomi Mobile Software Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

find_package(PkgConfig)

unset(ARGUS_INCLUDE_DIR CACHE)
find_path(ARGUS_INCLUDE_DIR Argus/Argus.h
          HINTS /usr/src/jetson_multimedia_api/include)

unset(ARGUS_LIBRARIE CACHE)
find_library(ARGUS_LIBRARY NAMES nvargus
  HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra)

set(ARGUS_LIBRARIES ${ARGUS_LIBRARY})

set(ARGUS_INCLUDE_DIRS ${ARGUS_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set ARGUS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(ARGUS DEFAULT_MSG
                                  ARGUS_LIBRARY ARGUS_INCLUDE_DIR)

mark_as_advanced(ARGUS_INCLUDE_DIR ARGUS_LIBRARY)
