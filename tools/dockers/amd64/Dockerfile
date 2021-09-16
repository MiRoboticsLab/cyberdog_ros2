#!/usr/bin/python3
#
# Copyright 2021 Beijing Xiaomi Mobile Software Co., Ltd.
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

ARG FROM_IMAGE=ubuntu:focal
FROM $FROM_IMAGE

# replace mirror
RUN apt-get update && \
  apt-get install -q -y --no-install-recommends ca-certificates && \
  mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse\n" > /etc/apt/sources.list && \
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse\n" >> /etc/apt/sources.list && \
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse\n" >> /etc/apt/sources.list && \
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
  ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
  apt-get update && \
  apt-get install -q -y --no-install-recommends tzdata && \
  rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
  bash-completion \
  dirmngr \
  gnupg2 \
  lsb-release \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

# setup sources.list
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu/ focal main" > /etc/apt/sources.list.d/ros2-latest.list

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
  build-essential \
  git \
  wget \
  python3-colcon-common-extensions \
  python3-colcon-mixin \
  python3-rosdep \
  python3-vcstool

# install python packages
RUN pip3 install -U \
  argcomplete \
  flake8 \
  flake8-blind-except \
  flake8-builtins \
  flake8-class-newline \
  flake8-comprehensions \
  flake8-deprecated \
  flake8-docstrings \
  flake8-import-order \
  flake8-quotes \
  pytest \
  pytest-repeat \
  pytest-rerunfailures

# install cross compiler
RUN apt-get update && apt-get install --no-install-recommends -y \
  gcc-aarch64-linux-gnu \
  gcc-arm-linux-gnueabihf 

# remove apt cache
RUN rm -rf /var/lib/apt/lists/*

# add tmp dir
RUN mkdir -p /tmp && chmod 777 /tmp

# add kernel cross compiler
RUN mkdir -p /home/builder/build_tools
RUN wget -q https://cdn.cnbj2m.fds.api.mi-img.com/cyberdog-package/build/l4t_gcc_7.3.1.tgz && \
  tar -xf l4t_gcc_7.3.1.tgz -C /home/builder/build_tools && \
  rm l4t_gcc_7.3.1.tgz

CMD ["bash"]
