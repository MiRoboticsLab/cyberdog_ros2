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

ARG FROM_IMAGE=arm64v8/ubuntu:bionic
ARG gfw=0
FROM $FROM_IMAGE AS base-image
ENV CMAKE_VER="3.21.2"

FROM base-image AS build-in-gfw-1
ENV GITHUB_URL="hub.fastgit.org"
ENV GITHUB_RAW="raw.fastgit.org"

FROM base-image AS build-in-gfw-0
ENV GITHUB_URL="github.com"
ENV GITHUB_RAW="raw.githubusercontent.com"

FROM build-in-gfw-${gfw} AS final
# replace mirror (replace one near you)
RUN apt-get update && \
  apt-get install -q -y --no-install-recommends ca-certificates && \
  mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic main restricted universe multiverse\n" > /etc/apt/sources.list && \
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-updates main restricted universe multiverse\n" >> /etc/apt/sources.list && \
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-backports main restricted universe multiverse\n" >> /etc/apt/sources.list && \
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ bionic-security main restricted universe multiverse" >> /etc/apt/sources.list

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
  ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
  apt-get update && \
  apt-get install -q -y --no-install-recommends tzdata

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
  bash-completion \
  dirmngr \
  gnupg2 \
  lsb-release \
  python3-pip

# setup ros2 sources.list (replace one near you)
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu/ bionic main" > /etc/apt/sources.list.d/ros2-latest.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
  build-essential \
  git \
  wget \
  python3-colcon-common-extensions \
  python3-colcon-mixin \
  python3-rosdep \
  python3-vcstool \
  libopencv-dev \
  libzmq3-dev \
  libgtest-dev \
  cmake \
  libacl1-dev \
  libsqlite3-dev \
  libtinyxml2-dev \
  libbullet-dev \
  libncurses-dev \
  libbison-dev \
  libcurl4-openssl-dev \
  bison \
  liblog4cxx-dev \
  libeigen3-dev \
  libasio-dev \
  python3-dev \
  libboost-all-dev \
  libglib2.0-dev \
  libprotobuf-dev \
  libprotoc-dev \
  protobuf-compiler \
  protobuf-compiler-grpc \
  libasound2-dev \
  libconsole-bridge-dev \
  libmpg123-dev \
  libv4l-dev \
  libssl-dev

# set python3 default
RUN rm -f /usr/bin/python && \
  ln -s /usr/bin/python3 /usr/bin/python

# change pip mirror (replace one near you)
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# install python packages 
RUN pip3 install wheel cython && \
  pip3 install --upgrade setuptools
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
  pytest-repeat \
  pytest-rerunfailures \
  pytest \
  pytest-cov \
  pytest-runner \
  numpy \
  lark \
  ifcfg \
  netifaces \
  boost && \
  python3 -m pip install -U importlib-metadata importlib-resources

# upgrade CMake
RUN mkdir -p /root/cmake_tmp && \
  wget -q https://${GITHUB_URL}/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-linux-aarch64.sh -O /root/cmake_tmp/cmake.sh && \
  chmod +x /root/cmake_tmp/cmake.sh && \
  /root/cmake_tmp/cmake.sh --skip-license --prefix=/usr && \
  rm -rf /root/cmake_tmp

# get ros 2 repos
RUN mkdir -p /root/build_ros/src && \
  cd /root/build_ros && \
  wget -q https://${GITHUB_RAW}/MiRoboticsLab/cyberdog_ros2/main/tools/ros2_fork/mini.repos && \
  wget -q https://${GITHUB_RAW}/MiRoboticsLab/cyberdog_ros2/main/tools/ros2_fork/custom.repos

# speed up github
RUN cd /root/build_ros && \
  sed -i "s/github.com/${GITHUB_URL}/g" `grep github.com -rl * .*`

# get ros 2 code (comment it if you are out of GFW)
RUN cd /root/build_ros && \
  vcs import src < mini.repos

# setup nvidia l4t list
RUN echo "deb https://repo.download.nvidia.com/jetson/common r32.5 main\n" > /etc/apt/sources.list.d/nvidia-l4t.list && \
  echo "deb https://repo.download.nvidia.com/jetson/t194 r32.5 main" >> /etc/apt/sources.list.d/nvidia-l4t.list && \
  apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc

# setup l4t environment
RUN mkdir -p /opt/nvidia/l4t-packages/ && \
    touch /opt/nvidia/l4t-packages/.nv-l4t-disable-boot-fw-update-in-preinstall && \
    rm /var/lib/dpkg/info/* -rf && \
    apt-get update && \
    apt-get --yes --no-install-recommends --allow-change-held-packages -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install \
      apt-utils \
      nvidia-l4t-core \
      nvidia-l4t-jetson-multimedia-api \
      cuda-toolkit-10-2 \
      libnvinfer-dev \
      libnvonnxparsers-dev \
      libnvparsers-dev \
      libnvinfer-plugin-dev


# speed up github again (comment it if you are out of GFW)
RUN cd /root/build_ros && \
  sed -i "s/github.com/${GITHUB_URL}/g" `grep github.com -rl * .*` && \
  sed -i "s/raw.githubusercontent.com/${GITHUB_RAW}/g" `grep raw.githubusercontent.com -rl * .*` && \
  export ROS_VERSION=2 && \
  colcon build --merge-install --install-base /opt/ros2/galactic

# set ros 2 environment configs
RUN echo "ros2_galactic_on(){" >> /root/.bashrc && \
  echo "export ROS_VERSION=2" >> /root/.bashrc && \
  echo "export ROS_PYTHON_VERSION=3" >> /root/.bashrc && \
  echo "export ROS_DISTRO=galactic" >> /root/.bashrc && \
  echo "source /opt/ros2/galactic/setup.bash" >> /root/.bashrc && \
  echo "}" >> /root/.bashrc

# remove cache
RUN apt-get clean && \
  rm -rf /root/build_ros

CMD ["bash"]
