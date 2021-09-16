# Camera Module for XIAOMI&reg; CyberDog&trade; Device
These are packages for using xiaomi cyberdog cameras with ROS2.

## packages description

### AI_SDK
XIAOMI AI vision algotithm C++ libraries, include **FaceDetect**, **BodyDetect** and **ReId** algorithm.

This package provides header, libraries and AI models.

### cyberdog_vision
XIAOMI AI vision algotithm C++ API.

### cyberdog_camera
Camera Application based on Nvidia [Argus](https://docs.nvidia.com/jetson/l4t-multimedia/group__LibargusAPI.html) api and [ROS2](https://www.ros.org/).

This package implement a ros2 application which provides camera functions such as capture, recording and AI detection.

Cyberdog_camera use Argus API to control camera hardware and capture image, use ROS2 API to manager camera node and export service interactions.
