# cyberdog_wireless

## 一、简介

该模块包括两个项目：`bluetooth`和`wifirssi`。

`bluetooth`的主要功能由以下两部分组成：

1、注册蓝牙GATT service，发送BLE广播，等待手机APP连接成功后，接收APP发送的Wifi信息，实现Wifi的联网操作，并返回联网结果。

2、通过APP提供的命令，实现蓝牙手柄的扫描，连接，达到通过手柄控制CyberDog的目的。如手柄连接成功，还将启动自动连接功能。

`wifirssi`的主要功能是创建定时任务，实时查询当前已连接Wifi的信号强度，并通过ros2的topic通讯进行传输。

## 二、启动

`bluetooth`和`wifirssi`通过ros2启动，具体可参考cyberdog bringup。

## 三、模块依赖

`bluetooth`依赖的模块包括LED、Touch、Audio、手机APP等外设模块，以及运动控制等内部通讯模块。

`wifirssi`无依赖。

## 四、通讯格式

除其它通讯模块的依赖外，蓝牙自定义的通讯格式有两个，均与手柄相关。

1、BtRemoteCommand.srv

```c++
uint8 GET_STATUS = 0
uint8 SCAN_DEVICE = 1
uint8 CONNECT_DEVICE = 2
uint8 DISCONNECT_DEVICE = 3
uint8 REMOTE_RECONNECT_DEVICE = 4

uint8 command
string address
---
bool success
```

用于手机APP通过wifi和`bluetooth`通讯使用，可实现蓝牙手柄的状态查询，外设扫描、连接、断开、删除保存文件等功能。

command的参数范围如上所示，address表示对应command需要操作的mac地址，只针对CONNECT_DEVICE有效。当`bluetooth`接收到命令且可以执行时返回success为True，否则为返回success为False。

2、BtRemoteEvent.msg

```c++
uint8 scan_status
uint8 remote_status
string address
string scan_device_info
string error
```

用于执行APP命令后的执行返回。

scan_status表示扫描状态，范围如下，默认为SCAN_STATUS_END。

```python
SCAN_STATUS_START = 1
SCAN_STATUS_INFO = 2
SCAN_STATUS_END = 3
SCAN_STATUS_ERROR = 4
```

remote_status表示手柄的连接状态，范围如下，默认为REMOTE_STATUS_DISCONNECTED。

```python
REMOTE_STATUS_CONNECTING = 5
REMOTE_STATUS_CONNECTED = 6
REMOTE_STATUS_CONNECT_ERROR = 7
REMOTE_STATUS_NOTIFY = 8
REMOTE_STATUS_DISCONNECTING = 9
REMOTE_STATUS_DISCONNECTED = 10
REMOTE_STATUS_DISCONNECT_ERROR = 11
```

address表示正在执行的设备地址，scan_device_info表示扫描到的设备信息，error表示扫描或连接的错误状态。

## 五、bluetooth逻辑

1、`gattserver.main()`入口启动，注册通讯、注册蓝牙GATT Service，注册广播，并发起广播。

- 通讯部分包括：外设交互（用于灯效显示的`LedClientAsync`、用于提示音播报的`AudioPlayActionClient`、用于触摸操作的`TouchSubscriber`），内联通讯（用于发送手柄摇杆信息的`RemotePublisherKey`、用于调整身高的`parameterPublisher`，用于接收APP发送手柄命令的`RemoteCommandService`，用于上报手柄切换模式的`gaitActionClient`，用于切换手柄和APP控制的`ChangeModeActionClient`，用于通知CyberDog和手机IP地址的`PublisherPhoneIP`）。

- GATT Service注册包括两个Characteristic，分别为用于接收wifi信息的`WIFI_CHRC_UUID_WRITE`，以及用于wifi信息上报的`WIFI_CHRC_UUID_NOTIFY`。

- 蓝牙广播会发送设备的Name，以及Service的UUID等信息。

2、待手机APP连接成功后，会配置`WIFI_CHRC_UUID_NOTIFY`为Notification模式，并进入`StartNotify`()函数，此时表示蓝牙连接成功，等待接收APP发送的Wifi信息。

3、APP发送Wifi信息后，`WIFI_CHRC_UUID_WRITE`通过`WriteValue`接收并解析此信息，并通过`gettserver.handleThread()`发起Wifi的连接。

4、当Wifi处于连接状态下，且Wifi信息和当前一致，表示Wifi连接成功。如果当前Wifi信息并没有处于连接状态，则查找Wifi配置，确定是需要回连还是发起新的连接，最终通过`nmcli`命令实现Wifi的连接。如果Wifi连接成功，APP会断开蓝牙，否则继续保持当前的蓝牙连接。

5、当APP蓝牙和CyberDog未连接，CyberDog也没有并没有发送蓝牙广播，同时没有手柄连接时，长按Touch，会再次启动广播。

6、`bluetooth`也可通过dds模块接收APP发送的手柄命令，这里包括查询状态，扫描，连接，断开，移除保持文件。通过这些命令，可以似的机器人连接手柄并通过手柄操作机器人。具体可参考`BtRemoteCommand.srv`的实现。

7、手柄连接成功后，会自动记录手柄的mac，并保存到`/etc/btRemoteAddress.conf`文件中。当手柄断开或APP蓝牙和CyberDog断开后，将启动自动扫描、自动连接功能，方便手柄的重连操作。

## 六、二次开发

软件环境：python3.6，硬件环境：blueZ V4.80，外部依赖库：bluepy V1.3.0

代码为python开发，CyberDog上[bluetooth](/opt/ros2/cyberdog/lib/python3.6/site-packages/bluetooth/)和[wifirssi](/opt/ros2/cyberdog/lib/python3.6/site-packages/wifirssi)均可直接修改，无需编译。

