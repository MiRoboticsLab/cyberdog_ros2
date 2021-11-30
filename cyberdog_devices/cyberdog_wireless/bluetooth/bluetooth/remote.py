#!/usr/bin/python3
#
# Copyright (c) 2021 Beijing Xiaomi Mobile Software Co., Ltd. All rights reserved.
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

import os
import struct
import threading
from threading import Thread
import time

import bluepy.btle
from bluepy.btle import DefaultDelegate, Peripheral, ScanEntry, Scanner, UUID

from ception_msgs.msg import BtRemoteEvent
from ception_msgs.srv import BtRemoteCommand
from motion_msgs.action import ChangeGait
from motion_msgs.action import ChangeMode
from motion_msgs.msg import Frameid
from motion_msgs.msg import Parameters
from motion_msgs.msg import SE3VelocityCMD
from rclpy.action import ActionClient
from rclpy.node import Node

try:
    from . import record
    from . import rclcomm
    from . import gattserver
except ImportError:
    import record
    import rclcomm
    import gattserver

# define for code
enableConnectTimeout: bool = False

SCAN_LONG_TIMEOUT = 5.0
SCAN_SHORT_TIMEOUT = 1.0
REMOTE_NAME = 'RadioLink'
KEY_SERVICE_UUID = UUID(0xFFE0)
KEY_CHAR_UUID = UUID(0xFFE4)
KEY_CHAR_CCC_UUID = 0x2902
RETRY_NUM = 2
TIME_SHORT_SLEEP = 0.5
TIME_LONG_SLEEP = 5.0  # for autoConnection scan interval
WAITING_FOR_NOTIFY_TIMEOUT = 1.0
NODE_CHECK_NUMS = 3
NODE_WAIT_SECS = 5.0
peripDevice = None
log: None = None
remote_pub: None = None
key_pub: None = None
para_pub: None = None
global_body_height = 0
gait_action: None = None
global_gait = 0
mode_action: None = None

STATUS_NONE = 0
# scan and remote status
SCAN_STATUS_START = 1
SCAN_STATUS_INFO = 2
SCAN_STATUS_END = 3
SCAN_STATUS_ERROR = 4
REMOTE_STATUS_CONNECTING = 5
REMOTE_STATUS_CONNECTED = 6
REMOTE_STATUS_CONNECT_ERROR = 7
REMOTE_STATUS_NOTIFY = 8
REMOTE_STATUS_DISCONNECTING = 9
REMOTE_STATUS_DISCONNECTED = 10
REMOTE_STATUS_DISCONNECT_ERROR = 11
scanStatus: int = SCAN_STATUS_END
remoteStatus: int = REMOTE_STATUS_DISCONNECTED
conTimer = None  # for connect timeout

# auto connection thread mode
AUTO_CONNECT_THREAD_IDLE = 0  # thread idle
AUTO_CONNECT_THREAD_WAITING = 1  # waiting to enter auto thread
AUTO_CONNECT_THREAD_ENTER = 2  # enter auto thread
AUTO_CONNECT_THREAD_SCAN = 3  # auto thread do scan
AUTO_CONNECT_THREAD_CONNECT = 4  # auto thread do connect
AUTO_CONNECT_THREAD_QUIT = 5  # auto thread will quit
autoConThreadRunning = AUTO_CONNECT_THREAD_IDLE
autoConAddress = ''  # auto connection device, read/write from file
autoAddressPath = '/etc/btRemoteAddress.conf'  # save address
autoConTimer = None  # thread time
AUTO_CONNECT_START_TIME = 10


# receive command from client
class RemoteCommandService(Node):

    def __init__(self):
        super().__init__('minimal_service', namespace=rclcomm.get_namespace())
        self.srv = self.create_service(BtRemoteCommand, 'btRemoteCommand', self.command_callback)

    # return: False for command type error or can not start, other is True
    def command_callback(self, request, response):
        global remote_pub, log, scanStatus, remoteStatus, autoConThreadRunning
        log.logger.info('Incoming request command: %d address: %s' %
                        (request.command, request.address))

        if gattserver.bleStatus is gattserver.BLE_STATUS_ADV or \
                gattserver.bleStatus is gattserver.BLE_STATUS_CONNECT:
            log.logger.error('current bleStatus is %d, do nothing' % gattserver.bleStatus)
            response.success = False
            return response
        else:
            response.success = True
        """ command range
        GET_STATUS = 0
        SCAN_DEVICE = 1
        CONNECT_DEVICE = 2
        DISCONNECT_DEVICE = 3
        REMOTE_RECONNECT_DEVICE = 4
        """
        if request.command == 0:  # GET_STATUS
            t = Thread(target=getAllStatus)
        elif request.command == 1:  # SCAN_DEVICE
            if gattserver.enableAutoRemoteControl is True:
                stopAutoConnect()  # check and stop auto connect
            if scanStatus is not SCAN_STATUS_END:
                response.success = False
            t = Thread(target=doScan, args=(SCAN_LONG_TIMEOUT,))
        elif request.command == 2:  # CONNECT_DEVICE
            if gattserver.enableAutoRemoteControl is True:
                stopAutoConnect()
            if remoteStatus is not REMOTE_STATUS_DISCONNECTED:
                response.success = False
            t = Thread(target=doConnect, args=(request.address,))
        elif request.command == 3:  # DISCONNECT_DEVICE
            if gattserver.enableAutoRemoteControl is True:
                stopAutoConnect()
            if REMOTE_STATUS_DISCONNECTING <= remoteStatus:
                response.success = False
            t = Thread(target=doDisconnect, args=(request.address,))
        elif request.command == 4:  # REMOTE_RECONNECT_DEVICE
            if gattserver.enableAutoRemoteControl is True:
                stopAutoConnect()
            t = Thread(target=delAutoAddress, args=(request.address,))
        else:  # return False when command is out of range
            response.success = False
            log.logger.warning('receive command error')
            return response
        t.start()
        return response


# publish scanStatus or remoteStatus
class RemotePublisherEvent(Node):

    def __init__(self):
        global remote_pub
        super().__init__('remotePubEve', namespace=rclcomm.get_namespace())
        self.publisher_ = self.create_publisher(BtRemoteEvent, 'remoteEvent', 0)
        remote_pub = self

    def publisher(self, _address, _scan_device_info, _error):
        global scanStatus, remoteStatus
        msg = BtRemoteEvent()
        msg.scan_status = scanStatus
        msg.remote_status = remoteStatus
        msg.address = _address
        msg.scan_device_info = _scan_device_info
        msg.error = str(_error)
        self.publisher_.publish(msg)


# publish key event
class RemotePublisherKey(Node):

    def __init__(self):
        global key_pub
        super().__init__('remotePubKey', namespace=rclcomm.get_namespace())
        self.publisher_ = self.create_publisher(SE3VelocityCMD, 'body_cmd', 0)
        self.zero = False  # False for send 0/0/0/0 when connected
        key_pub = self

    def publisher(self, linear_x, linear_y, angular_y, angular_z):
        msg = SE3VelocityCMD()
        msg.sourceid = SE3VelocityCMD.REMOTEC  # from SE3VelocityCMD.msg
        msg.velocity.frameid.id = Frameid.BODY_FRAME  # from Frameid.msg
        msg.velocity.timestamp = self.get_clock().now().to_msg()
        msg.velocity.linear_x = float(linear_x)
        msg.velocity.linear_y = float(linear_y)
        msg.velocity.linear_z = float(0)
        msg.velocity.angular_x = float(0)
        msg.velocity.angular_y = float(angular_y)
        msg.velocity.angular_z = float(angular_z)
        self.publisher_.publish(msg)

    def set_zero(self, zero):
        self.zero = zero

    def get_zero(self):
        return self.zero


# action gait, send combination key from CH5 and CH7, when Key CH6 pressed
class gaitActionClient(Node):

    def __init__(self):
        global log, gait_action
        super().__init__('gait_action_client', namespace=rclcomm.get_namespace())
        self._action_client = ActionClient(self, ChangeGait, 'checkout_gait')
        for i in range(NODE_CHECK_NUMS):
            if not self._action_client.wait_for_server(timeout_sec=NODE_WAIT_SECS):
                log.logger.info('gait service not available, waiting again...')
            else:
                gait_action = self
                break
            if i >= NODE_CHECK_NUMS:
                log.logger.error('init gait failed!!!!!')

    def send_goal(self, gait):
        global log
        goal_msg = ChangeGait.Goal()
        goal_msg.motivation = 253
        goal_msg.gaitstamped.timestamp = self.get_clock().now().to_msg()
        goal_msg.gaitstamped.gait = gait
        return self._action_client.send_goal_async(goal_msg)


class parameterPublisher(Node):

    def __init__(self):
        global para_pub
        super().__init__('paraPub', namespace=rclcomm.get_namespace())
        self.publisher_ = self.create_publisher(Parameters, 'para_change', 0)
        para_pub = self

    def publisher(self, height):
        msg = Parameters()
        msg.gait_height = 0.8  # just for test
        msg.body_height = height
        msg.timestamp = self.get_clock().now().to_msg()
        self.publisher_.publish(msg)


class ChangeModeActionClient(Node):
    MODE_DEFAULT = 0
    MODE_LOCK = 1
    MODE_MANUAL = 3

    def __init__(self):
        global log, mode_action
        super().__init__('mode_action_client', namespace=rclcomm.get_namespace())
        self._action_client = ActionClient(self, ChangeMode, 'checkout_mode')
        for i in range(NODE_CHECK_NUMS):
            if not self._action_client.wait_for_server(timeout_sec=NODE_WAIT_SECS):
                log.logger.info('mode service not available, waiting again...')
            else:
                mode_action = self
                self.mode = self.MODE_DEFAULT  # default
                break
            if i >= NODE_CHECK_NUMS:
                log.logger.error('init mode failed!!!!!')

    def send_goal(self, mode):
        mode_msg = ChangeMode.Goal()
        mode_msg.modestamped.timestamp = self.get_clock().now().to_msg()
        mode_msg.modestamped.control_mode = mode
        mode_msg.modestamped.mode_type = 0  # DEFAULT_TYPE
        self.mode = mode
        return self._action_client.send_goal_async(mode_msg)

    def get_last_mode(self):
        return self.mode


class ScanDelegate(DefaultDelegate):

    def __init__(self):
        DefaultDelegate.__init__(self)

    def handleDiscovery(self, dev, isNewDev, isNewData):
        global remote_pub, log, scanStatus
        """if isNewDev:
            print ("Discovered device", dev.addr)
        elif isNewData:
            print ("Received new data from", dev.addr)"""
        if str(dev.getValueText(ScanEntry.COMPLETE_LOCAL_NAME)).count(REMOTE_NAME):
            scanStatus = SCAN_STATUS_INFO
            if remote_pub is not None:
                remote_pub.publisher(dev.addr, '{"name":"%s","rssi":"%s"}' %
                                     (REMOTE_NAME, dev.rssi), '')  # json format
                log.logger.debug('handleDiscovery: scanStatus=2(SCAN_STATUS_INFO), '
                                 'addr=%s addrType=%s rssi=%s connectable=%s scanData=%s'
                                 % (dev.addr,
                                    dev.addrType,
                                    dev.rssi,
                                    dev.connectable,
                                    dev.getScanData()))
            # for (adType, desc, value) in dev.getScanData():
            #     log.logger.debug('adType=%d  %s = %s' % (adType, desc, value))


def doScan(_tout):
    global remote_pub, log, scanStatus, autoConAddress

    time.sleep(0.05)  # waiting for service response first
    if scanStatus is not SCAN_STATUS_END:  # publish current and end scan
        if remote_pub is not None:
            remote_pub.publisher('', '', '')
            log.logger.warning('doScan: scanStatus=%d, already scanning' % scanStatus)
            return False

    # create a new scan
    scanStatus = SCAN_STATUS_START
    if remote_pub is not None:
        remote_pub.publisher('', '', '')
        log.logger.debug('doScan: scanStatus=1(SCAN_STATUS_START)')

    scanner = Scanner().withDelegate(ScanDelegate())
    for i in range(RETRY_NUM):
        try:
            log.logger.info('doScan: start for %d seconds' % _tout)
            devices = scanner.scan(timeout=_tout)
            break
        except bluepy.btle.BTLEException as e:
            scanStatus = SCAN_STATUS_ERROR
            if remote_pub is not None:
                remote_pub.publisher('', '', e)
                log.logger.error('doScan: scanStatus=4(SCAN_STATUS_ERROR), '
                                 'error=%s, wait for scan end' % e)
            if i + 1 >= RETRY_NUM:
                log.logger.warning('doScan: failed and quit!')
                scanStatus = SCAN_STATUS_END
                if remote_pub is not None:
                    remote_pub.publisher('', '', '')
                    log.logger.warning('doScan: scanStatus=3(SCAN_STATUS_END), scan error and end')
                return False
            else:
                log.logger.warning('doScan: failed and retry again %d/%d' % (i + 2, RETRY_NUM))
            time.sleep(TIME_SHORT_SLEEP)

    foundTotalNum = 0
    foundDeviceNum = 0
    foundDevice: bool = False

    for dev in devices:
        foundTotalNum += 1
        if str(dev.getValueText(ScanEntry.COMPLETE_LOCAL_NAME)).count(REMOTE_NAME):
            foundDeviceNum += 1
            """print ("Device %s (%s), RSSI=%d dB" % (dev.addr, dev.addrType, dev.rssi))
            for (adtype, desc, value) in dev.getScanData():
                print ("adtype=%d  %s = %s" % (adtype, desc, value))"""
            if dev.addr == autoConAddress and '' != autoConAddress:
                foundDevice = True
    log.logger.info('doScan: found device %s num is %d/%d' %
                    (REMOTE_NAME, foundDeviceNum, foundTotalNum))
    scanStatus = SCAN_STATUS_END  # scan ended
    if remote_pub is not None:
        remote_pub.publisher('', '', '')
        log.logger.debug('doScan: scanStatus=3(SCAN_STATUS_END), scan success')
    if gattserver.enableAutoRemoteControl is True:
        startAutoConnect()
    return foundDevice


def calculateData(_data, _min, _max):
    if 200 <= _data < 900:
        return float(_min * (900 - _data) / 700)
    elif 1100 < _data <= 1800:
        return float(_max * (_data - 1100) / 700)
    else:
        return 0


def updateMove(rH, rV, lV, lH):
    global global_gait, key_pub
    update: bool = True
    if 0xF7 == global_gait:
        linear_x = calculateData(lV, -1, 1)
        linear_y = calculateData((2000 - rH), -0.75, 0.75)
        angular_z = calculateData((2000 - lH), -1.5, 1.5)
    elif 0xF8 == global_gait:
        linear_x = calculateData(lV, -1.3, 1.3)
        linear_y = calculateData((2000 - rH), -0.975, 0.975)
        angular_z = calculateData((2000 - lH), -1.3, 1.3)
    elif 0xF9 == global_gait:
        linear_x = calculateData(lV, -1.6, 1.6)
        linear_y = calculateData((2000 - rH), -1.2, 1.2)
        angular_z = calculateData((2000 - lH), -1.6, 1.6)
    elif 0xFA == global_gait:
        linear_x = calculateData(lV, -0.1, 0.1)
        linear_y = calculateData((2000 - rH), -0.1, 0.1)
        angular_z = calculateData((2000 - lH), -0.3, 0.3)
    elif 0xFB == global_gait:
        linear_x = calculateData(lV, -0.1, 0.1)
        linear_y = calculateData((2000 - rH), -0.1, 0.1)
        angular_z = calculateData((2000 - lH), -0.3, 0.3)
    else:
        update = False

    if update is True:
        angular_y = calculateData((2000 - rV), -0.2, 0.25)
        if linear_x == 0 and linear_y == 0 and angular_y == 0 and angular_z == 0:
            if key_pub is not None:
                if key_pub.get_zero() is False:
                    key_pub.publisher(0, 0, 0, 0)
                    key_pub.set_zero(True)
        else:
            if key_pub is not None:
                if key_pub.get_zero() is True:
                    key_pub.set_zero(False)
                key_pub.publisher(linear_x, linear_y, angular_y, angular_z)
                # log.logger.debug('updateMove: gait=%d, lx=%f, ly=%f, ay=%f, az=%f' %
                #                  ((global_gait&0x0F), linear_x, linear_y, angular_y, angular_z))


def calculatePara(data):
    if data < 0x03E8:
        return 0
    elif data == 0x03E8:
        return 1
    else:
        return 2


def num_to_gait(num):
    numbers = {
        0: 1, 1: 2, 2: 3, 3: 7, 4: 8, 5: 9, 6: 4, 7: 10, 8: 11
    }
    return numbers.get(num, None)


def updateGait(ch5, ch6, ch7):
    global global_gait, gait_action, global_gait
    if 0x0708 == ch6:  # gait range: 1~4, 7~11, when ch6 pressed
        num = (calculatePara(ch7) * 3) + calculatePara(ch5)
        gait = num_to_gait(num)
        if 0xF0 == global_gait or 0 == global_gait:
            if mode_action is not None:
                mode_action.send_goal(mode_action.MODE_MANUAL)
                log.logger.debug('updateGait: action mode MANUAL')
        if global_gait != gait:
            global_gait = gait
            gait_action.send_goal(global_gait)
            log.logger.debug('updateGait: ch6 pressed, and publish gait is %d' % global_gait)
    else:  # gait range: 0xF1~0xF4, 0xF7~0xFB, when ch6 up
        if global_gait < 0xF0:
            global_gait |= 0xF0


def cleanWhenDisconnect():
    global global_gait, peripDevice
    global_gait = 0
    peripDevice = None


def updateBodyHeight(ch8):
    global global_body_height, para_pub
    if ch8 < 467:
        height = 0.26
    elif ch8 < 734:
        height = 0.27
    elif ch8 < 1000:
        height = 0.28
    elif ch8 < 1266:
        height = 0.29
    elif ch8 < 1533:
        height = 0.31
    else:
        height = 0.32
    if global_body_height == 0:
        global_body_height = height
    elif global_body_height != height:
        global_body_height = height
        if para_pub is not None:
            para_pub.publisher(global_body_height)
            log.logger.debug('updateBodyHeight: body height=%f' % global_body_height)


class MyDelegate(DefaultDelegate):
    # Constructor (run once on startup)
    def __init__(self, params):
        DefaultDelegate.__init__(self)

    # func is caled on notifications
    def handleNotification(self, cHandle, data):
        global log, remoteStatus, remote_pub, peripDevice

        if remoteStatus is REMOTE_STATUS_CONNECTED:
            remoteStatus = REMOTE_STATUS_NOTIFY
            if remote_pub is not None:
                remote_pub.publisher(peripDevice.addr, '', '')
                log.logger.debug('handleNotification: remoteStatus=8(REMOTE_STATUS_NOTIFY), '
                                 'addr=%s' % peripDevice.addr)
            if gattserver.audioClient.check_ready():
                gattserver.audioClient.send_goal(gattserver.audioClient.REMOTE_CONNECT_SUCCESS)

        if remoteStatus is not REMOTE_STATUS_NOTIFY:
            log.logger.warning('remoteStatus=%s, waiting for connect end' % remoteStatus)
        else:
            if 20 == len(data) and 0x18 == data[1]:
                # log.logger.info('Notification from Handle: 0x' + \
                #                  format(cHandle, '02X') + ' Value: ' + format(data))
                rH = (data[2] << 8) + data[3]  # rightHorizontal
                rV = (data[4] << 8) + data[5]  # rightVertical
                lV = (data[6] << 8) + data[7]  # leftVertical
                lH = (data[8] << 8) + data[9]  # leftHorizontal
                updateMove(rH, rV, lV, lH)

                # only update once when Key_ch6 pressed
                channel5 = (data[10] << 8) + data[11]
                channel6 = (data[12] << 8) + data[13]
                channel7 = (data[14] << 8) + data[15]
                updateGait(channel5, channel6, channel7)

                channel8 = (data[16] << 8) + data[17]
                updateBodyHeight(channel8)


def doConnect(addr):
    global peripDevice, log, remoteStatus, remote_pub, mode_action

    time.sleep(0.05)  # waiting for service response first
    # if peripDevice is not None and str(peripDevice.addr).count(addr) is False:
    #     log.logger.debug('some devices is already connected, do disconnect first!!!')
    #     doDisconnect(peripDevice.addr)

    if remoteStatus is not REMOTE_STATUS_DISCONNECTED:
        if remote_pub is not None:
            remote_pub.publisher(peripDevice.addr, '', '')
            log.logger.warning('doConnect: remoteStatus=%d, addr=%s, should disconnect device '
                               'first' % (remoteStatus, peripDevice.addr))
            return

    if len(addr.split(':')) != 6:
        log.logger.error('Expected MAC address, got %s, doConnect failed' % addr)
        return

    if gattserver.audioClient.check_ready():
        gattserver.audioClient.send_goal(gattserver.audioClient.REMOTE_CONNECTING)

    log.logger.info('doConnect: start to connect %s' % addr)
    cleanWhenDisconnect()
    remoteStatus = REMOTE_STATUS_CONNECTING
    if remote_pub is not None:
        remote_pub.publisher(addr, '', '')
        log.logger.debug('doConnect: remoteStatus=5(REMOTE_STATUS_CONNECTING), addr=%s' % addr)

    # connect device
    for i in range(RETRY_NUM):
        try:
            if enableConnectTimeout is True:
                log.logger.info('doConnect: create time for connect timeout')
                conStartTimeout(10)
            peripDevice = Peripheral(addr, bluepy.btle.ADDR_TYPE_RANDOM)

            # displays all services
            # services = peripDevice.getServices()
            # for service in services:
            #     print(service)

            # displays all characteristics
            # chList = peripDevice.getCharacteristics()
            # print("Handle   UUID                                Properties")
            # print("-------------------------------------------------------")
            # for ch in chList:
            #     print("  0x" + format(ch.getHandle(), '02X') + " " + str(ch.uuid) +
            #           " " + ch.propertiesToString())
            break
        except bluepy.btle.BTLEException as e:
            log.logger.info('doConnect: Peripheral get error=%s' % e)
            if str(e).count('Device disconnected') or str(e).count('object has no attribute'):
                remoteStatus = REMOTE_STATUS_DISCONNECTED
            else:  # Failed to connect peripheral
                remoteStatus = REMOTE_STATUS_CONNECT_ERROR
        finally:
            if enableConnectTimeout is True:
                log.logger.debug('doConnect: clean connect timeout')
                conCleanTimeout()
            if remoteStatus is REMOTE_STATUS_CONNECT_ERROR:
                if remote_pub is not None:
                    remote_pub.publisher(addr, '', '')
                    log.logger.warning('doConnect: remoteStatus=7(REMOTE_STATUS_CONNECT_ERROR), '
                                       'addr=%s' % addr)
                remoteStatus = REMOTE_STATUS_DISCONNECTED
            if remoteStatus is REMOTE_STATUS_DISCONNECTED:
                if gattserver.enableAutoRemoteControl is True:
                    startAutoConnect()
                cleanWhenDisconnect()
                if gattserver.audioClient.check_ready():
                    gattserver.audioClient.send_goal(gattserver.audioClient.REMOTE_DISCONNECTED)
                if remote_pub is not None:
                    remote_pub.publisher(addr, '', '')
                    log.logger.error('doConnect: remoteStatus=10(REMOTE_STATUS_DISCONNECTED), '
                                     'addr=%s, error and force to disconnect' % addr)
                if mode_action is not None:
                    if mode_action.get_last_mode() is mode_action.MODE_MANUAL:
                        mode_action.send_goal(mode_action.MODE_DEFAULT)
                        log.logger.debug('connect failed, action mode default')
                return
        time.sleep(TIME_SHORT_SLEEP)

    # find and read/write att char
    try:
        peripDevice.setDelegate(MyDelegate(peripDevice))
        KeyService = peripDevice.getServiceByUUID(KEY_SERVICE_UUID)
        KeyChar = KeyService.getCharacteristics(KEY_CHAR_UUID)[0]
        handleKey = KeyChar.getHandle()
        log.logger.debug('get handleKey is: 0x' + format(handleKey, '02X'))

        for descriptor in peripDevice.getDescriptors(handleKey, handleKey + 0x0003):
            if descriptor.uuid == KEY_CHAR_CCC_UUID:
                log.logger.debug('KeyClient Characteristic Configuration found at handle 0x' +
                                 format(descriptor.handle, '02X'))
                hButtonCCC = descriptor.handle
                peripDevice.writeCharacteristic(hButtonCCC, struct.pack('<bb', 0x01, 0x00))
                log.logger.debug('Notification is turned on for KeyClient')
    except bluepy.btle.BTLEException as e:
        log.logger.error('doConnect: att read/write error=%s' % e)
        # if str(e).count('Device disconnected') or str(e).count('object has no attribute'):
        remoteStatus = REMOTE_STATUS_DISCONNECTED
    finally:
        if remoteStatus is REMOTE_STATUS_DISCONNECTED:
            cleanWhenDisconnect()
            if gattserver.audioClient.check_ready():
                gattserver.audioClient.send_goal(gattserver.audioClient.REMOTE_DISCONNECTED)
            if remote_pub is not None:
                remote_pub.publisher(addr, '', '')
                log.logger.debug('doConnect: read/write att failed, publish this error')
            if gattserver.enableAutoRemoteControl is True:
                startAutoConnect()
            return

    remoteStatus = REMOTE_STATUS_CONNECTED
    if remote_pub is not None:
        remote_pub.publisher(peripDevice.addr, '', '')
        log.logger.debug('doConnect: remoteStatus=6(REMOTE_STATUS_CONNECTED), addr=%s, '
                         'connect success and wait for notify' % peripDevice.addr)

    if peripDevice.addr:
        writeAutoAddress(peripDevice.addr)  # write device address

    log.logger.debug('doConnect end, waiting for notification')
    while True:
        try:
            if peripDevice.waitForNotifications(WAITING_FOR_NOTIFY_TIMEOUT):
                # handleNotification()
                continue
            log.logger.debug('Waiting... Waited more than 1 sec for notification')
        except bluepy.btle.BTLEException as e:
            # when disconnected
            if str(e).count('Device disconnected') or str(e).count('object has no attribute'):
                log.logger.debug('doConnect: device disconnected, error=%s' % e)
            else:
                remoteStatus = REMOTE_STATUS_CONNECT_ERROR
                if remote_pub is not None:
                    remote_pub.publisher(peripDevice.addr, '', e)
                    log.logger.debug('doConnect: remoteStatus=7(REMOTE_STATUS_CONNECT_ERROR), '
                                     'addr=%s, error=%s' % (peripDevice.addr, e))
                time.sleep(TIME_SHORT_SLEEP)
            # auto disconnect
            if remoteStatus is REMOTE_STATUS_CONNECTED or \
                    remoteStatus is REMOTE_STATUS_CONNECT_ERROR or \
                    remoteStatus is REMOTE_STATUS_NOTIFY:  # remote disconnect
                remoteStatus = REMOTE_STATUS_DISCONNECTED
                if remote_pub is not None:
                    remote_pub.publisher(peripDevice.addr, '', '')
                    log.logger.debug('doConnect: remoteStatus=10'
                                     '(REMOTE_STATUS_DISCONNECTED), addr=%s, '
                                     'notify error and disconnect' % peripDevice.addr)
                if mode_action is not None:
                    if mode_action.get_last_mode() is mode_action.MODE_MANUAL:
                        mode_action.send_goal(mode_action.MODE_DEFAULT)
                        log.logger.debug('connect failed or disconnected, action mode default')
                cleanWhenDisconnect()
                if gattserver.audioClient.check_ready():
                    gattserver.audioClient.send_goal(gattserver.audioClient.REMOTE_DISCONNECTED)
                if gattserver.enableAutoRemoteControl is True:
                    startAutoConnect()
            else:
                log.logger.debug('doConnect: remoteStatus is %d, '
                                 'update in disconnect thread' % remoteStatus)
            break
        time.sleep(TIME_SHORT_SLEEP)


def conStartTimeout(tout):
    global conTimer
    conCleanTimeout()
    conTimer = threading.Timer(tout, conTimeoutFun)
    conTimer.start()


def conTimeoutFun():
    global peripDevice, remoteStatus, conTimer
    conTimer = None
    log.logger.debug('timeout for connect device')
    if remoteStatus is REMOTE_STATUS_CONNECTING and peripDevice is not None:
        remoteStatus = REMOTE_STATUS_DISCONNECTING
        log.logger.debug('force to disconnect device')
        peripDevice.disconnect()
    remoteStatus = REMOTE_STATUS_DISCONNECTED


def conCleanTimeout():
    global conTimer
    if conTimer is not None:
        conTimer.cancel()
        conTimer = None


def doDisconnect(addr):
    global peripDevice, log, remoteStatus, remote_pub, mode_action
    time.sleep(0.05)  # waiting for service response first
    log.logger.info('doDisconnect: get remoteStatus=%d, addr=%s' % (remoteStatus, addr))
    if REMOTE_STATUS_DISCONNECTING <= remoteStatus:
        if remote_pub is not None:
            remote_pub.publisher('', '', '')
            log.logger.warning('doDisconnect: remoteStatus=%d, '
                               'failed and already disconnect' % remoteStatus)
            return

    if len(addr.split(':')) != 6:
        log.logger.debug('Expected MAC address, got %s, doConnect failed' % addr)
        return
    # if peripDevice is None or str(peripDevice.addr).count(addr) is not True:
    #     log.logger.warning('doDisconnect error: current addr=%s, receive addr=%s, '
    #                      'remoteStatus=%s' % (peripDevice.addr, addr, remoteStatus))
    # return

    remoteStatus = REMOTE_STATUS_DISCONNECTING
    if remote_pub is not None:
        remote_pub.publisher(addr, '', '')
        log.logger.debug('doDisconnect: remoteStatus=9'
                         '(REMOTE_STATUS_DISCONNECTING), addr=%s' % addr)
    try:
        peripDevice.disconnect()
    except bluepy.btle.BTLEException as e:  # notice: disconnection from doConnect and doDisconnect
        log.logger.warning('doDisconnect: failed and error=%s' % e)
        if str(e).count('Device disconnected') or str(e).count('object has no attribute'):
            log.logger.debug('device disconnected, error=%s' % e)
        else:
            remoteStatus = REMOTE_STATUS_DISCONNECT_ERROR
            if remote_pub is not None:
                remote_pub.publisher(addr, '', e)
                log.logger.debug('doDisconnect: remoteStatus=11(REMOTE_STATUS_DISCONNECT_ERROR), '
                                 'addr=%s, error=%s' % (addr, e))
    finally:
        if remoteStatus is not REMOTE_STATUS_DISCONNECTED:
            remoteStatus = REMOTE_STATUS_DISCONNECTED
            if remote_pub is not None:
                remote_pub.publisher(addr, '', '')
                log.logger.debug('doDisconnect: remoteStatus=10'
                                 '(REMOTE_STATUS_DISCONNECTED), addr=%s' % addr)
            if mode_action is not None:
                if mode_action.get_last_mode() is mode_action.MODE_MANUAL:
                    mode_action.send_goal(mode_action.MODE_DEFAULT)
                    log.logger.debug('doDisconnect: action mode default')
            cleanWhenDisconnect()
            if gattserver.audioClient.check_ready():
                gattserver.audioClient.send_goal(gattserver.audioClient.REMOTE_DISCONNECTED)
            # startAutoConnect()  # do not startAutoConnect when disconnect by user
        else:
            log.logger.debug('remoteStatus is already update in connect thread')
        log.logger.debug('doDisconnect: success and end')


def getAllStatus():
    global peripDevice, log, remoteStatus, remote_pub, scanStatus
    time.sleep(0.05)  # waiting for service response first
    if remote_pub is not None:
        if remoteStatus is REMOTE_STATUS_DISCONNECTED:
            remote_pub.publisher('', '', '')
            log.logger.debug('getAllStatus: scanStatus=%d, remoteStatus=%d' %
                             (scanStatus, remoteStatus))
        else:
            remote_pub.publisher(peripDevice.addr, '', '')
            log.logger.debug('getAllStatus: scanStatus=%d, remoteStatus=%d, addr=%s' %
                             (scanStatus, remoteStatus, peripDevice.addr))


# check and disconnect current device
def checkAndAutoDisconnect():
    global peripDevice, log, remoteStatus
    log.logger.info('checkAndAutoDisconnect: get remoteStatus is %d' % remoteStatus)
    if remoteStatus is REMOTE_STATUS_DISCONNECTED:
        log.logger.debug('checkAndAutoDisconnect: already disconnected')
        pass
    elif remoteStatus <= REMOTE_STATUS_NOTIFY:
        if peripDevice is not None:
            doDisconnect(peripDevice.addr)
            time.sleep(5)  # wait more than 3s for RC disconnected
    else:  # waiting for disconnect
        while remoteStatus is not REMOTE_STATUS_DISCONNECTED:
            time.sleep(0.05)
        log.logger.debug('checkAndAutoDisconnect: device is disconnected')


def getlog():
    global log
    if log is None:
        log = record.logAll


def readAutoAddress():
    if os.path.exists(autoAddressPath):
        with open(autoAddressPath, 'r') as f:
            addr = f.read()
            if len(addr.split(':')) != 6:
                os.remove(autoAddressPath)
                return ''
            else:
                return addr
    else:
        return ''


def writeAutoAddress(addr):
    global autoConAddress
    with open('/etc/btRemoteAddress.conf', 'w') as f:
        f.write(addr)
        autoConAddress = addr


def delAutoAddress(addr):
    global autoConAddress
    time.sleep(0.05)  # waiting for service response first
    autoConAddress = readAutoAddress()
    if autoConAddress == addr and '' != autoConAddress:
        os.remove(autoAddressPath)
        autoConAddress = ''
    else:
        log.logger.error('current addr=%s, receive addr=%s' % (autoConAddress, addr))


# only idle and waiting mode can start
# True means run, False means already run
def startAutoConnect():
    global log, autoConThreadRunning, autoConTimer
    log.logger.info('startAutoConnect: get autoConThreadRunning is %d' % autoConThreadRunning)
    if autoConThreadRunning is AUTO_CONNECT_THREAD_IDLE or \
            autoConThreadRunning is AUTO_CONNECT_THREAD_WAITING:
        if autoConTimer is not None:  # clean timer
            autoConTimer.cancel()
            autoConTimer = None
        autoConThreadRunning = AUTO_CONNECT_THREAD_WAITING
        autoConTimer = threading.Timer(AUTO_CONNECT_START_TIME, autoConnection)  # reset timer
        autoConTimer.start()
        log.logger.info('startAutoConnect: waiting %d seconds for start' % AUTO_CONNECT_START_TIME)
        return True
    else:
        log.logger.info('startAutoConnect: already started')
        return False


# must stop auto connection thread
# True means stop success, False means stop failed
def stopAutoConnect():
    global log, autoConThreadRunning, autoConTimer
    log.logger.info('stopAutoConnect: get autoConThreadRunning is %d' % autoConThreadRunning)
    if autoConThreadRunning is AUTO_CONNECT_THREAD_IDLE:
        pass
    elif autoConThreadRunning is AUTO_CONNECT_THREAD_WAITING:
        if autoConTimer is not None:  # clean timer
            autoConTimer.cancel()
            autoConTimer = None
        autoConThreadRunning = AUTO_CONNECT_THREAD_IDLE
    elif autoConThreadRunning is AUTO_CONNECT_THREAD_ENTER or \
            autoConThreadRunning is AUTO_CONNECT_THREAD_SCAN or \
            autoConThreadRunning is AUTO_CONNECT_THREAD_CONNECT:
        autoConThreadRunning = AUTO_CONNECT_THREAD_QUIT
        autoConTimer.join()
        autoConTimer = None
    elif autoConThreadRunning is AUTO_CONNECT_THREAD_QUIT:
        if autoConTimer is not None:
            autoConTimer.join()
            autoConTimer = None
    log.logger.info('stopAutoConnect: end')
    return True


def autoConnection():
    global log, autoConAddress, autoConThreadRunning, remoteStatus

    if autoConThreadRunning is not AUTO_CONNECT_THREAD_WAITING:  # idle or running
        log.logger.warning('autoConnection: autoConThreadRunning is %d not '
                           'AUTO_CONNECT_THREAD_WAITING, quit' % autoConThreadRunning)
        return
    else:
        autoConThreadRunning = AUTO_CONNECT_THREAD_ENTER  # enter for prepare something
        log.logger.info('autoConnection: THREAD WAITING-> ENTER')

    while autoConThreadRunning is AUTO_CONNECT_THREAD_ENTER:
        # 1 check address from config file
        autoConAddress = readAutoAddress()
        if '' == autoConAddress:
            log.logger.warning('auto device is null')
            autoConThreadRunning = AUTO_CONNECT_THREAD_IDLE
            break

        # 2 check connection status, do nothing when connected
        if remoteStatus is not REMOTE_STATUS_DISCONNECTED:
            log.logger.warning('get remoteStatus=%d, stop auto connection' % remoteStatus)
            autoConThreadRunning = AUTO_CONNECT_THREAD_IDLE
            break

        # 3 change to SCAN status ,do auto scan
        if autoConThreadRunning is not AUTO_CONNECT_THREAD_ENTER:
            log.logger.warning('autoConnection: autoConThreadRunning is %d not '
                               'AUTO_CONNECT_THREAD_ENTER' % autoConThreadRunning)
            autoConThreadRunning = AUTO_CONNECT_THREAD_IDLE
            break
        else:
            autoConThreadRunning = AUTO_CONNECT_THREAD_SCAN
            log.logger.info('autoConnection: THREAD ENTER -> SCAN')

        # 4 scan device
        scanResult: bool = False  # False means do not find RC
        while scanResult is False and \
                autoConThreadRunning is AUTO_CONNECT_THREAD_SCAN:  # scan not device and scan again
            scanResult = doScan(SCAN_SHORT_TIMEOUT)  # only scan 1 second
            if scanResult is True:  # sleep 0.5s and quit when find device
                time.sleep(TIME_SHORT_SLEEP)  # sleep for connect
            else:  # wait 10*0.5s, and check thread
                for i in range(10):
                    if autoConThreadRunning is not AUTO_CONNECT_THREAD_SCAN:
                        break
                    time.sleep(TIME_SHORT_SLEEP)

        # change to CONNECTION status, do auto connection
        if autoConThreadRunning is not AUTO_CONNECT_THREAD_SCAN:
            log.logger.warning('autoConnection: autoConThreadRunning is %d not '
                               'AUTO_CONNECT_THREAD_SCAN' % autoConThreadRunning)
            autoConThreadRunning = AUTO_CONNECT_THREAD_IDLE
            break
        else:
            autoConThreadRunning = AUTO_CONNECT_THREAD_CONNECT
            log.logger.info('autoConnection: THREAD SCAN -> CONNECT')

        # 5 deal scan device
        if scanResult is True and remoteStatus is REMOTE_STATUS_DISCONNECTED:
            # create task for connect
            t = Thread(target=doConnect, args=(autoConAddress,))
            t.start()
            time.sleep(0.1)  # must long then 0.05
        else:
            log.logger.warning('scanResult=%d, remoteStatus=%d' % (scanResult, remoteStatus))
            autoConThreadRunning = AUTO_CONNECT_THREAD_IDLE
            break

        # 6 waiting for change to REMOTE_STATUS_CONNECTING
        if remoteStatus is REMOTE_STATUS_DISCONNECTED:
            i = 0
            for i in range(5):
                if remoteStatus is REMOTE_STATUS_CONNECTING:
                    break
                else:
                    time.sleep(0.05)
            if i >= 5:
                log.logger.warning('doConnect failed???')

        # 7 check connection result
        while True:
            # connect failed, do thread again
            if remoteStatus is REMOTE_STATUS_DISCONNECTED:
                autoConThreadRunning = AUTO_CONNECT_THREAD_ENTER
                log.logger.warning('autoConnection: THREAD CONNECT -> ENTER')
                break
            # connect success, quit auto connect thread
            elif remoteStatus is REMOTE_STATUS_CONNECTED or \
                    remoteStatus is REMOTE_STATUS_NOTIFY:
                autoConThreadRunning = AUTO_CONNECT_THREAD_IDLE
                log.logger.info('autoConnection: THREAD CONNECT -> IDLE')
                break
            else:
                time.sleep(0.5)
    log.logger.debug('autoConnection: autoConThreadRunning is %d and quit' % autoConThreadRunning)
    if autoConThreadRunning is AUTO_CONNECT_THREAD_QUIT:
        autoConThreadRunning = AUTO_CONNECT_THREAD_IDLE
    return
