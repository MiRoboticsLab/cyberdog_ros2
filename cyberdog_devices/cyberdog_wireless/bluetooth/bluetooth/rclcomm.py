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

import subprocess

from ception_msgs.srv import SensorDetectionNode
from interaction_msgs.action import AudioPlay
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

try:
    from . import record
except ImportError:
    import record

NODE_CHECK_NUMS = 3
NODE_WAIT_SECS = 5.0

log: None = None


def getlog():
    global log
    if log is None:
        log = record.logAll


def runcmd(command):
    ret = subprocess.run(command, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, encoding='utf-8', timeout=10)
    if ret.returncode == 0:
        print('success:', ret)
    else:
        print('error:', ret)
    return ret.returncode, ret.stdout


def get_namespace():
    global ns_, log
    ret, ns = runcmd('cat /sys/firmware/devicetree/base/serial-number')
    if ret == 0:
        ns = ns.strip()
    else:
        ns = ''
    ns = ns[0:-1]
    n = len(ns)
    if n > 0:
        n = min(n, 7)
        ns = ns[-n:]
        ns_ = 'mi' + ns
    # log.logger.debug('serial number:%s' % (ns_))
    return ns_


class LedClientAsync(Node):
    LED_NUMBER_BLUETOOTH = 2

    TYPE_EFFECTS = 1
    TYPE_FUNCTION = 2
    TYPE_ALARM = 3
    COMMAND_OFF = 0
    COMMAND_ALWAYSON = 0xFFFFFFFFFFFFFFFF

    # led status
    LED_BT_ADV = 27  # HEAD_LED_ORANGE_BLINK
    LED_BT_CONNECTED = 20  # HEAD_LED_DARKBLUE_ON
    LED_WIFI_CONNECTING = 26  # HEAD_LED_DARKBLUE_BLINK
    LED_WIFI_CONNECT_SUCCESS = 20  # HEAD_LED_DARKBLUE_ON
    LED_WIFI_CONNECT_FAILED = 22  # HEAD_LED_ORANGE_ON

    def __init__(self):
        global log
        super().__init__('led_client_async', namespace=get_namespace())
        self.cli = self.create_client(SensorDetectionNode, 'cyberdog_led')
        # while not self.cli.wait_for_service(timeout_sec=5.0):  # modify 1 - 5 seconds
        #     self.get_logger().info('service not available, waiting again...')
        for i in range(NODE_CHECK_NUMS):
            if not self.cli.wait_for_service(timeout_sec=NODE_WAIT_SECS):
                log.logger.warning('led service not available, waiting again...')
            else:
                break
            if i >= NODE_CHECK_NUMS:
                log.logger.error('init led failed!!!!!')
        self.req = SensorDetectionNode.Request()
        self.latest = 0

    def send_request(self, command, priority, timeout):
        self.req.command = command
        self.req.clientid = self.LED_NUMBER_BLUETOOTH
        self.req.priority = priority
        self.req.timeout = timeout
        self.latest = command
        self.future = self.cli.call_async(self.req)

    def get_last_cmd(self):
        return self.latest

    def get_last_always(self):
        if self.req.timeout == self.COMMAND_ALWAYSON:
            return True
        else:
            return False

    def check_ready(self):
        global log
        if self.cli.service_is_ready() is True:
            return True
        else:
            log.logger.warning('led is not connect')
            return False


class PublisherPhoneIP(Node):

    def __init__(self):
        super().__init__('phonepublisher', namespace=get_namespace())
        self.publisher_ = self.create_publisher(String, 'ip_notify', 0)

    def publisher(self, data):
        msg = String()
        msg.data = '%s' % data
        self.publisher_.publish(msg)


class AudioPlayActionClient(Node):
    AUDIO_NUMBER_BLUETOOTH = 6

    BOOT_COMPLETE = 1
    TOUCH_ADV_OR_WIFI_CONNECT_SUCCESS = 4  # touch and start adv, or wifi connect success
    REMOTE_CONNECTING = 5
    REMOTE_CONNECT_SUCCESS = 4
    REMOTE_DISCONNECTED = 6
    TTS_REMOTE_CONNECTING = 105
    TTS_REMOTE_CONNECT_FAILED = 106

    def __init__(self):
        global log
        super().__init__('audioplay_action_client', namespace=get_namespace())
        self._action_client = ActionClient(self, AudioPlay, 'audio_play')
        # self._action_client.wait_for_server()  # waiting for connect server
        for i in range(NODE_CHECK_NUMS):
            if not self._action_client.wait_for_server(timeout_sec=NODE_WAIT_SECS):
                log.logger.info('audio service not available, waiting again...')
            else:
                break
            if i >= NODE_CHECK_NUMS:
                log.logger.error('init audio failed!!!!!')

    def send_goal(self, name_id):
        goal_msg = AudioPlay.Goal()
        goal_msg.order.header.frame_id = 'audio_play'
        goal_msg.order.header.stamp = self.get_clock().now().to_msg()
        goal_msg.order.name.id = name_id  # according AudioSongName.msg, name is song number
        goal_msg.order.user.id = self.AUDIO_NUMBER_BLUETOOTH  # according AudioUser.msg
        # self._action_client.wait_for_server()
        return self._action_client.send_goal_async(goal_msg)

    def check_ready(self):
        global log
        if self._action_client.server_is_ready() is True:
            return True
        else:
            log.logger.warning('audio is not connect')
            return False
