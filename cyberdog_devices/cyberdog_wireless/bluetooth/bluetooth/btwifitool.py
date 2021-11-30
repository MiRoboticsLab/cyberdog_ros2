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

import operator
import os
import subprocess

try:
    from . import record
except ImportError:
    import record

log: None = None
mainStatus = -1


STATUS_NONE = 0
STATUS_WIFI_GET_INFO = 1
STATUS_WIFI_CONNECT = 2
STATUS_WIFI_CONNECTING = 3
STATUS_WIFI_NO_SSID = 4
STATUS_WIFI_ERR_PWD = 5
STATUS_WIFI_OTHER = 6
STATUS_WIFI_SUCCESS = 7
STATUS_ROS2_RESTARTING = 8
STATUS_ROS2_START_PRE = 9
STATUS_ROS2_START = 10
STATUS_ROS2_START_POST = 11
STATUS_ROS2_RUNNING = 12
STATUS_ROS2_OTHER = 13
STATUS_WIFI_INTERRUPT = 14
STATUS_WIFI_TIMEOUT = 15

NETWORK_PATH = '/etc/NetworkManager/system-connections/'


def getlog():
    global log
    if log is None:
        log = record.logAll


# get current connection ssid, null if there is no connection.
def activeWifiSsid():
    cmd = 'sudo iwgetid wlan0 -r'
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output.wait()
    tmp = str(output.stdout.read(), encoding='utf-8')
    return tmp[:-1]


# gat current connection bssid, null if there is no connection.
def activeWifiBssid():
    # do not use lowercase
    # cmd = 'iw wlan0 link | grep "Connected to " | awk -F " " \'{print $3}\''
    # output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # output.wait()
    # tmp = str(output.stdout.read(), encoding = "utf-8")
    # return tmp[:-1]  # null for no connection
    if activeWifiSsid() == '':
        return ''
    # get capital letter for bssid, 00:00:00:00:00:00 for no connection
    cmd = 'sudo iwgetid wlan0 -a | awk -F " " \'{print $4}\''
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output.wait()
    tmp = str(output.stdout.read(), encoding='utf-8')
    return tmp[:-1]


# do a new wlan connect
def nmcliConnectWifi(ssid, pwd):
    # global log
    cmd = "sudo nmcli device wifi connect '"
    cmd += ssid
    cmd += "' password '"
    cmd += pwd
    cmd += "'"
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output.wait()  # wait for cmd done
    tmp = str(output.stdout.read(), encoding='utf-8')
    return tmp


# parsing the wlan connect/reconnect return value
def return_connect_status(output):
    nossid = 'No network with SSID'
    errorpwd = 'Secrets were required, but not provided'
    connected = 'successfully activated'
    interrupt = 'The base network connection was interrupted'
    activatefail = 'Connection activation failed'
    timeout = 'Timeout expired'
    # global log
    # log.logger.info('output' + repr(output))
    if operator.contains(output, connected):
        return STATUS_WIFI_SUCCESS
    elif operator.contains(output, errorpwd):
        return STATUS_WIFI_ERR_PWD
    elif operator.contains(output, nossid):
        return STATUS_WIFI_NO_SSID
    elif operator.contains(output, interrupt) or operator.contains(output, activatefail):
        return STATUS_WIFI_INTERRUPT
    elif operator.contains(output, timeout):
        return STATUS_WIFI_TIMEOUT
    else:
        return STATUS_WIFI_OTHER


# check if ssid has been connected
def isContainSSID(ssid):
    # lscmd = 'sudo ls '
    # lscmd += NETWORK_PATH
    # lscmd += " |grep -w '"
    # lscmd += ssid
    # lscmd += "'"
    # output = subprocess.Popen(lscmd,
    #                           shell=True,
    #                           stdout=subprocess.PIPE,
    #                           stderr=subprocess.STDOUT)
    # tmp = str(output.stdout.read(), encoding="utf-8")
    # if "" == tmp:
    #     return False
    # else:
    #     return True

    # The second way to check file
    # for root, dirs, files in os.walk(r"/etc/NetworkManager/system-connections"):
    #     for file in files:
    #         if ssid == os.path.join(file):
    #             return True
    # return False

    # The third way to check file
    dirs = os.listdir(NETWORK_PATH)
    for file in dirs:
        with open(NETWORK_PATH+file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.find('id=') != -1:
                    log.logger.debug('isContainSSID get this id=' + repr(line[3:-1]))
                    if line[3:-1] == ssid:
                        return True
    return False


# get a pwd from ssid file
def getPwd(ssid):
    # catcmd = "sudo cat "
    # catcmd += NETWORK_PATH
    # catcmd += "'"
    # catcmd += ssid
    # catcmd += "'|grep psk="
    # output = subprocess.Popen(catcmd,
    #                           shell=True,
    #                           stdout=subprocess.PIPE,
    #                           stderr=subprocess.STDOUT)
    # return output.stdout.read()[4:-1]

    # The second way to find pwd
    dirs = os.listdir(NETWORK_PATH)
    for file in dirs:
        with open(NETWORK_PATH + file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.find('id=') != -1:
                    log.logger.debug('getPwd get this id=' + repr(line[3:-1]))
                    if line[3:-1] == ssid:
                        for lin in lines:
                            if lin.find('psk=') != -1:
                                log.logger.debug('getPwd get this pwd=' + repr(lin[4:-1]))
                                return lin[4:-1]


# change the pwd in the ssid file
def exchangePSK(ssid, pwd):
    sedcmd = "sudo sed -i 's/^psk=.*/psk="
    sedcmd += pwd  # do not need ' '
    sedcmd += "/' "
    sedcmd += NETWORK_PATH
    sedcmd += "'"

    # sedcmd += ssid
    dirs = os.listdir(NETWORK_PATH)
    for file in dirs:
        with open(NETWORK_PATH + file, 'r+') as f:
            lines = f.readlines()
            for line in lines:
                if line.find('ssid=') != -1:
                    if line[5:-1] == ssid:
                        sedcmd += file
    sedcmd += "'"
    output = subprocess.Popen(sedcmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tmp = str(output.stdout.read(), encoding='utf-8')
    return tmp


# disconnect wlan
def disWifi():
    discmd = 'sudo nmcli dev dis wlan0'
    output = subprocess.Popen(discmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output.wait()
    tmp = str(output.stdout.read(), encoding='utf-8')
    return tmp


# parsing the disconnect cmd return value
def getDisStatus(output):
    dissuccess = 'successfully disconnected'
    disagain1 = 'This device is not active'
    disagain2 = 'devices disconnected'
    if operator.contains(output, dissuccess) or \
            operator.contains(output, disagain1) or \
            operator.contains(output, disagain2):
        return True
    else:
        return False


# restart network-manager
def restartNetworkService():
    restartcmd = 'sudo systemctl restart network-manager '
    output = subprocess.Popen(restartcmd,
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    output.wait()


# reconnect a ssid, like: nmcli connection up ifname wlan0, nmcli device connect wlan0
def connectUpWifi(ssid):
    restartcmd = "sudo nmcli connect up '"
    restartcmd += ssid
    restartcmd += "'"
    output = subprocess.Popen(restartcmd,
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    output.wait()
    tmp = str(output.stdout.read(), encoding='utf-8')
    return tmp


# get current robot ip
def activeWifiIP():
    ipcmd = 'sudo ifconfig wlan0 | grep "inet " | awk -F " " \'{print $2}\''
    output = subprocess.Popen(ipcmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ip = str(output.stdout.read(), encoding='utf-8')
    return ip[:-1]


# check phone connection status
def bleConnectStatus():
    if '' == getConnectLEAddress():
        return False
    else:
        return True


# check phone bt mac
def getConnectLEAddress():
    cmd = 'hcitool con | grep "> LE" | awk -F " " \'{print $3}\''
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output.wait()
    address = str(output.stdout.read(), encoding='utf-8')
    # log.logger.info("read add is:" + address)
    return address


# get phone connection handle
def getConnectLEHandle():
    cmd = 'hcitool con | grep "> LE" | awk -F " " \'{print $5}\''
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output.wait()
    handle = str(output.stdout.read(), encoding='utf-8')
    return handle


# disconnect phone by handle
def disconnectLEDevice(handle):
    if '' == handle:
        log.logger.debug('not need to disconnect')
    else:
        log.logger.debug('The handle ' + repr(handle) + 'is going to disconnect')
        cmd = 'hcitool ledc'
        cmd += handle
        output = subprocess.Popen(cmd,
                                  shell=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
        output.wait()


# get ros2 status
def getRos2Status():
    statuscmd = 'sudo systemctl status cyberdog_ros2.service |grep Active |awk -F " " '
    statuscmd += "'{print $3}'"
    output = subprocess.Popen(statuscmd,
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    statusback = output.stdout.read().decode('utf-8')
    ros2_numbers = {
        'start-pre': STATUS_ROS2_START_PRE,
        'start': STATUS_ROS2_START,
        'start-post': STATUS_ROS2_START_POST,
        'running': STATUS_ROS2_RUNNING,
    }
    return ros2_numbers.get(statusback[1:-2], STATUS_ROS2_OTHER)


# restart ros2
def restartRos2():
    restartcmd = 'sudo systemctl restart cyberdog_ros2.service'
    output = subprocess.Popen(restartcmd,
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    status = STATUS_ROS2_RESTARTING
    output.stdout.read().decode('utf-8')
    return status
