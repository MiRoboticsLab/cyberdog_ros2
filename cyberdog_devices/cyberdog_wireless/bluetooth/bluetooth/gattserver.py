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

import json
import random
import subprocess
from threading import Thread
import time

import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service

try:
    from gi.repository import GObject
except ImportError:
    import gobject as GObject
try:
    from . import btwifitool
    from . import rclcomm
    from . import record
    from . import remote
except ImportError:
    import btwifitool
    import rclcomm
    import record
    import remote

from interaction_msgs.msg import Touch
import rclpy
from rclpy.node import Node

ad_manager = None
service_manager = None
mainloop = None
wifiConnectFlag: bool = False
mainStatus = -1
mainIP = ''
mainSSID = ''
mainPwd = ''
log: None = None
robotAdv = None
robotApp = None
bleStatus = 0
pubPhoneIP = None
notifyChar = None
gnotifystr = {'ssid': None, 'bssid': None, 'ip': None, 'status': 0}
ledClient = None
audioClient = None
wifiConnection: bool = False
checkThreadRun: bool = False
checkThreadID = None
currentPhoneIP = ''  # current phone ip

# define for code
enableInitAdv: bool = True    # auto adv when init
enableHoldTouch: bool = True  # only use hold mode touch
enableAutoRemoteControl: bool = True  # for bluetooth remote control handle auto connect
enableStopAutoDisconnectRC: bool = True
enableReTouchADV: bool = False  # reconfigure adv when touch

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
STATUS_WIFI_FAILED = 16

BLE_STATUS_NONE = 0
BLE_STATUS_ADV = 1
BLE_STATUS_CONNECT = 2
BLE_STATUS_DISCONNECTED = 3


NETWORK_PATH = '/etc/NetworkManager/system-connections/'

BLUEZ_SERVICE_NAME = 'org.bluez'
ADAPTER_INTERFACE = BLUEZ_SERVICE_NAME + '.Adapter1'
DEVICE_INTERFACE = BLUEZ_SERVICE_NAME + '.Device1'
GATT_MANAGER_IFACE = 'org.bluez.GattManager1'
DBUS_OM_IFACE = 'org.freedesktop.DBus.ObjectManager'
DBUS_PROP_IFACE = 'org.freedesktop.DBus.Properties'
LE_ADVERTISING_MANAGER_IFACE = 'org.bluez.LEAdvertisingManager1'
LE_ADVERTISEMENT_IFACE = 'org.bluez.LEAdvertisement1'
GATT_SERVICE_IFACE = 'org.bluez.GattService1'
GATT_CHRC_IFACE = 'org.bluez.GattCharacteristic1'
GATT_DESC_IFACE = 'org.bluez.GattDescriptor1'


class InvalidArgsException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.freedesktop.DBus.Error.InvalidArgs'


class NotSupportedException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.bluez.Error.NotSupported'


class NotPermittedException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.bluez.Error.NotPermitted'


class InvalidValueLengthException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.bluez.Error.InvalidValueLength'


class FailedException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.bluez.Error.Failed'


class Advertisement(dbus.service.Object):
    PATH_BASE = '/org/bluez/example/advertisement'

    def __init__(self, bus, index, advertising_type):
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.ad_type = advertising_type
        self.service_uuids = None
        self.manufacturer_data = None
        self.solicit_uuids = None
        self.service_data = None
        self.local_name = None
        self.include_tx_power = None
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        properties = {}
        properties['Type'] = self.ad_type
        if self.service_uuids is not None:
            properties['ServiceUUIDs'] = dbus.Array(self.service_uuids,
                                                    signature='s')
        if self.solicit_uuids is not None:
            properties['SolicitUUIDs'] = dbus.Array(self.solicit_uuids,
                                                    signature='s')
        if self.manufacturer_data is not None:
            properties['ManufacturerData'] = dbus.Dictionary(
                self.manufacturer_data, signature='qv')
        if self.service_data is not None:
            properties['ServiceData'] = dbus.Dictionary(self.service_data,
                                                        signature='sv')
        if self.local_name is not None:
            properties['LocalName'] = dbus.String(self.local_name)
        if self.include_tx_power is not None:
            properties['IncludeTxPower'] = dbus.Boolean(self.include_tx_power)
        return {LE_ADVERTISEMENT_IFACE: properties}

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_service_uuid(self, uuid):
        if not self.service_uuids:
            self.service_uuids = []
        self.service_uuids.append(uuid)

    def add_solicit_uuid(self, uuid):
        if not self.solicit_uuids:
            self.solicit_uuids = []
        self.solicit_uuids.append(uuid)

    def add_manufacturer_data(self, manuf_code, data):
        if not self.manufacturer_data:
            self.manufacturer_data = dbus.Dictionary({}, signature='qv')
        self.manufacturer_data[manuf_code] = dbus.Array(data, signature='y')

    def add_service_data(self, uuid, data):
        if not self.service_data:
            self.service_data = dbus.Dictionary({}, signature='sv')
        self.service_data[uuid] = dbus.Array(data, signature='y')

    def add_local_name(self, name):
        if not self.local_name:
            self.local_name = ''
        self.local_name = dbus.String(name)

    @dbus.service.method(DBUS_PROP_IFACE,
                         in_signature='s',
                         out_signature='a{sv}')
    def GetAll(self, interface):
        global log
        log.logger.info('GetAll')
        if interface != LE_ADVERTISEMENT_IFACE:
            raise InvalidArgsException()
        log.logger.info('returning props')
        return self.get_properties()[LE_ADVERTISEMENT_IFACE]

    @dbus.service.method(LE_ADVERTISEMENT_IFACE,
                         in_signature='',
                         out_signature='')
    def Release(self):
        global log
        log.logger.info('%s: Released!' % self.path)


class RobotAdvertisement(Advertisement):

    def __init__(self, bus, index):
        global log
        Advertisement.__init__(self, bus, index, 'peripheral')
        self.add_service_uuid('C420')
        self.add_manufacturer_data(0xffff, [0x00, 0x01, 0x02, 0x03, 0x04])
        self.add_service_data('9999', [0x00, 0x01, 0x02, 0x03, 0x04])

        f_psn = subprocess.Popen('cat /dev/mmcblk0p11 | grep -a PSN | cut -c 6-23',
                                 shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        psn = f_psn.stdout.read()
        f_psn.stdout.close()
        if psn:
            log.logger.info('get psn is:' + str(psn))
        else:
            f_sn = subprocess.Popen('cat /sys/firmware/devicetree/base/serial-number',
                                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            psn = f_sn.stdout.read()[-5:-1]
            log.logger.info('main: get as short serial number as psn is:' + str(psn))
            f_sn.stdout.close()

        self.add_local_name(str('铁蛋') + str(psn, encoding='utf-8'))
        self.include_tx_power = True


class RobotApplication(dbus.service.Object):
    # org.bluez.GattApplication1 interface implementation

    def __init__(self, bus):
        self.path = '/'
        self.services = []
        dbus.service.Object.__init__(self, bus,
                                     self.path)
        global log
        log.logger.info('add service: ' + str(bus))
        self.add_service(RobotService(bus, 0))  # add RobotService

    def get_path(self):
        global log
        log.logger.info('get_path: ' + str(self))
        return dbus.ObjectPath(self.path)

    def add_service(self, service):
        global log
        log.logger.info('add_service: ' + str(service))
        self.services.append(service)

    @dbus.service.method(DBUS_OM_IFACE, out_signature='a{oa{sa{sv}}}')
    def GetManagedObjects(self):
        response = {}
        for service in self.services:
            response[service.get_path()] = service.get_properties()
            chrcs = service.get_characteristics()
            for chrc in chrcs:
                response[chrc.get_path()] = chrc.get_properties()
                descs = chrc.get_descriptors()
                for desc in descs:
                    response[desc.get_path()] = desc.get_properties()

        return response


class Service(dbus.service.Object):
    # org.bluez.GattService1 interface implementation

    PATH_BASE = '/org/bluez/example/service'

    def __init__(self, bus, index, uuid, primary):
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.uuid = uuid
        self.primary = primary
        self.characteristics = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            GATT_SERVICE_IFACE: {
                'UUID': self.uuid,
                'Primary': self.primary,
                'Characteristics': dbus.Array(
                    self.get_characteristic_paths(),
                    signature='o')
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_characteristic(self, characteristic):
        self.characteristics.append(characteristic)

    def get_characteristic_paths(self):
        result = []
        for chrc in self.characteristics:
            result.append(chrc.get_path())
        return result

    def get_characteristics(self):
        return self.characteristics

    @dbus.service.method(DBUS_PROP_IFACE,
                         in_signature='s',
                         out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != GATT_SERVICE_IFACE:
            raise InvalidArgsException()

        return self.get_properties()[GATT_SERVICE_IFACE]


class Characteristic(dbus.service.Object):
    # org.bluez.GattCharacteristic1 interface implementation

    def __init__(self, bus, index, uuid, flags, service):
        self.path = service.path + '/char' + str(index)
        self.bus = bus
        self.uuid = uuid
        self.service = service
        self.flags = flags
        self.descriptors = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            GATT_CHRC_IFACE: {
                'Service': self.service.get_path(),
                'UUID': self.uuid,
                'Flags': self.flags,
                'Descriptors': dbus.Array(
                    self.get_descriptor_paths(),
                    signature='o')
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_descriptor(self, descriptor):
        self.descriptors.append(descriptor)

    def get_descriptor_paths(self):
        result = []
        for desc in self.descriptors:
            result.append(desc.get_path())
        return result

    def get_descriptors(self):
        return self.descriptors

    @dbus.service.method(DBUS_PROP_IFACE,
                         in_signature='s',
                         out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != GATT_CHRC_IFACE:
            raise InvalidArgsException()

        return self.get_properties()[GATT_CHRC_IFACE]

    @dbus.service.method(GATT_CHRC_IFACE,
                         in_signature='a{sv}',
                         out_signature='ay')
    def ReadValue(self, options):
        global log
        log.logger.info('Default ReadValue called, returning error')
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE, in_signature='aya{sv}')
    def WriteValue(self, value, options):
        global log
        log.logger.info('Default WriteValue called, returning error')
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE)
    def StartNotify(self):
        global log
        log.logger.info('Default StartNotify called, returning error')
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE)
    def StopNotify(self):
        global log
        log.logger.info('Default StopNotify called, returning error')
        raise NotSupportedException()

    @dbus.service.signal(DBUS_PROP_IFACE,
                         signature='sa{sv}as')
    def PropertiesChanged(self, interface, changed, invalidated):
        pass


class Descriptor(dbus.service.Object):
    # org.bluez.GattDescriptor1 interface implementation

    def __init__(self, bus, index, uuid, flags, characteristic):
        self.path = characteristic.path + '/desc' + str(index)
        self.bus = bus
        self.uuid = uuid
        self.flags = flags
        self.chrc = characteristic
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            GATT_DESC_IFACE: {
                'Characteristic': self.chrc.get_path(),
                'UUID': self.uuid,
                'Flags': self.flags,
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_PROP_IFACE,
                         in_signature='s',
                         out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != GATT_DESC_IFACE:
            raise InvalidArgsException()

        return self.get_properties()[GATT_DESC_IFACE]

    @dbus.service.method(GATT_DESC_IFACE,
                         in_signature='a{sv}',
                         out_signature='ay')
    def ReadValue(self, options):
        global log
        log.logger.info('Default ReadValue called, returning error')
        raise NotSupportedException()

    @dbus.service.method(GATT_DESC_IFACE, in_signature='aya{sv}')
    def WriteValue(self, value, options):
        global log
        log.logger.info('Default WriteValue called, returning error')
        raise NotSupportedException()


# RobotService for ATT
class RobotService(Service):
    ROBOT_SVC_UUID = '0000C420-0000-1000-8000-00805F9B34FB'

    def __init__(self, bus, index):
        Service.__init__(self, bus, index, self.ROBOT_SVC_UUID, True)
        self.add_characteristic(writeWifiParameterFromPhone(bus, 0, self))  # for Write
        self.add_characteristic(notifyWifiStatusToPhone(bus, 1, self))  # for Notify


class writeWifiParameterFromPhone(Characteristic):
    WIFI_CHRC_UUID_WRITE = '0000C421-0000-1000-8000-00805F9B34FB'

    def __init__(self, bus, index, service):
        Characteristic.__init__(
            self, bus, index,
            self.WIFI_CHRC_UUID_WRITE,
            ['write-without-response'],
            service)

    # Write cmd for wlan from phone
    def WriteValue(self, value, options):
        global log, mainStatus, mainIP, wifiConnectFlag, mainSSID, mainPwd
        global checkThreadRun, checkThreadID, currentPhoneIP
        mainIP = ''
        # mainStatus = STATUS_WIFI_GET_INFO
        wifiConnectFlag = True
        log.logger.info('WriteValue value: ' + repr(bytes(value)))
        log.logger.info('WriteValue value utf' + repr(bytes(value).decode('utf-8')))
        format_value = bytes(value).decode('utf-8')  # ''.join(chr(byte) for byte in value)
        log.logger.info('WriteValue format value ' + format_value)
        text = json.loads(format_value)
        ssid = text.get('ssid')
        pwd = text.get('pwd')
        ns = text.get('ns')
        ip = text.get('ip')  # get phone ip address
        if ssid:
            log.logger.info('WriteValue ssid ' + repr(text['ssid']))
            mainSSID = ssid
        else:
            log.logger.warning('WriteValue ssid is none!!!')
            mainSSID = ''

        if pwd:
            log.logger.info('WriteValue pwd ' + repr(text['pwd']))
            mainPwd = pwd
        else:
            log.logger.warning('WriteValue pwd is none!!!')
            mainPwd = ''

        if ns:  # no ns, don't save
            log.logger.info('WriteValue ns ' + repr(text['ns']))
            with open('/etc/namespace.conf', 'w') as f:
                f.write(ns)
        if ip:  # if get phone ip address, publish it
            currentPhoneIP = ip

        if ssid:
            # stop check thread
            checkThreadRun = False
            checkThreadID.join()
            checkThreadID = None
            log.logger.info('check thread ended')

            # start write thread
            threadWifi = Thread(target=handleThread)  # create a thread for write value
            threadWifi.start()
            threadWifi.join()
            log.logger.debug('WriteValue: handleThread is end, and restart check thread')

            # restart check thread
            checkThreadRun = True
            checkThreadID = Thread(target=checkPhoneAndNotify)
            checkThreadID.start()
        else:
            log.logger.warning('WriteValue: handleThread is end without ssid')
        return


# notify wlan state
class notifyWifiStatusToPhone(Characteristic):
    WIFI_CHRC_UUID_NOTIFY = '0000C422-0000-1000-8000-00805F9B34FB'

    def __init__(self, bus, index, service):
        Characteristic.__init__(
            self, bus, index,
            self.WIFI_CHRC_UUID_NOTIFY,
            ['notify'],
            service)

        global notifyChar
        notifyChar = self

    # phone start notify
    def StartNotify(self):
        global log, bleStatus, ledClient, audioClient, checkThreadRun, checkThreadID

        if bleStatus != BLE_STATUS_ADV:
            log.logger.warning('StartNotify: Current bleStatus is: ' + repr(bleStatus))
        log.logger.debug('StartNotify: set bleStatus ' + repr(bleStatus) +
                         ' -> BLE_STATUS_CONNECT')
        bleStatus = BLE_STATUS_CONNECT

        RegADV(False)  # unregister adv and update led
        doNotifyOnce()  # notify once immediately when phone connected

        if ledClient.check_ready():
            if ledClient.get_last_always():
                ledClient.send_request(ledClient.LED_BT_ADV,
                                       ledClient.TYPE_FUNCTION,
                                       ledClient.COMMAND_OFF)  # clean adv led
                log.logger.debug('StartNotify: BT connected, '
                                 'clean adv LED and set LED_BT_CONNECTED')
            else:
                log.logger.debug('StartNotify: BT connected, set LED_BT_CONNECTED')
            ledClient.send_request(ledClient.LED_BT_CONNECTED,
                                   ledClient.TYPE_FUNCTION,
                                   1000000000)

        # thread for check bt connection status, and do notify when status changed
        checkThreadRun = True
        checkThreadID = Thread(target=checkPhoneAndNotify)
        checkThreadID.start()

    # phone stop notify
    def StopNotify(self):
        global log, bleStatus
        if BLE_STATUS_DISCONNECTED != bleStatus:
            bleStatus = BLE_STATUS_DISCONNECTED
            log.logger.debug('StopNotify: set bleStatus '
                             'BLE_STATUS_CONNECT -> BLE_STATUS_DISCONNECTED')
            RegATT(False)
        else:
            log.logger.warning('StopNotify: Current bleStatus is: ' + repr(bleStatus))


def doNotifyOnce():
    global log, notifyChar, mainStatus, gnotifystr, bleStatus, wifiConnection, mainSSID
    if BLE_STATUS_CONNECT != bleStatus:
        log.logger.warning('doNotifyOnce: can not notify because bleStatus is ' + repr(bleStatus))
        return
    if wifiConnection is True:
        ssid = mainSSID
    else:
        ssid = btwifitool.activeWifiSsid()

    gnotifystr = {'ssid': ssid,
                  'bssid': btwifitool.activeWifiBssid(),
                  'ip': btwifitool.activeWifiIP(),
                  'status': mainStatus}

    log.logger.info('doNotifyOnce: ' + repr(gnotifystr))
    arraySend = convert_to_dbus_array(json.dumps(gnotifystr))
    notifyChar.PropertiesChanged(GATT_CHRC_IFACE, {'Value': arraySend}, [])


# thread for write phone cmd
def handleThread():
    global log, mainStatus, mainSSID, mainPwd, wifiConnectFlag, ledClient, audioClient
    global wifiConnection, currentPhoneIP, pubPhoneIP
    log.logger.info('handleThread flag ' + repr(wifiConnectFlag) + repr(mainPwd) + repr(mainSSID))

    mainStatus = STATUS_WIFI_CONNECTING

    if ledClient.check_ready():
        log.logger.debug('BT connected and WLAN is connecting, set LED_WIFI_CONNECTING')
        ledClient.send_request(ledClient.LED_WIFI_CONNECTING,
                               ledClient.TYPE_FUNCTION,
                               5000000000)

    log.logger.info('handleThread activeWifiSsid ' + repr(btwifitool.activeWifiSsid()))

    if mainSSID == btwifitool.activeWifiSsid():  # compare current ssid and write ssid
        conpwd = btwifitool.getPwd(mainSSID)  # get current ssid pwd from file
        log.logger.info('handleThread getPwd 1 ' + repr(conpwd))
        if conpwd == mainPwd:  # same pwd means the connection is already success
            mainStatus = STATUS_WIFI_SUCCESS
            log.logger.debug('handleThread The ' + repr(mainSSID) +
                             ' is already connected, do notify')
        else:
            mainStatus = STATUS_WIFI_ERR_PWD  # pwd is not the same
            log.logger.debug('handleThread The ' + repr(mainSSID) +
                             ' input wrong passwd, do notify')
            doNotifyOnce()
            time.sleep(random.random())
            mainStatus = STATUS_WIFI_SUCCESS
            log.logger.debug('handleThread The ' + repr(mainSSID) +
                             ' is already connected, use old passwd, do notify')
        doNotifyOnce()
        if pubPhoneIP is not None:
            pubPhoneIP.publisher('%s:%s' % (currentPhoneIP, btwifitool.activeWifiIP()))
            log.logger.debug('publishA phone and dog ip %s:%s' %
                             (currentPhoneIP, btwifitool.activeWifiIP()))
            currentPhoneIP = ''
        if ledClient.check_ready():
            log.logger.debug('BT and WLAN is connected, set LED_WIFI_CONNECT_SUCCESS')
            ledClient.send_request(ledClient.LED_WIFI_CONNECT_SUCCESS,
                                   ledClient.TYPE_FUNCTION,
                                   5000000000)
        if audioClient.check_ready():
            log.logger.debug('BT and WLAN is connected, '
                             'send audio cmd TOUCH_ADV_OR_WIFI_CONNECT_SUCCESS')
            audioClient.send_goal(audioClient.TOUCH_ADV_OR_WIFI_CONNECT_SUCCESS)

        return mainStatus

    wifiConnection = True
    # disconnect wlan, or prevent wlan reconnect
    cmdoutput = btwifitool.disWifi()
    log.logger.info('disWifi :' + repr(cmdoutput))
    if not btwifitool.getDisStatus(cmdoutput):
        log.logger.debug('disconnect current wifi connect failed!!!!!')
        pass
    time.sleep(1)  # sleep for wlan scan

    if not btwifitool.isContainSSID(mainSSID):  # get a new connection request
        # a scan is required or not???
        # scan cmd: sudo iw dev wlan0 scan | grep -i "SSID: "
        log.logger.debug('handleThread do a first nmcli connect wifi command')
        cmdoutput = btwifitool.nmcliConnectWifi(mainSSID, mainPwd)  # do a new connection
        log.logger.info('handleThread first nmcliConnectWifi result' + cmdoutput)
    else:  # change pwd and reconnect
        conpwd = btwifitool.getPwd(mainSSID)  # get old pwd
        log.logger.info('handleThread getPwd 2 ' + repr(conpwd))
        if conpwd != mainPwd:
            log.logger.debug('exchange oldpwd is:' + conpwd + ' newpwd is: ' + mainPwd)
            btwifitool.exchangePSK(mainSSID, mainPwd)  # exchange pwd
            log.logger.debug('restartNetworkService.........')
            btwifitool.restartNetworkService()  # must restart network-manager
            time.sleep(3)  # sleep for restart success
            # check wlan is auto connected or not
            log.logger.debug('check wifi status again')
            if mainSSID == btwifitool.activeWifiSsid():  # if connected, return success
                mainStatus = STATUS_WIFI_SUCCESS
                doNotifyOnce()
                if pubPhoneIP is not None:
                    pubPhoneIP.publisher('%s:%s' % (currentPhoneIP, btwifitool.activeWifiIP()))
                    log.logger.debug('publishB phone and dog ip %s:%s' %
                                     (currentPhoneIP, btwifitool.activeWifiIP()))
                    currentPhoneIP = ''
                if ledClient.check_ready():
                    log.logger.debug('BT and WLAN is connected, '
                                     'set LED_WIFI_CONNECT_SUCCESS')
                    ledClient.send_request(ledClient.LED_WIFI_CONNECT_SUCCESS,
                                           ledClient.TYPE_FUNCTION,
                                           5000000000)
                if audioClient.check_ready():
                    log.logger.debug('BT and WLAN is connected, '
                                     'send audio cmd TOUCH_ADV_OR_WIFI_CONNECT_SUCCESS')
                    audioClient.send_goal(audioClient.TOUCH_ADV_OR_WIFI_CONNECT_SUCCESS)
                wifiConnection = False
                return mainStatus
            else:
                cmdoutput = btwifitool.disWifi()  # or disconnect wlan
                log.logger.info('disWifi :' + repr(cmdoutput))
                if not btwifitool.getDisStatus(cmdoutput):
                    log.logger.debug('disconnect current wifi connect failed!!!!!')
                time.sleep(1)  # for system ready
                log.logger.debug('handleThread command connectUpWifi ' + repr(mainSSID))
                cmdoutput = btwifitool.connectUpWifi(mainSSID)  # reconnect
                log.logger.info('handleThread connectUpWifi return: ' + repr(cmdoutput))
            # Test 3 End
        else:  # same pwd, reconnect
            log.logger.debug('handleThread command connectUpWifi ' + repr(mainSSID))
            cmdoutput = btwifitool.connectUpWifi(mainSSID)  # wlan reconnect
            log.logger.info('handleThread reconnect wifi ' + repr(cmdoutput))
    time.sleep(1)  # for system ready
    tmpStatus = btwifitool.return_connect_status(cmdoutput)  # check connection status
    log.logger.info('handleThread return_connect_status is: ' + repr(tmpStatus))
    if tmpStatus == STATUS_WIFI_INTERRUPT or \
            tmpStatus == STATUS_WIFI_NO_SSID or \
            tmpStatus == STATUS_WIFI_OTHER:
        if ledClient.check_ready():
            log.logger.warning('wlan connect failed, set LED_WIFI_CONNECT_FAILED')
            ledClient.send_request(ledClient.LED_WIFI_CONNECT_FAILED,
                                   ledClient.TYPE_ALARM,
                                   5000000000)

        log.logger.warning('wlan connect failed, restart network-manager')
        btwifitool.restartNetworkService()  # reset network-manager when error
        time.sleep(3)  # for network
        # do connect wlan again
        if tmpStatus == STATUS_WIFI_INTERRUPT:
            resultretry = btwifitool.connectUpWifi(mainSSID)  # reconnect
        else:
            resultretry = btwifitool.nmcliConnectWifi(mainSSID, mainPwd)  # do connect
        tmpStatus = btwifitool.return_connect_status(resultretry)  # check status
        log.logger.info('handleThread retry restartNetworkService ' + repr(resultretry))
    # log.logger.info('handleThread return_connect_status tmp ' + repr(tmpStatus))
    # if tmpStatus != STATUS_WIFI_TIMEOUT:
    #     mainStatus = tmpStatus
    mainStatus = tmpStatus
    log.logger.info('handleThread return_connect_status main ' + repr(mainStatus))
    doNotifyOnce()
    if STATUS_WIFI_SUCCESS == mainStatus:
        log.logger.debug('handleThread wifi connected')
        if pubPhoneIP is not None:
            pubPhoneIP.publisher('%s:%s' % (currentPhoneIP, btwifitool.activeWifiIP()))
            log.logger.debug('publishC phone and dog ip %s:%s' %
                             (currentPhoneIP, btwifitool.activeWifiIP()))
            currentPhoneIP = ''
        if ledClient.check_ready():
            ledClient.send_request(ledClient.LED_WIFI_CONNECT_SUCCESS,
                                   ledClient.TYPE_FUNCTION,
                                   5000000000)
        if audioClient.check_ready():
            audioClient.send_goal(audioClient.TOUCH_ADV_OR_WIFI_CONNECT_SUCCESS)
    else:
        log.logger.debug('handleThread wifi connect failed=%d' % mainStatus)
        if ledClient.check_ready():
            ledClient.send_request(ledClient.LED_WIFI_CONNECT_FAILED,
                                   ledClient.TYPE_ALARM,
                                   5000000000)
    wifiConnection = False
    return mainStatus


# Converts strings to hexadecimal
def convert_to_dbus_array(string):
    value = []
    for c in string:
        value.append(dbus.Byte(c.encode()))
    return value


def get_managed_objects():
    bus = dbus.SystemBus()
    manager = dbus.Interface(bus.get_object('org.bluez', '/'),
                             'org.freedesktop.DBus.ObjectManager')
    return manager.GetManagedObjects()


def find_adapter_in_objects(objects, pattern=None):
    bus = dbus.SystemBus()
    for path, ifaces in objects.items():
        adapter = ifaces.get(ADAPTER_INTERFACE)
        if adapter is None:
            continue
        if not pattern or pattern == adapter['Address'] or \
                path.endswith(pattern):
            obj = bus.get_object(BLUEZ_SERVICE_NAME, path)
            return dbus.Interface(obj, ADAPTER_INTERFACE)
    raise Exception('Bluetooth adapter not found')


def register_app_cb():
    global log
    log.logger.info('register_app_cb: GATT application registered!!!')


def register_app_error_cb(error):
    global log, mainloop
    log.logger.info('Failed to register application: ' + str(error))
    mainloop.quit()


def register_ad_cb():
    global log
    log.logger.info('register_ad_cb: Advertisement registered!!!')


def register_ad_error_cb(error):
    global log, mainloop, bleStatus
    log.logger.info('Failed to register advertisement: ' + str(error))
    bleStatus = BLE_STATUS_DISCONNECTED
    log.logger.warning('register_ad_error_cb: set bleStatus to BLE_STATUS_DISCONNECTED!!!!!')
    # mainloop.quit()


def find_adapter(bus):
    remote_om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, '/'),
                               DBUS_OM_IFACE)
    objects = remote_om.GetManagedObjects()

    for o, props in objects.items():
        if GATT_MANAGER_IFACE in props.keys():
            if LE_ADVERTISING_MANAGER_IFACE in props:
                return o
    return None


def checkPhoneAndNotify():
    global log, mainStatus, gnotifystr, bleStatus, ledClient, audioClient, wifiConnection, mainSSID
    global checkThreadRun
    log.logger.debug('checkPhoneAndNotify: Thread start')
    while BLE_STATUS_CONNECT == bleStatus and checkThreadRun is True:
        # 1 check phone bt connection status
        if btwifitool.bleConnectStatus() is not True:
            # Switch to the disconnected state when disconnected in the connected state
            log.logger.debug('checkPhoneAndNotify: phone is disconnect,set bleStatus '
                             'BLE_STATUS_CONNECT -> BLE_STATUS_DISCONNECTED')
            if BLE_STATUS_CONNECT == bleStatus:
                bleStatus = BLE_STATUS_DISCONNECTED
                RegATT(False)
            break

        # 2 connection status, check wlan and ros2 status
        currentIP = btwifitool.activeWifiIP()
        # 2.1 wlan is not connect
        if currentIP == '':  # when init or connecting, do nothing
            if mainStatus != STATUS_NONE and mainStatus != STATUS_WIFI_CONNECTING:
                mainStatus = STATUS_NONE
                log.logger.debug('checkPhoneAndNotify: current wlan is not connect, '
                                 'set mainStatus to STATUS_NONE')
                doNotifyOnce()
        # 2.2 wlan is connected
        else:
            # 2.2.1 wlan in other wlan status, not success
            if mainStatus <= STATUS_WIFI_OTHER or mainStatus >= STATUS_WIFI_INTERRUPT:
                mainStatus = STATUS_WIFI_SUCCESS
                log.logger.info('checkPhoneAndNotify: set mainStatus to STATUS_WIFI_SUCCESS')
                doNotifyOnce()
                time.sleep(1)
            # 2.2.2 wlan success or ros2 status
            if STATUS_WIFI_SUCCESS <= mainStatus <= STATUS_ROS2_OTHER:
                tmpStatus = btwifitool.getRos2Status()
                if mainStatus != tmpStatus:
                    mainStatus = tmpStatus
                    log.logger.info('checkPhoneAndNotify: ros2 state changed, '
                                    'set mainStatus to ' + repr(mainStatus))
                    doNotifyOnce()

        # 3 do notify when there is any change
        if wifiConnection is True:
            ssid = mainSSID
        else:
            ssid = btwifitool.activeWifiSsid()
        tmpnotifystr = {'ssid': ssid,
                        'bssid': btwifitool.activeWifiBssid(),
                        'ip': btwifitool.activeWifiIP(),
                        'status': mainStatus}
        if gnotifystr.get('ssid') != tmpnotifystr.get('ssid') or \
                gnotifystr.get('bssid') != tmpnotifystr.get('bssid') or \
                gnotifystr.get('ip') != tmpnotifystr.get('ip') or \
                gnotifystr.get('status') != tmpnotifystr.get('status'):
            log.logger.debug('checkPhoneAndNotify： old gnotifystr is ' + repr(gnotifystr))
            gnotifystr = tmpnotifystr
            log.logger.debug('checkPhoneAndNotify： new gnotifystr is ' + repr(gnotifystr))
            log.logger.info('checkPhoneAndNotify: update new notify')
            doNotifyOnce()
        # else:
        #     log.logger.debug('checkPhoneAndNotify: the same dict, do nothing and wait 2s')

        for i in range(4):
            if checkThreadRun is not True:
                break
            time.sleep(0.5)

    log.logger.debug('checkPhoneAndNotify: Thread end')


class TouchSubscriber(Node):
    LPWG_TOUCHANDHOLD_DETECTED = 7

    def __init__(self):
        super().__init__('touch_subscriber', namespace=rclcomm.get_namespace())
        self.subscription = self.create_subscription(
            Touch, 'TouchState', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        global log, bleStatus, audioClient, ledClient
        log.logger.info('listener_callback: I heard: "%d"' % msg.touch_state)

        if enableHoldTouch is True:
            if msg.touch_state != self.LPWG_TOUCHANDHOLD_DETECTED:  # only detect hold mode touch
                return

        if enableStopAutoDisconnectRC is True:
            if remote.remoteStatus is not remote.REMOTE_STATUS_DISCONNECTED:
                log.logger.warning('current remoteStatus is %d, can not start ADV, '
                                   'please disconnect remote first' % remote.remoteStatus)
                return

        # stop auto connect and disconnect remote handle
        if enableAutoRemoteControl is True:
            remote.stopAutoConnect()
            remote.checkAndAutoDisconnect()

        log.logger.debug('current bleStatus : ' + repr(bleStatus))
        if BLE_STATUS_NONE == bleStatus:
            bleStatus = BLE_STATUS_ADV
            RegADV(True)
            RegATT(True)
            if ledClient.check_ready():
                log.logger.debug('RegADV: BT enable adv, set LED_BT_ADV')
                ledClient.send_request(ledClient.LED_BT_ADV, ledClient.TYPE_FUNCTION, 5000000000)
        elif BLE_STATUS_ADV == bleStatus:
            if enableReTouchADV is True:
                RegADV(False)
                time.sleep(1)
                RegADV(True)
                if ledClient.check_ready():
                    log.logger.debug('RegADV: BT enable adv, set LED_BT_ADV')
                    ledClient.send_request(ledClient.LED_BT_ADV,
                                           ledClient.TYPE_FUNCTION,
                                           5000000000)
        elif BLE_STATUS_CONNECT == bleStatus:
            if btwifitool.bleConnectStatus() is not True:  # when disconnected, regADV again
                log.logger.warning('The phone is disconnected, waiting for update bleStatus')
            else:
                log.logger.debug('Phone is connected, waiting for disconnect...')
        elif BLE_STATUS_DISCONNECTED == bleStatus:
            bleStatus = BLE_STATUS_ADV
            RegADV(True)
            RegATT(True)
            if ledClient.check_ready():
                log.logger.debug('RegADV: BT enable adv, set LED_BT_ADV')
                ledClient.send_request(ledClient.LED_BT_ADV,
                                       ledClient.TYPE_FUNCTION,
                                       5000000000)
        else:
            log.logger.error('unknown bleStatus ' + repr(bleStatus))

        if BLE_STATUS_ADV == bleStatus:
            if audioClient.check_ready():
                log.logger.debug('send audio cmd TOUCH_ADV_OR_WIFI_CONNECT_SUCCESS from touch')
                audioClient.send_goal(audioClient.TOUCH_ADV_OR_WIFI_CONNECT_SUCCESS)


def RegADV(state):
    global ad_manager, service_manager, robotAdv
    if state is True:
        log.logger.info('RegADV: RegisterAdvertisement')
        ad_manager.RegisterAdvertisement(robotAdv.get_path(), {},
                                         reply_handler=register_ad_cb,
                                         error_handler=register_ad_error_cb)
    else:
        log.logger.info('RegADV: UnregisterAdvertisement')
        ad_manager.UnregisterAdvertisement(robotAdv.get_path())


def RegATT(state):
    global log, service_manager, robotApp, ledClient
    if state is True:  # enable adv and gatt
        log.logger.info('RegATT: RegisterApplication')
        service_manager.RegisterApplication(robotApp.get_path(), {},
                                            reply_handler=register_app_cb,
                                            error_handler=register_app_error_cb)
    else:  # release att when bt disconnect
        log.logger.info('RegATT: UnregisterApplication')
        service_manager.UnregisterApplication(robotApp.get_path())
        # start auto connect
        if enableAutoRemoteControl is True:
            log.logger.info('RegATT: call startAutoConnect')
            remote.startAutoConnect()


def touchWaiting():
    touch_subscriber = TouchSubscriber()
    rclpy.spin(touch_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    touch_subscriber.destroy_node()
    rclpy.shutdown()


def main(args=None):
    global log, mainStatus, pubPhoneIP, bleStatus, ledClient, audioClient
    log = record.Logger('/var/log/bluetooth.log', level='debug')
    rclcomm.getlog()
    btwifitool.getlog()
    remote.getlog()

    bleStatus = BLE_STATUS_NONE
    mainStatus = STATUS_NONE
    rclpy.init(args=args)

    log.logger.info('main: init')
    ledClient = rclcomm.LedClientAsync()
    audioClient = rclcomm.AudioPlayActionClient()
    remote.RemotePublisherEvent()
    remote.RemotePublisherKey()
    remote.parameterPublisher()
    touch_subscriber = TouchSubscriber()
    remote_Service = remote.RemoteCommandService()
    remote.gaitActionClient()
    remote.ChangeModeActionClient()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(touch_subscriber)
    executor.add_node(remote_Service)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    pubPhoneIP = rclcomm.PublisherPhoneIP()  # for publish phone ip

    global ad_manager, service_manager, mainloop, robotAdv, robotApp

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    adapter = find_adapter(bus)
    if not adapter:
        log.logger.error('GattManager1 interface not found')
        return
    service_manager = dbus.Interface(
        bus.get_object(BLUEZ_SERVICE_NAME, adapter),
        GATT_MANAGER_IFACE)
    robotApp = RobotApplication(bus)
    adapter_props = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter),
                                   'org.freedesktop.DBus.Properties')
    adapter_props.Set('org.bluez.Adapter1', 'Powered', dbus.Boolean(1))
    ad_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter),
                                LE_ADVERTISING_MANAGER_IFACE)
    robotAdv = RobotAdvertisement(bus, 0)
    mainloop = GObject.MainLoop()

    if enableInitAdv is True:
        bleStatus = BLE_STATUS_ADV
        log.logger.debug('main: set bleStatus BLE_STATUS_NONE -> BLE_STATUS_ADV')
        RegADV(True)
        RegATT(True)
        if audioClient.check_ready():
            log.logger.debug('main init and send audio cmd BOOT_COMPLETE')
            audioClient.send_goal(audioClient.BOOT_COMPLETE)
        if ledClient.check_ready():
            log.logger.debug('RegADV: BT enable adv, set LED_BT_ADV')
            ledClient.send_request(ledClient.LED_BT_ADV,
                                   ledClient.TYPE_FUNCTION,
                                   ledClient.COMMAND_ALWAYSON)

    mainloop.run()


if __name__ == '__main__':
    main()
