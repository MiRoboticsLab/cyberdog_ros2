# bcm_gps

该模块主要提供GPS的驱动封装

- 外部参数通过`cyberdog_ception_bridge`模块中`params/bcmgps_config.toml`载入
- GPS模块自身日志保存在`cyberdog_ception_bridge`模块中`logs`路径下

### 主要接口部分如下：

```cpp
namespace bcm_gps
{
typedef LD2BRM_PvtPvtPolledPayload GPS_Payload;
using NMEA_callback = std::function<void (uint8_t * str, uint32_t len)>;
using PAYLOAD_callback = std::function<void (std::shared_ptr<GPS_Payload> payload)>;

class GPS
{
public:
  explicit GPS(PAYLOAD_callback PAYLOAD_cb = nullptr, NMEA_callback NMEA_cb = nullptr);
  ~GPS();
  bool Open();
  void Start();
  void Stop();
  void Close();
  bool Ready();
  void SetCallback(NMEA_callback NMEA_cb);
  void SetCallback(PAYLOAD_callback PAYLOAD_cb);

  void SetL5Bias(uint32_t biasCm);
  void SetLteFilterEn(bool enable = true);
};  // class GPS
}  // namespace bcm_gps
```

#### public函数：

构造函数：初始化GPS对象

```cpp
explicit GPS(PAYLOAD_callback PAYLOAD_cb = nullptr, NMEA_callback NMEA_cb = nullptr);
```

- `PAYLOAD_cb`：payload形式回调函数
- `NMEA_cb`：NMEA标准格式字符串回调函数



开启设备：返回是否开启成功（构造函数会自动开启设备，主要用于手动调用Close()后重新开启）

```cpp
bool Open();
```



打开定位

```cpp
void Start();
```



关闭定位

```cpp
void Stop();
```



关闭设备（关闭后会清空注册的回调函数，重新打开需要Open()后重新注册回调函数获取数据）

```cpp
void Close();
```



获取硬件设备准备状态：返回准备是否成功

```cpp
bool Ready();
```



注册回调函数

```cpp
void SetCallback(NMEA_callback NMEA_cb);
void SetCallback(PAYLOAD_callback PAYLOAD_cb);
```

- `PAYLOAD_cb`：payload形式回调函数
- `NMEA_cb`：NMEA标准格式字符串回调函数



Input L5 bias in centimeter

```cpp
void SetL5Bias(uint32_t biasCm);
```

- biasCm：L5 bias in centimeter



Enable/Disable LTE Filter

```cpp
void SetLteFilterEn(bool enable = true);
```

- enable：Enable/Disable



#### GPS_Payload消息结构

```cpp
/** @brief BRM-NAV-PVT (0x01 0x07)
*
*/
PACK(
  typedef struct
{
  U4 iTOW;
  U2 year;
  U1 month;
  U1 day;
  U1 hour;
  U1 min;
  U1 sec;
  X1 valid;
  U4 tAcc;
  I4 nano;
  U1 fixType;
  X1 flags;
  X1 flags2;
  U1 numSV;
  I4 lon;
  I4 lat;
  I4 height;
  I4 hMSL;
  U4 hAcc;
  U4 vAcc;
  I4 velN;
  I4 velE;
  I4 velD;
  I4 gSpeed;
  I4 headMot;
  U4 sAcc;
  U4 headAcc;
  U2 pDOP;
  I1 leapS;
  U1 reserved1[5];
  I4 headVeh;
  I2 magDec;
  U2 magAcc;
}) LD2BRM_PvtPvtPolledPayload;
```



### 外部参数

外部参数通过`cyberdog_ception_bridge`模块中`params/bcmgps_config.toml`载入

```toml
skip_download = false  # need in first line
onlyfirst_download = true
tty = "/dev/ttyTHS0"
patch_path = "/usr/sbin/bream.patch"

#uint8_t infMsgMask[6] = {0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F};
#full logging (GLLIO, GLLAPI, RAWDATA, DEVKF, USR2, )
infMsgMask = [31, 31, 31, 31, 31, 31] 

#1(L1 Best) 2(L1 Auto) 3(L1 ULP), 4(L1L5 Best) 5(L1L5 Auto)
PowerModePreset = 4

MsgRate = [
  1,  # Report GPGGA rate
  1,  # Report GPRMC rate
  1,  # Report GPGSV rate
  1,  # Report NAVEOE rate
  1,  # Report NAVPVT rate
  1,  # Report PGLOR SPEED rate
  1,  # Report PGLOR FIX rate
  1,  # Report PGLOR SAT rate
  1,  # Report PGLOR LSQ rate
  1,  # Report PGLOR PWR rate
  1,  # Report PGLOR STA rate
  1,  # Report NAVODO rate
  1,  # Report NAVSAT rate
  1,  # Report NAVDOP rate
  1,  # Report CBEE status rate
  1,  # Report ASC SUBFRAMES rate
  1,  # Report ASC MEAS rate
  1   # Report ASC AGC rate
]

AckAiding = true
```

- `skip_download`：是否跳过下载patch（请置于首行）
- `onlyfirst_download`：仅编译后首次下载（该选项为`true`将会在下载后将`skip_download`置为`true`）
- `tty`：设备号
- `patch_path`：下载patch路径
- `infMsgMask`：GPS模块日志配置
- `PowerModePreset`：GPS频段选择
  - 1：(L1 Best)
  - 2：(L1 Auto) 
  - 3：(L1 ULP)
  - 4：(L1L5 Best) (默认)
  - 5：(L1L5 Auto)
- `MsgRate`：消息频率设置（0为关闭）
- `AckAiding`：true/false



### Example

```cpp
class Example
{
public:
    std::shared_ptr<bcm_gps::GPS> gps_;
  	std::shared_ptr<bcm_gps::GPS_Payload> payload_;
    
    Example()
    {
        gps_ = std::make_shared<bcm_gps::GPS>(std::bind(
      		&Example::payload_callback, this, std::placeholders::_1));
    }

    void Func()
    {
		gps_->Start();
        gps_->Stop();
        gps_->Close();
    }
    void payload_callback(std::shared_ptr<bcm_gps::GPS_Payload> payload)
    {
        payload_ = payload;
    }
};  // class example
```

