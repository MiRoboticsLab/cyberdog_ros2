# audio_base

## audio_player
该模块用于播放多段或单段不定长PCM音频数据、wav格式音频文件，其底层实现为SDL2与SDL_mixer

官方帮助文档：[SDL2](https://wiki.libsdl.org/CategoryAPI)，[SDL_mixer](https://www.libsdl.org/projects/SDL_mixer/docs/SDL_mixer_frame.html)

### 包含文件：

```include/audio_base/audio_player.hpp```

```src/audio_player.cpp```

### 主要接口部分如下：

```cpp
namespace cyberdog_audio
{
#define DELAY_CHECK_TIME 1000
#define DEFAULT_PLAY_CHANNEL_NUM 4
#define DEFAULT_VOLUME MIX_MAX_VOLUME

#define AUDIO_FREQUENCY 16000
#define AUDIO_FORMAT MIX_DEFAULT_FORMAT
#define AUDIO_CHANNELS 1
#define AUDIO_CHUCKSIZE 2048

using callback = std::function<void (void)>;

class AudioPlayer
{
public:
  explicit AudioPlayer(
    int channel,
    callback finish_callback = nullptr,
    int volume_group = INDE_VOLUME_GROUP,
    int volume = DEFAULT_VOLUME);
  static int GetFreeChannel();
  static bool InitSuccess();

  static bool OpenReference();
  static void CloseReference();
  static int GetReferenceData(Uint8 * buff, int need_size);
  static size_t GetReferenceDataSize();

  void SetFinishCallBack(callback finish_callback);
  int SetVolume(int volume);
  void SetVolumeGroup(int volume_gp);
  int GetVolume();

  void AddPlay(Uint8 * buff, int len);
  void AddPlay(const char * file);
  void StopPlay();
  bool IsPlaying();
  bool InitReady();
};
}

```

### 其中宏定义：

- `DELAY_CHECK_TIME`：播放线程每`DELAY_CHECK_TIME`(ms)检查一次是否正在播放音频，是否需要退出
- `DEFAULT_PLAY_CHANNEL_NUM`：混音通道默认数量（通道数会根据根据需要动态增加数量）
- `DEFAULT_VOLUME`：默认音量（0-128）
- `AUDIO_FREQUENCY`、`AUDIO_FORMAT`、`AUDIO_CHANNELS`、`AUDIO_CHUCKSIZE`：为SDL_mixer初始化参数（[官方文档](https://www.libsdl.org/projects/SDL_mixer/docs/SDL_mixer_frame.html)）

### public函数：
构造函数：
初始化播放器对象

```cpp
AudioPlayer(
  int channel,
  callback finish_callback = nullptr,
  int volume_group = INDE_VOLUME_GROUP,
  int volume = DEFAULT_VOLUME);
```
- `channel`：使用的混音通道
- `finish_callback`：播放队列完成后的回调函数
- `volume_group`：音量通道组，改变该组中单个通道音量时，将改变所有同组通道音量
- `volume`：该通道音量

获取空闲通道：返回空闲通道channel号
```cpp
static int GetFreeChannel();
```

获取SDL初始化状态：返回SDL是否初始化成功
```cpp
static bool InitSuccess();
```

打开音频播放反馈的参考通道并开始录制数据：返回是否成功打开
```cpp
static bool OpenReference();
```

关闭音频播放反馈的参考通道并清除录制数据
```cpp
static void CloseReference();
```

获取录制的参考通道数据：返回实际获取数据长度
```cpp
static int GetReferenceData(Uint8 * buff, int need_size);
```
- `buff`：数据头指针
- `need_size`：需求数据长度

获取当前参考通道数据长度：返回当前数据长度
```cpp
static size_t GetReferenceDataSize();
```

重新设置播放队列完成后的回调函数：
```cpp
void SetFinishCallBack(callback finish_callback);
```

设置该混音通道音量(0-128)：返回实际设置音量，-1表示初始化存在错误
```cpp
int SetVolume(int volume);
```

设置混音通道音量分组：同组音量将同时改变
```cpp
void SetVolumeGroup(int volume_gp);
```

获取当前通道音量：返回当前通道音量(0-128)，-1表示初始化存在错误
```cpp
int GetVolume();
```

添加播放队列：
```cpp
void AddPlay(Uint8 * buff, int len);
```
- `buff`：PCM数据头指针
- `len`：数据长度
- 该函数可装载动态长度数据，也可和`void AddPlay(const char * file)`函数混用
```cpp
void AddPlay(const char * file);
```
- `file`：wav音频文件路径
- 该函数可和`void AddPlay(Uint8 * buff, int len)`函数混用

停止并清空播放队列：
```cpp
void StopPlay();
```

播放检查：返回是否正在播放音频
```cpp
bool IsPlaying();
```

初始化检查：返回实例化对象是否初始化成功
```cpp
bool InitReady();
```

### 示例：
```cpp
#include "audio_player.cpp"

using namespace cyberdog_audio;

using player_ptr = std::shared_ptr<AudioPlayer>;

class AudioPlayerDemo
{
private:
  player_ptr player_;
public:
  AudioPlayerDemo();
  ~AudioPlayerDemo();
  void Init();
  void FinishCallback();
  void UsePlay();
};

AudioPlayerDemo::AudioPlayerDemo() {
  Init();
}

AudioPlayerDemo::~AudioPlayerDemo() {
  player_ = nullptr;
}

void AudioPlayerDemo::Init() {
  player_ = std::make_shared<AudioPlayer>(1, std::bind(&AudioPlayerDemo::FinishCallback, this));
}

void AudioPlayerDemo::FinishCallback() {
  printf("FinishCallback\n");
}

void AudioPlayerDemo::UsePlay() {
  if (player_->InitReady()) {
    player_->AddPlay("/audio.wav");
  } else {
    printf("Init Failed\n");
  }
}
```
