# audio_assistant

---
## **模块简介**
该模块主要用于实现语音助手的相关功能，这部分的代码集成了语音助手相关SDK的相关功能，主要包括唤醒功能，语音识别（ASR）功能，自然语言处理（NLP）功能。
整个流程可以理解为由以下步骤构成：
1、在设备端取7路录音数据，送给唤醒模块，其中7路数据的前6路是6路麦克风录入的数据，第7路是Smart PA的参考通道的信号，该信号的数据与喇叭发声的数据相同，取这路信号是为了支持语音助手算法的回声消除功能。
2、唤醒模块对输入的7路数据做处理，对唤醒词“铁蛋铁蛋”进行识别，唤醒成功，进入下一步。
3、唤醒成功后，跟在唤醒词之后的数据会被送到ASR进行处理，语音输入的数据再经过ASR后会被处理为文本输出，之后送入NLP模块。
4、NLP模块对输入的文本进行处理，经过TTS输出合成的语音，或者具体的指令（语音动作指令）。

由于一些原因，目前语音助手算法部分的功能以动态库的形式给出，audio_assistant这个模块只展示了对这些算法的一个集成部分，因此下次只本模块的功能做一个简述。

## **功能简介**
audio assistant模块提供了语音助手在线功能和语音助手离线功能的一些方法，目前的逻辑为的当APP上语音识别的开关开启后，本模块在接收到audio intraction发出的开启语音助手功能的请求后，首先通过audio interaction发出的wifi状态的topic来判断当前网络状态，当网络状态良好的情况下优先运行语音助手在线功能，否则运行语音助手离线功能。

语音助手离线功能与在线功能的区别是，离线功能目前仅支持几个有限的语音动作指令，不支持在线交谈等功能，而在线功能不仅支持语音动作指令，还支持一定程度的语音交互功能，涉及：天气，闲聊，智能设备控制（如：开灯）等。

为了方便与audio interaction之间的通信，本模块的功能在一个单独的ROS节点中分几个线程来运行，包括录音线程、唤醒线程、ASR线程、NLP线程、TTS合成线程，以及一个看门狗线程。
```
  /*raw data recorde task*/
  threadMainCapture = std::make_shared<std::thread>(&AudioAssistant::AiMainCaptureTask, this);
  threadMainCapture->detach();
  /*wakeup task*/
  threadAiWakeup = std::make_shared<std::thread>(&AudioAssistant::AiWakeupTask, this);
  threadAiWakeup->detach();
  /*native asr task*/
  threadNativeAsr = std::make_shared<std::thread>(&AudioAssistant::AiNativeAsrTask, this);
  threadNativeAsr->detach();
  /*setup online sdk*/
  threadAiOnline = std::make_shared<std::thread>(&AudioAssistant::AiOnlineTask, this);
  threadAiOnline->detach();
  /*setup tts handler task*/
  threadTts = std::make_shared<std::thread>(&AudioAssistant::AiTtsTask, this);
  threadTts->detach();
  /*setup audio assistant wdog*/
  threadWdog = std::make_shared<std::thread>(&AudioAssistant::AiWdogTask, this);
  threadWdog->detach();
```

录音线程需要从设备端采集6+1路录音数据，具体的实现方法可参见**combiner.cpp**的实现。

唤醒线程主要用于完成唤醒词的识别，需要提出的是，铁蛋这一代的产品不支持以外部中断的方式触发的硬件唤醒功能，这部分的功能完全通过软件算法来实现。唤醒功能的集成部分可参见**ai_keyword.cpp**的实现。

本地ASR线程支持语音助手离线功能，用于在唤醒后完成语音识别以及文本处理的功能，此部分集成代码可参见**ai_asr_native.cpp**以及**ai_nlp_native.cpp**的实现。目前语音助手离线功能支持7个语音指令（包括唤醒），如下：
*铁蛋铁蛋*
*站起来*
*趴下去*
*握个手*
*跳个舞*
*原地转圈*
*后退一步*

在线线程主要用于支持语音助手在线功能，祥见**AivsMain.cpp**。

TTS线程用于处理tts合成，tts转码（mp3转wav），以及tts播放的一系列功能，详见**AivsMain.cpp**，**mp3decoder.cpp**，tts的播放以来audio base中提供的方法实现。

