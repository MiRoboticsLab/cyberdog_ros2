# error code
uint32 NORMAL                 =    0
uint32 UNKNOWN_ERROR          =    1
uint32 RECORD_FAILED          =    2
uint32 PLAY_FAILED            =    4
uint32 TIMEOUT                =    8
uint32 REJECT                 =    16
uint32 SERVER_INACTIVE        =    32
# value
std_msgs/Header header
AudioSongName name
bool succeed
uint32 error