# Result of Voiceprint execution status 
#
std_msgs/Header header
#

# error code
int32 NORMAL                 =    0
int32 UNKNOWN_ERROR          =    1
int32 RECORD_FAILED          =    2
int32 PLAY_FAILED            =    4
int32 TOO_NOISY_BACKGROUND   =    8
int32 TIMEOUT                =   16


int8 type
bool succeed
int32 error 