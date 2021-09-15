# Feedback status
uint8 DEFAULT  = 0
uint8 ACCEPT   = 1
uint8 REJECT   = 2
uint8 PLAYING  = 3
uint8 ERROR    = 4

# value
std_msgs/Header header
float64 progress_rate
AudioSongName name
uint8 status