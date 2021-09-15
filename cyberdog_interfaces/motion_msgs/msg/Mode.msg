# Mode with timestamp
#
builtin_interfaces/Time timestamp
#
# Control Mode 
#
uint8 MODE_DEFAULT    = 0
uint8 MODE_LOCK       = 1
uint8 MODE_MANUAL     = 3
uint8 MODE_SEMI       = 13
uint8 MODE_EXPLOR     = 14
uint8 MODE_TRACK      = 15
#
# Mode Type
#
uint8 DEFAULT_TYPE = 0
uint8 TRACK_F      = 1
uint8 TRACK_S      = 2
uint8 EXPLR_NAV_AB = 3
uint8 EXPLR_MAP_U  = 4
uint8 EXPLR_MAP_N  = 5
#
uint8 control_mode
uint8 mode_type