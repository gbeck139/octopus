; Configuration file for Fly-E3-Pro-v3 (firmware version 3.5)
; executed by the firmware on start-up

global B_OFFSET = 0
global X_LENGTH = 115                                    ; actually 117.0 but this stops some squshing issues
; global X_LENGTH = 117

; General preferences
G90                                                        ; send absolute coordinates...
M83                                                        ; ...but relative extruder moves
M550 P"Ceramic Printer"                                    ; set printer name

; Network
M552 S1                                                    ; enable network
M586 P0 S1                                                 ; enable HTTP
M586 P1 S0                                                 ; disable FTP
M586 P2 S0                                                 ; disable Telnet

; Drives (update M569 for your drivers; adjust S0/S1 for direction)
M569 P1 S1 D3 V4000                                        ; Driver 1 (X motor, one CoreXY-theta motor)
M569 P2 S1 D3 V4000                                        ; Driver 2 (B motor, other CoreXY-theta motor; assuming E1 is driver 3)
M569 P4 S1                                                 ; Driver 4 (Z motor; assuming Z0 is driver 1)
M569 P0 S1                                                 ; Driver 0 (C motor; assuming Y is driver 0)
M569 P3 S1                                                 ; Driver 3 (extruder E0; assuming standard)
M84 S30                                                    ; Set idle timeout

M350 C16 X16 Z16 B16 E16 I1                                             ; configure microstepping with interpolation
M92 C88.8888 X50.00 Z200.00 B50.00 E467.5                              ; set steps per mm
M566 C300.00 X300.00 Z300.00 B300.00 E300.00                            ; set maximum instantaneous speed changes (mm/min)
M203 C3600.00 X5000.00 Z3000.00 B5000.00 E300.00                       ; set maximum speeds (mm/min)
M201 C1000.00 X2000.00 Z300.00 B800.00 E500.00                          ; set accelerations (mm/s^2)
M906 C1300 X1300 Z1300 B1300 E800 I30                                   ; set motor currents (mA) and motor idle factor in per cent

; Axis Limits
M208 C-20000000 X0 Z-6.5 B-90.5 S1                                             ; set axis minima. 
M208 C20000000 X{global.X_LENGTH} Z155 B{100} S0        ; set axis maxima. Some compensation for b axis sensorless homing

; Axis Mapping  (must match physical wiring above)
M584 X1 B2 Z4 C0 E3        ; X→P1, B→P2, Z→P4, C→P0, E→P3

; Endstops
M574 X2 S1 P"^!ystop"                                         ; X HIGH (max) end, physical switch
M574 B1 S1 P"^!zstop"                                         ; B low end, physical switch on E1
M574 Z1 S1 P"^!e0stop"                           	     ; Z MIN, switch wired to E0 endstop


; Heaters
M140 H-1                                                   ; disable heated bed (overrides default heater mapping)
M308 S0 P"e0temp" Y"thermistor" T100000 B4956 C1.587780e-7 ; configure sensor 0 as thermistor on pin e0temp
M950 H0 C"e0heat" T0                                       ; create nozzle heater output on e0heat and map it to sensor 0
M307 H0 R2.478 K0.750:0.000 D9.76 E1.35 S1.00 B0 V24.0    ; pwm config from autotune
M143 H0 S270                                               ; set temperature limit for heater 0 to 120C
M302 P1                                                    ; allow cold extrudes

; Fans
M950 F0 C"fan0" Q500                                            ; create fan 0 on pin fan0 and set its frequency
M106 P0 S0 H-1                                               ; set fan 0 value. 

; Tools
M563 P0 D0 F0 H0                                           ; define tool 0
G10 P0 X0 Y0 Z0                                            ; set tool 0 axis offsets
G10 P0 R0 S0                                               ; set initial tool 0 active and standby temperatures to 0C
T0                                                         ; select tool
