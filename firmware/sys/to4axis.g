
M350 C16 X16 Z16 B16 E16 I1                                             ; configure microstepping with interpolation
M92 C88.8888 X50.00 Z200.00 B50.00 E467.5                            ; set steps per mm
M566 C300.00 X300.00 Z300.00 B300.00 E.300                            ; set maximum instantaneous speed changes (mm/min)
M203 C3600.00 X5000.00 Z3000.00 B5000.00 E3600.00                       ; set maximum speeds (mm/min)
M201 C1000.00 X2000.00 Z300.00 B800.00 E500.00                          ; set accelerations (mm/s^2)
M906 C1300 X1300 Z1300 B1300 E800 I30                                   ; set motor currents (mA) and motor idle factor in per cent

; Axis Limits
M208 C-20000000 X0 Z-6.5 B-90.5 S1                                             ; set axis minima. 
M208 C20000000 X{global.X_LENGTH} Z155 B{100} S0        ; set axis maxima. Some compensation for b axis sensorless homing

; Define Kinematics
M669 K0 C0:0:0:0:1 X-1:0:0:1:0 Z0:0:1:0:0 B0.22222222:0:0:0.222222222:0 ; 4 axis controlM307 H0 R0.822 K0.611:0.000 D10.94 E1.35 S1.00 B0 V24.0