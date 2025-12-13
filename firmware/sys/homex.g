; homex.g — Home X to MAX (global.X_LENGTH)
; Strategy: temporarily disable B driver so it freewheels,
; then home X with a normal H1 move.

M400
M564 H0 S0
G91                                ; relative moves
M17

; 1) Disable B driver so only X is actively driven
M84 P2                             ; P2 = driver 2 (your B motor)

; 2) Make sure we're off the switch a bit
G1 X-5 H2 F2000                    ; move away from max, ignore endstops

; 3) Fast seek toward X MAX
; *** If this goes the wrong way, flip the sign on X200 (and X-5 below). ***
G1 X200 H1 F2000                   ; move toward X max, stop on X endstop

; 5) Back off and slow re-touch for accuracy
G1 X-5 H2 F2000                     ; release switch
G1 X10 H1 F300                     ; gentle bump back into switch

G90
G92 X{global.X_LENGTH - 77.5}             ; X = 115.5 at the switch
M118 P0 S"X homed to 115.5"

; 6) Re-enable B driver
M17 P2                             ; turn B motor back on

M564 H1 S1
