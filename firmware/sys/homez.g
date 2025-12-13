; homez.g — Home Z to MIN using Z endstop on E0
; Recenter X using both X and B before Z homing.

M400
M564 H0 S0
M17

; --- Move X to center with B disabled ---
M84 P2                     ; disable B motor (driver 2)
G90
G1 X0 F2000                ; move X to center of bed (X=0)
                            ; flip to X{something else} if your true center is different
M17 P2

; --- Home Z down to the endstop on E0 ---
G91
G1 Z-200 H1 F 500              ; move toward bed until Z endstop triggers
G1 Z3 H2 F500                  ; back off
G1 Z-5 H1 F250                 ; slow second touch

G90
G92 Z-6                         ; Z = 0 at contact
M118 P0 S"Z homed to -6"

G1 Z15 F500                    ; lift to safe height

M564 H1 S1
