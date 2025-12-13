; homeb.g — Home B to MIN using B endstop, with X moving too

M400
M564 H0 S0
G91                                      ; relative moves
M17

; --- Move X to center with B disabled ---
M84 P1                     ; disable X motor (driver 1)

; 1) Fast seek toward B MIN
G1 B-200 H1 F1000               ; B watched, X helper

; 2) Back off
G1 B5 H2 F1000

; 3) Slow re-touch
G1 B-10 H1 F300
G90
G92 B0                 ; apply your logical B offset
M118 P0 S"B homed to 0"

; 4) Re-enable X driver
M17 P1                             ; turn X motor back on

M564 H1 S1
