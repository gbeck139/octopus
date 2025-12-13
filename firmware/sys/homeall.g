; homeall.g

M669 K0 C0:0:0:0:1 X1:0:0:0:0 Z0:0:1:0:0 B0:0:0:1:0
M400
M564 H0 S0
M17

G91
G1 Z40 H2 F1000        ; lift for safety
G90

M98 P"homex.g"

M98 P"homeb.g"

M98 P"homec.g"
M118 P0 S"c homed to 0"

M98 P"homey.g"
M118 P0 S"y homed to 0"

M98 P"homez.g"
M118 P0 S"All Homed"

M98 P"to4axis.g"

M564 H1 S1
G92 X0 B0 C0 Y0 Z10

G1 B95 F1000

G92 B0                 ; apply your logical B offset
M118 P0 S"B homed to 0"