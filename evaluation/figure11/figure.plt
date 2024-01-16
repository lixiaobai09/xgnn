#!/usr/bin/env gnuplot

reset
file="scalability.dat"
set output "scalability.eps"
set terminal postscript "Helvetica,20" eps enhance color dl 2 size 5, 1.8

set key spacing 1.5 samplen 1.5
set xlabel "Number of GPU"
set ylabel "Epoch Time (sec)"

set linetype 1 lc rgb "#bf0000"
set linetype 2 lc rgb "#ff9900"

set multiplot layout 1,2


set lmargin at screen 0.125
set rmargin at screen 0.45
set ytics 0,2
set yrange [0:9]

plot \
     file index "pa" using 1:3 title "DGL+C" w lp lw 3 pt 3 ps 2 lc 1, \
     file index "pa" using 1:4 title "XGNN" w lp lw 3 pt 4 ps 2 lc 2, \
     # file using 1:2 title "DGL" w lp lw 3 pt 2 ps 2 , \
     # file using 1:5 title "DGL-CPU-Ext" w lp lw 3 pt 4 ps 2, \

set lmargin at screen 0.65
set rmargin at screen 0.975
set ytics 0,1
set yrange [0:4.5]

plot \
     file index "uk" using 1:3 title "DGL+C" w lp lw 3 pt 3 ps 2 lc 1, \
     file index "uk" using 1:4 title "XGNN" w lp lw 3 pt 4 ps 2 lc 2, \

unset multiplot

