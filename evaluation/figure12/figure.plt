#!/usr/bin/env gnuplot

reset
file="memory-usage.dat"
set output "memory-usage.eps"
set terminal postscript "Helvetica,24" eps enhance color dl 2 size 5.8, 2.6

set key spacing 1.5 samplen 1.5 maxrows 2 font ",26" center at graph 0.3, graph 0.84
set key width 4
# set xlabel "Number of GPUs"
set y2label "Memory Usage (GB)" offset -2.5, 0
set ylabel "Mini-Batch Time (ms)" offset 2, 0
set y2range [0:128]
set yrange [0:0.04*1000]
set ytics nomirror
set ytics 0,0.01*1000 offset 0.5,0
set y2tics 0,30 offset -1,0

set xtic offset -0.6,0.2 nomirror rotate by -40 left font ", 24"

width=0.20
gap=width/2+0.05
set boxwidth width
set style fill noborder

set xrange [0-3*width:3+width*3]


set tmargin 1.4
set bmargin 3.25
set lmargin 5.5
set rmargin 6.5

# set linetype 1 lc rgb "#F07F7F"
# set linetype 2 lc rgb "#BF0000"
# set linetype 3 lc rgb "#B1B1FB"
# set linetype 4 lc rgb "#0000ED"

set linetype 1 lc rgb "#ffd699"
set linetype 2 lc rgb "#ff9900"
set linetype 3 lc rgb "#b0cda7"
set linetype 4 lc rgb "#3b8424"

set style fill 


gpu_num="1 2 4 8"
do for [i=0:3] {
    h1=-0.0105*1000
    h2=-0.008*1000
    set label center at i, h1 sprintf("%sG", word(gpu_num, i+1)) font ", 24"
    set arrow from i-gap-width/2-0.08, h2 to i+gap+width/2+0.08, h2 nohead
}

plot \
    file using ($0-gap):(1000*($4+$5+$6)/$8):xtic(sprintf("time")) ax x1y1 title "Trainer" with boxes fs solid 1 lc 1, \
    file using ($0-gap):(1000*$4/$8):xtic(sprintf("time")) ax x1y1 title "Sampler" with boxes fs solid 1 lc 2, \
    file using ($0+gap):(($2+$3)/1024.0/1024.0/1024.0):xtic(sprintf("mem")) ax x1y2 title "Feature" with boxes fs solid 1 lc 3, \
    file using ($0+gap):($3/1024.0/1024.0/1024.0):xtic(sprintf("mem")) ax x1y2 title "Graph" with boxes fs solid 1 lc 4, \
