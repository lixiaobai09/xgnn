outputfname = "factor-analysis.eps"
dat_file='factor_analysis.res'

set datafile sep '\t'

set terminal postscript "Helvetica,16" eps color dl 2
set style data histogram

set style histogram clustered gap 1.2
set style fill solid border -2
set pointsize 1
set size 0.55,0.35
set boxwidth 0.7 relative
# set no zeroaxis

set tics font ",14" scale 0.5

set rmargin 1
set lmargin 4
set tmargin 0.5
set bmargin 2.4

set output outputfname

### Key
set key inside right Right  top enhanced nobox autotitles columnhead
set key samplen 1.5 spacing 1.5 height 0.2 width 0.5 font ',12'  at graph 0.25, graph 0.95 noopaque

set xrange [-.5:22.5]
set xtics nomirror rotate by -60 left offset -0.2,0.3 scale 0 font ",10"

h=-0.25
g=h-0.07

set arrow from 0-0.4,graph h to  4.7,graph h nohead lt 2 lw 1 lc "#000000" front
set arrow from 6-0.4,graph h to  10.7,graph h nohead lt 2 lw 1 lc "#000000" front
set arrow from 12-0.4,graph h to  16.7,graph h nohead lt 2 lw 1 lc "#000000" front
set arrow from 18-0.4,graph h to  22.7,graph h nohead lt 2 lw 1 lc "#000000" front

set label "GCN + TW"          center at 2.5, graph g font ",10" tc rgb "#000000" front
set label "GCN + PA"          center at 8.5, graph g font ",10" tc rgb "#000000" front
set label "GSG + UK"          center at 14.5, graph g font ",10" tc rgb "#000000" front
set label "GSG + CF"          center at 20.5, graph g font ",10" tc rgb "#000000" front
set label "OOM"               left   at 18,   graph 0.02   rotate by 90   font ",10" tc rgb "#000000" front

## Y-axis
set ylabel "Time (sec)" offset 1.5,0 font ",14"
set yrange [0:1.6]
set ytics ("0" 0, "0.5" 0.5, "1.0" 1.0, "1.5" 1.5)
set ytics offset 0.5,0

plot dat_file      using 4:xticlabels(3)       lc "#c00000" title "Sample" \
    ,dat_file      using 5:xticlabels(3)       lc "#ff9900" title "Train" \
