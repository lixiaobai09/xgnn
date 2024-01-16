outputfname = "placement-solver-tw-fig-a.eps"
dat_file='placement_solver_tw.res'

set datafile sep '\t'

set terminal postscript "Helvetica,16" eps color dl 2
set style data histogram

set style histogram clustered gap 1.5
set style fill solid border -2
set pointsize 1
set size 0.32,0.345
set boxwidth 0.7 relative
# set no zeroaxis

set tics font ",14" scale 0.5

set rmargin 1
set lmargin 6
set tmargin 1.1
set bmargin 1.5

set output outputfname

### Key
set key inside right Right  top enhanced nobox autotitles columnhead
set key samplen 1.5 spacing 1.5 height 0.2 width 0.5 font ',12'  at graph 1, graph 0.98 noopaque

set xrange [-.5:3.5]
set xtics nomirror offset -0.9,0 scale 0 font ",14" left

t=0.00
h=-0.21-t
g=h-0.08-t

# set arrow from 0-0.4,graph h to  2.4,graph h nohead lt 2 lw 1 lc "#000000" front
# set arrow from 4-0.4,graph h to  6.4,graph h nohead lt 2 lw 1 lc "#000000" front
# 
# set label "CF"          center at 1.0, graph g font ",12" tc rgb "#000000" front
# set label "TW"          center at 5.0, graph g font ",12" tc rgb "#000000" front


## Y-axis
set ylabel "Time (sec)" offset 2.8,0 font ",14"
set yrange [0:0.21]
set ytics 0.1
# set yrange [0:0.3]
# set ytics 0.1
set ytics offset 0.5,0 #format "%.1f" #nomirror

## 4Gc label
# set label "2.52" right rotate by -40  at 1-0.2, graph 1.03 font ",8" tc rgb "#000000" front
# set label "2.27" right rotate by -40  at 1+0.3, graph 1.03 font ",8" tc rgb "#000000" front
# set arrow heads

# set arrow from 1-0.7, graph 0.91 to 1-0.1, graph 0.985 lt 2 lw 1 lc "#000000" front head size 0.5,15,15
# set arrow from 1+0.8, graph 0.91 to 1+0.35, graph 0.985 lt 2 lw 1 lc "#000000" front head size 0.5,15,15



plot dat_file      using 4:xticlabels(3)       lc "#3a8524" title "Clique" \
    ,dat_file      using 6:xticlabels(3)       lc "#c00000" title "CliqueOpt" \
    ,dat_file      using 5:xticlabels(3)       lc "#ff9900" title "Solver" \
