# outputfname = "cache_hit.eps"
dat_file='cache_hit.dat'

# col numbers
col_dataset=1
col_cache_policy=2
col_cache_ratio=3
col_hit_rate=4

# split location
split_location=50
end_location=350

set fit logfile '/dev/null'
set fit quiet
set datafile sep '\t'

set terminal postscript "Helvetica,16" eps enhance color dl 2
set pointsize 1
set size 0.8,0.8
set zeroaxis

set tics font ",14" scale 0.5

# set rmargin 2
# set lmargin 5.5
# set tmargin 1.5
# set bmargin 2.5

# set output outputfname

#### magic to filter expected data entry from dat file
format_str="<awk -F'\\t' '{ if($".col_dataset." == \"%s\"     && ". \
                              "$".col_cache_policy." == \"%s\"      ) { print }}' ".dat_file." "
cmd_filter_dat_by_policy(dataset, policy)=sprintf(format_str, dataset, policy)
##########################################################################################

# set size 0.27, 0.4
# set origin 0,0
# set rmargin 0.5
# set lmargin 5.5
# set tmargin 1.5
# set bmargin 1
### Key
set key inside left Left reverse bottom invert enhanced nobox 
set key samplen 1 spacing 1.5 height 0.2 font ',13' at graph 0.7, graph 0.8 opaque

### X-axis
set xrange [0:100]
set xtics 0, 20, 100
set xlabel "Cache Ratio(%)\n\n".ARG1 # offset 2.8,0.5
set xtics nomirror offset -0.2,0.3

## Y-axis
set yrange [0:100]
set ytics 0,20,100
set ylabel "Hit Rate(%)" # offset 1.,0
set ytics offset 0.5,0 #format "%.1f"

plot cmd_filter_dat_by_policy(ARG1, "degree")    using (column(col_cache_ratio)*100):(column(col_hit_rate)*100)  w l lw 1  lc "#c00000" title "Degree" \
    ,cmd_filter_dat_by_policy(ARG1, "presample") using (column(col_cache_ratio)*100):(column(col_hit_rate)*100)  w l lw 1  lc "#ff9900" title "Pre-sample" \

