get_nvlrx() {
    for ncu_log in "$@" 
    do  
        cat $ncu_log | grep nvlrx__bytes.sum | awk \
            '{ if ($2 == "Mbyte") sum += $3; else sum += $3 / 1024.0 } END { print sum / NR}'
    done
}
