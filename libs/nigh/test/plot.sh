#!/usr/bin/env bash

file="$1"
title="$2"
shift 2
(
    # set scale to 2 or hidpi screens on term qt
    echo "scale=1"
    #echo "set term qt size 1024*scale,768*scale persist raise"

    echo "set term svg size 640,480 dynamic"
    echo "set output '$file'"
    
    echo "set title '$title'"
    echo "set ylabel 'us/search'"
    echo "set xlabel 'data structure size'"
    echo "set logscale x"
    echo "set key top left"
    echo "set format x '10^{%T}'"

    echo "ymax = 0"
    for file in $@ ; do
        title=$(sed -n '/^# Strategy = / { s/.*= // ; p ; }' "$file")
        if [[ $title != Linear ]] ; then
            echo "stats '$file' u 1:4 nooutput"
            echo "ymax = STATS_max_y > ymax ? STATS_max_y : ymax"
        fi
    done
    echo "set yrange [0:ymax]"
    
    sep="plot "
    for file in $@ ; do
        title=$(sed -n '/^# Strategy = / { s/.*= // ; p ; }' "$file")
        echo -n "$sep '$file' using 1:4 w lp lw scale title '$title'"
        sep=", "
    done
    echo ""
    
) | /usr/bin/env gnuplot
