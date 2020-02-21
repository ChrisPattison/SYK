set autoscale
unset logscale y
unset logscale x
set xtic auto
set ytic auto
set yrange [0:]
set datafile separator ","
plot  "data/variance.dat" using 1:2

