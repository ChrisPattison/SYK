set autoscale
set logscale y
unset logscale x
set key autotitle columnhead
set xtic auto
set ytic auto
set datafile separator ","
plot for [n=2:*] "data/variance.dat" using 1:2

