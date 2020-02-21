set autoscale
set logscale y
set logscale x
set key autotitle columnhead
set xtic auto
set ytic auto
set datafile separator ","
plot for [n=2:*] "data/spectral.dat" using 1:n with lines

