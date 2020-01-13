# set terminal pdf
# set output "out.pdf"
set autoscale
set logscale y
set logscale x
set xtic auto
set ytic auto
plot "out.dat" using 1:2

