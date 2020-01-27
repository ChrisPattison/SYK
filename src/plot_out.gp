# set terminal pdf
# set output "out.pdf"
set autoscale
set logscale y
set logscale x
set xtic auto
set ytic auto
plot for [n=2:*] "out.dat" using 1:n

