set terminal postscript eps color solid "Helvetica" 24 size 8,8
set output "statsig_highlow.eps"
set xlabel "Classifier"
set ylabel "Recall Score"
set title "Statistical Significance High/Low Chart"
set datafile separator ','
set key autotitle columnhead
set xtics rotate
set yrange [0.5:1.0]
set xrange [-1:15]
set grid xtics ytics
show grid
plot "statsig.csv" using 0:4:2:3:xticlabels(1) notitle linewidth 5 with yerrorbars
