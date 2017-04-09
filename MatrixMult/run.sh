#!/bin/bash
for run in {1..3..1}
do
	echo 'New run'
	echo $run
	echo 'New run' >> parallel.log
	echo $run >> parallel.log
	for i in {600..9800..400}
	do
		mpirun -host grid10,grid11,grid12,grid13,grid14,grid15,grid16,grid17,grid18,grid19 ./MParallel $i >> parallel.log
	done
done
