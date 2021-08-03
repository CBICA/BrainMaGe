#!/bin/bash
net='brainmage'
log_path="inference_logs/"
bs="/"
hyphen="-"

for precision in FP32 INT8
do
   for i in 4 
      do
         python benchmark-ov-corescaling.py --device 'CPU' --nc $i --data_type $precision --data_path ../nfbs-dataset-test-preprocessed.csv >> $log_path$net$hyphen$precision$hyphen"num_cores"$hyphen$i$hyphen"numstreams-cpudefault.txt"
      done
done
