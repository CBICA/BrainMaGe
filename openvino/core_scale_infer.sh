#!/bin/bash
net='brainmage'
log_path=`pwd`"/corescale-logs-$(date +%Y-%m-%d_%H-%M-%S)/"
bs="/"
hyphen="-"

mkdir $log_path

for precision in PTFP32 FP32 INT8
   do
      for i in 1 2 4 8
         do
            python benchmark-ov-corescaling.py --device 'CPU' --nc $i --data_type $precision 2>&1 | tee $log_path$net$hyphen$precision$hyphen"num_cores"$hyphen$i$hyphen"CPU.log"
         done
   done
