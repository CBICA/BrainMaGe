#!/bin/bash
# Clear cache
echo "echo 3 > /proc/sys/vm/drop_caches"

benchmark_app -m ./int8_openvino_model/resunet_ma_int8.xml -nireq 1 -nstreams 1 -niter 10 -pc 2>&1 | tee bench-app-ov-int8-niter-10.log

# Clear cache
echo "echo 3 > /proc/sys/vm/drop_caches"

benchmark_app -m ../BrainMaGe/weights/ov/fp32/resunet_ma.xml -nireq 1 -nstreams 1 -niter 10 -pc 2>&1 | tee bench-app-ov-fp32-niter-10.log