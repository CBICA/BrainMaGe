## Script to run benchmark_app experiments for BrainMaGe

### This script assumes the models have been converted to OpenVINO IR
<br> The FP32 IR files are located under: brainmage_root / 'BrainMaGe/weights/ov/fp32'
<br> The INT8 IR files are located under: brainmage_root / 'openvino/int8_openvino_model'

### To run the script on a DevCloud node
Login to DevCloud and open a new terminal
<br> Login to the node on which you'd like to run the benchmarking using qsub: `qsub -I -l nodes=1:<node ID>`
<br> cd into the benchmarks script directory and run the benchmark_nstream.sh script (Refer to the above run script instructions to make the necessary modifications first)
