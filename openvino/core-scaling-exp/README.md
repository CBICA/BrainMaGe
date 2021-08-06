## Script to run core scaling experiments for BrainMaGe

### This script assumes the models have been converted to OpenVINO IR
To use the core scaling experiments core_scale_infer.sh script, specify the path to the dataset csv file in the script (for ex. if the data path is /home/user/BrainMaGe/openvino/dataset.csv, specify the data path as --data_path /home/user/BrainMaGe/openvino/dataset.csv in the core_scale_infer.sh script). <br>
If the input and mask indexes are different, specify those, for ex: --input_path_idx 1, --mask_path_idx 2 etc. <br>
<br> Modify the log_path to point to an existing directory where the benchmarks logs should be stored. 
<br> To change the benchmarking configurations: <br>
Modify the precision (line 7) for which you'd like to run the benchmarking for. (for ex. FP32, INT8) <br>
Modify the num cores configuration (line 16) for which you'd like to run the benchmarking for. (A range can be specified for ex: 1 2 4 to run tests for number of cores 1, 2 and 4). <br>

### To run the script on a DevCloud node
Login to DevCloud and open a new terminal
<br> Login to the node on which you'd like to run the benchmarking using qsub: qsub -I -l nodes=1:<node ID>
 <br> cd into the benchmarks script directory and run the core_scale_infer.sh script (Refer to the above run script instructions to make the necessary modifications)
