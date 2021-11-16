# Converting to OpenVINO IR and Benchmarking

## Prereqs for running

### Clone repo BrainMaGe, checkout openvino branch and Install BrainMaGe

```bash
git clone https://github.com/ravi9/BrainMaGe.git
cd BrainMaGe
git checkout openvino
git lfs pull
conda env create -f requirements.yml # create a virtual environment named brainmage
conda activate brainmage # activate it

python setup.py install #install BrainMaGe

```

### Install Dependencies

```bash
pip install torch==1.9.1 # Update PyTorch if needed.
pip install medpy
```

### Install OpenVINO

```bash
pip install --ignore-installed PyYAML openvino-dev
```

### Install OpenVINO Contrib

Setup `openvino_contrib` which has PyTorch extensions for Model Optimizer which can be used to convert native PyTorch to OpenVINO IR.

```bash
cd ~
git clone https://github.com/openvinotoolkit/openvino_contrib.git
export PYTHONPATH=`pwd`/openvino_contrib/modules/mo_pytorch/:$PYTHONPATH
```

## After above steps

Following steps are to download the dataset, generate the manifest for the dataset, export PyTorch to OpenVINO, quantize the OV model, benchmark

```bash
cd BrainMaGe/openvino
bash ./download-NFBS_Dataset.sh # Download NFBS sample dataset
python generate-csv-nfbs-dataset.py # generate dataset MANIFEST CSV

python export-to-ov.py # Convert PyTorch model to OpenVINO IR
python quantize.py # Quantize the OpenVINO IR model
python benchmark-pt-ov.py # Run a benchmark. Note the accuracy might fail on NFBS dataset.
```

### Benchmarking with OpenVINO Benchmark App

```bash
# Assuming you are in BrainMaGe/openvino folder

# Benchark FP32 model
benchmark_app -m ../BrainMaGe/weights/ov/fp32/resunet_ma.xml -nireq 1 -nstreams 1 -niter 20

# Benchark INT8 model
benchmark_app -m ./int8_openvino_model/resunet_ma_int8.xml -nireq 1 -nstreams 1 -niter 20

# Run with -pc to print performance counters for profiling
benchmark_app -m ./int8_openvino_model/resunet_ma_int8.xml -nireq 1 -nstreams 1 -niter 20 -pc 2>&1 | tee bench-app-ov-int8-niter-10.log

```

### IPython Notebook

Follow the [compare-pyt-ov-single-infer.ipynb](compare-pyt-ov-single-infer.ipynb) for step-by-step instructions.

### Core Scaling Benchmarking

```bash
# Assuming you are in BrainMaGe/openvino folder
# Update the script as per your needs.
bash ./core_scale_infer.sh

```

### Memory Profiling

```bash
conda install memory_profiler
python -m memory_profiler benchmark-pt-ov-memprofile.py 2>&1 | tee bench-mem-prfl.log
```
