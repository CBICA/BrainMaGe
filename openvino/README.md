# Converting to OpenVINO IR and Benchmarking

## Prereqs for running

### Clone this repo of BrainMaGe and Install BrainMaGe

```bash
git clone https://github.com/ravi9/BrainMaGe.git BrainMaGe-ravi9
cd BrainMaGe-ravi9
git lfs pull
conda env create -f requirements.yml # create a virtual environment named brainmage
conda activate brainmage # activate it

python setup.py install
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
cd BrainMaGe-ravi9/openvino
bash ./download-NFBS_Dataset.sh
python generate-csv-nfbs-dataset.py # Edit Path to where the NFBS dataset is downloaded.
python export-to-ov.py
python quantize.py
python benchmark-pt-ov.py
```

### Benchmarking with OpenVINO Benchmark App

```bash
bash ./run-benchmark-app.sh
```

### Memory Profiling

```bash
conda install memory_profiler
python -m memory_profiler benchmark-pt-ov-memprofile.py 2>&1 | tee bench-mem-prfl.log
```
