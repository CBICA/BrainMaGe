# Converting to OpenVINO IR and Benchmarking

## Prereqs for running:

- ### Clone this repo of BrainMaGe and Install BrainMaGe

```bash
git clone https://github.com/ravi9/BrainMaGe.git BrainMaGe-ravi9
cd BrainMaGe-ravi9
git lfs pull
conda env create -f requirements.yml # create a virtual environment named brainmage
conda activate brainmage # activate it

python setup.py install
```

- ### Install OpenVINO

```bash
# Install this specific version of OpenCV to prevent libGl errors
pip uninstall -y opencv-python
pip install -U opencv-python-headless==4.2.0.32 --user
pip install --ignore-installed PyYAML openvino-dev
```

- ### Install OpenVINO Contrib

Install this custom contrib to convert PyTorch models to OpenVINO IR without converting to ONNX

```bash
cd ~
git clone -b mo_resunet https://github.com/dkurt/openvino_contrib/
export PYTHONPATH=/path/to/openvino_contrib/modules/mo_pytorch/:$PYTHONPATH
```

## After above steps:

```bash
cd BrainMaGe-ravi9/openvino
bash ./download-NFBS_Dataset.sh
python generate-csv-nfbs-dataset.py # Edit Path to where the NFBS dataset is downloaded.
python export-to-ov.py
python quantize.py
python benchmark-pt-ov.py
```

- ### Benchmarking with OpenVINO Benchmark App

```bash
bash ./run-benchmark-app.sh
```

- ### Memory Profiling

```bash
conda install memory_profiler
python -m memory_profiler benchmark-pt-ov-memprofile.py 2>&1 | tee bench-mem-prfl.log
```
