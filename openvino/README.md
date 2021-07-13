# Converting to OpenVINO IR and Benchmarking

## Prereqs for running:

```bash
git clone -b mo_resunet https://github.com/dkurt/openvino_contrib/
source /opt/intel/openvino_2021/bin/setupvars.sh
export PYTHONPATH=/path/to/openvino_contrib/modules/mo_pytorch/:$PYTHONPATH
```


## After above steps:

```bash
python generate-csv-nfbs-dataset.py 
python export-to-ov.py 
python quantize.py
python benchmark-pt-ov.py
```

