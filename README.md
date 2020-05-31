# Penn-BET 

## Arranging Data

The data needs to be preprocessed before fed to the network.

The structure of the data needs to be as follows:
If the data is structure like this, there is no need to provide the csv to the ```dataloader```.
Although this is something that can be improved to make the ```dataloader``` more dynamic to support federated learning better, this is up for development.
```
Data_folder -- patient_1 -- patient_1_t1.nii.gz
                         -- patient_1_t2.nii.gz
                         -- patient_1_t1ce.nii.gz
                         -- patient_1_flair.nii.gz
                         -- patient_1_mask.nii.gz
               patient_2 -- ...
               ...
               ...
               patient_n -- ...
```

## Preprocessing Data

Use the following command for preprocessing, which will process all the modalities for a given subject together and write it in the specified output location:

```
./env/python PENN_BET/utils/preprocess.py -i ${inputSubjectDirectory} -o ${outputSubjectDirectory} -t threads
```
## Installation Instructions
Please note that you need to have a python3 installation for Penn-BET, but conda is preferred.
How to create conda environements

```bash
conda create -n pbet python=3.6 -y
conda activate pbet
git clone https://github.com/Geeks-Sid/Penn-BET
python setup.py install
pip install -e .
pip install requirements.txt
```

## How to Run

We have two modes in here : `train` and `test`
## Training

- Populate a config file with required parameters (please see [train_params.cfg](./Penn_BET/config/train_params.cfg) for an example)
- Note that preprocessed data should be used.
- Invoke the following command:
```
penn_bet_run -params train_params.cfg -train True -dev $device -load $resume.ckpt
```
## Inference

- We have three modes here. MA, Multi-4, Single(Not supported yet)(weights for it would be updated soon) 
- Populate a config file with required parameters (please see [test_params.cfg](./Penn_BET/config/test_params_multi_4.cfg) for an example of Multi-4 and [test_params_ma.cfg](./Penn_BET/config/test_params_ma.cfg) for an example of MA mode)
- It is highly suggested that Multi-4 should be only run with some certain preprocesing steps(link goes here) mentioned below.
- Invoke the following command:
```
penn_bet_run -params $test_params.cfg -test True -dev $device -mode MA
```
```
penn_bet_run -params $test_params.cfg -test True -dev $device -mode Multi-4
```
-Please note that the if you wish to use your own weights, you can use the `-load` option, but we suggest you to use our weights that are provided in the weights folder.
-Using this software is pretty trivial as long as instructions are followed. You can use it in any terminal on your linux system. The hd-bet command was installed automatically. We provide CPU as well as GPU support. Running on GPU is a lot faster though and should always be preferred. And we have not currently run tests on CPU. We might in the next version.
-You need an approxiamate GPU memory of ~5-6GB for testing and atleast ~8GB for training.

## TO-DO
-Add CCA for postprocessing
-Test on CPU
