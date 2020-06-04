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

## Installation Instructions

Please note that you need to have a python3 installation for Penn-BET, but [conda](https://www.anaconda.com/) is preferred.

```bash
git clone https://github.com/CBICA/Penn-BET.git
cd Penn-BET
conda create -n pbet python=3.6 -y # create a virtual environment
conda activate pbet # activate it
latesttag=$(git describe --tags) # get the latest tag [bash-only]
echo checking out ${latesttag}
git checkout ${latesttag}
python setup.py install # install dependencies
pip install -e . # install Penn-BET with a reference to scripts
pip install requirements.txt
```

## Preprocessing Data

Use the following command for preprocessing, which will process all the modalities for a given subject together and write it in the specified output location:

```bash
./env/python PENN_BET/utils/preprocess.py -i ${inputSubjectDirectory} -o ${outputSubjectDirectory} -t threads
```
**Note**: ```${inputSubjectDirectory}``` needs to be in the same format as described in [Arranging Data](##Arranging-Data). 

## Running Instructions

We have two modes in here : `train` and `test`.

### Training

- Populate a config file with required parameters (please see [train_params.cfg](./Penn_BET/config/train_params.cfg) for an example)
- Note that preprocessed data in the specific format [ref](##Arranging-Data) should be used.
- Invoke the following command:

```bash
penn_bet_run -params train_params.cfg -train True -dev $device -load $resume.ckpt
```

Note that ```-load $resume.ckpt``` is only needed if you are resuming your training. 

### Inference

- We have three modes here:
  - Modality Agnostic (MA)
  - Multi-4, i.e., all 4 modalities getting used
  - Single (**not yet** yet and its weights would be updated soon) 
- Populate a config file with required parameters. Examples:
  - MA: [test_params_ma.cfg](./Penn_BET/config/test_params_ma.cfg)
  - Multi-4: [test_params.cfg](./Penn_BET/config/test_params_multi_4.cfg)
- It is highly suggested that Multi-4 should be only run with some certain preprocessing steps (link goes here) mentioned below.
- Invoke the following command:

```bash
penn_bet_run -params $test_params.cfg -test True -dev $device -mode MA
```
```bash
penn_bet_run -params $test_params.cfg -test True -dev $device -mode Multi-4
```

```$device``` refers to the GPU device where you want your code to run or the CPU.

## Notes

- Please note that the if you wish to use your own weights, you can use the ```-load``` option, but we suggest you to use our weights that are provided in the weights folder.
- Using this software is pretty trivial as long as instructions are followed. 
- You can use it in any terminal on a supported system. 
- The ```penn_bet_run``` command gets installed automatically. 
- We provide CPU (untested as of 2020/05/31) as well as GPU support. 
  - Running on GPU is a lot faster though and should always be preferred. 
  - You need an GPU memory of ~5-6GB for testing and ~8GB for training.

## TO-DO

- Document information about ```csv_provided``` flag in configuration
- In inference, rename ```model_dir``` to ```results_dir``` for clarity in the configuration and script(s)
- Add CCA for post-processing
- Add version in the ```-h``` command using https://stackoverflow.com/a/2073599/1228757
- Add link to CaPTk as suggested mechanism for preprocessing (can refer to ```BraTSPipeline``` application)
- Test on CPU
- Put option to write logs to specific files in output directory
- Remove ```-mode``` parameter in ```penn_bet_run```
- Please post any requests as issues on this repository or send email to software@cbica.upenn.edu

## Contact

Please email software@cbica.upenn.edu with questions.
