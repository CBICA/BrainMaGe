## Source Tree Description

The details of the files are as follows

```The structure of the files is as follows :
Penn-BET -- model -- model.py           (The networks are defined in here)
                  -- seg_modules.py     (The layers of the network are defined here)
         -- utils -- csv_creator_adv.py (Function that automatically creates CSV for training)
                  -- cyclicLR.py        (The cosine learning rate scheduler)
                  -- data.py            (The dataloader for the particular problem)
                  -- losses.py          (Some of the loss functions collected over the years)
                  -- optimizers.py      (Fetching optimizers)
                  -- preprocess_data.py (Some helper functions for preocessing)
                  -- preprocess.py      (Preprocessing the data files)
         -- train.py                    (The function for training)
         -- train_params.cfg            (Config file for training)
```

## Python Environment

An example of creating the python environment is given utilizing [Anaconda](https://www.anaconda.com/) and can be replicated for your python package manager:

```bash
conda create -p ./env python=3.6.5 -y
conda activate ./env
conda install pytorch -y
# special command for nibabel
conda install -c conda-forge nibabel -y
# TODO: populate other libraries here
```
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

Use the following command for preprocessing:

```
./env/python utils/preprocess.py -i ${inputSubjectDirectory} -o ${outputSubjectDirectory} -t threads
```



## Training

- Populate a config file with required parameters (please see [train_params.cfg](./train_params.cfg) for an example)
- Note that preprocessed data should be used.
- Invoke the following command:

```bash
./env/python train.py ./train_params.cfg # or your own config file name
```
