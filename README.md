# Deep-BET 

## Arranging and Processing Data

The data needs to be preprocessed before fed to the network.

### Brain Preprocessing steps

The following steps need to be followed for preprocessing brain data:

- DICOM to NIfTI conversion
- Re-orientation to LPS/RAI
- N4 Bias correction
- Co-registration to T1CE modality
- Registration to [SRI-24 atlas](https://www.nitrc.org/projects/sri24/) in the LPS/RAI space
- Apply registration to re-oriented image to maximize image fidelity
- https://github.com/CBICA/Deep-BET#standardizing-dataset-intensities

Users can use the ```BraTSPipeline``` executable from the [Cancer Imaging Phenomics Toolkit (CaPTk)](https://github.com/CBICA/CaPTk/) to make this process easier. This pipeline currently uses a pre-trained model to extract the skull but the processed images (in the order defined above till registration) are also saved.

### Expected Directory structure for data

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

This can be circumvented by using a data CSV via a [data file](##Data-File-usage), using the ```csv_provided``` parameter.

## Installation Instructions

Please note that you need to have a python3 installation for Deep-BET, but [conda](https://www.anaconda.com/) is preferred.

```bash
git clone https://github.com/CBICA/Deep-BET.git
cd Deep-BET
conda env create -f requirements.yml # create a virtual environment named deepbet
conda activate deepbet # activate it
latesttag=$(git describe --tags) # get the latest tag [bash-only]
echo checking out ${latesttag}
git checkout ${latesttag}
python setup.py install # install dependencies
pip install -e . # install Deep-BET with a reference to scripts
```

## Standardizing Dataset Intensities

Use the following command for preprocessing, which will standardize the intensities of all the modalities for a given subject and write it in the specified output location:

```bash
./env/python Deep_BET/utils/intensity_standardize.py -i ${inputSubjectDirectory} -o ${outputSubjectDirectory} -t ${threads}
```
**Notes**: 
- ```${inputSubjectDirectory}``` needs to be in the same format as described in [Arranging Data](###Expected-Directory-structure-for-data) or you need to have a [data file](##Data-File-usage).
- `${threads}` are the maximum number of threads that can be used for computation and is generally dependent on the number of available CPU cores. Should be of type `int` and should satisfy: `0 < ${threads} < maximum_cpu_cores`. Depending on the type of CPU you have, it can vary from [1](https://ark.intel.com/content/www/us/en/ark/products/37133/intel-core-2-solo-processor-ulv-su3500-3m-cache-1-40-ghz-800-mhz-fsb.html) to [64](https://www.amd.com/en/products/cpu/amd-ryzen-threadripper-3990x) threads.

## Preparing Files

### For Training

-# Set the dataset in the above mentioned [format](https://github.com/CBICA/Deep-BET#expected-directory-structure-for-data)
-# Follow the Brain Preprocessing steps mentioned [here](https://github.com/CBICA/Deep-BET#brain-preprocessing-steps)
-# Follow the Skull Stripping intensity Standardization preprocessing mentioned [here](https://github.com/CBICA/Deep-BET#standardizing-dataset-intensities)
-# Insert these generated preprocessed files from *Step 3* in the config file. 
-# Follow similar steps for the validation dataset.

### For Testing

#### MA

This inference type does not need any preprocessing of input files, as everything is handled internally.

The input images can be directly passed to the [config file](./Deep_BET/config/test_params_ma.cfg).

#### Multi-4

If all the structural modalities (i.e., `T1, T2, T1ce, Flair`) are being used, processing the input data (as mentioned in the [Brain Preprocessing section](https://github.com/CBICA/Deep-BET#brain-preprocessing-steps)) is required. 

Pass the processed images over to the network via the [config files](./Deep_BET/config/test_params_multi_4.cfg).

## Running Instructions

We have two modes in here : `train` and `test`.

### Training

- Populate a config file with required parameters (please see [train_params.cfg](./Deep_BET/config/train_params.cfg) for an example)
- Note that preprocessed data in the specific format [ref](###Expected-Directory-structure-for-data) should be used.
- Invoke the following command:

```bash
deep_bet_run -params train_params.cfg -train True -dev $device -load $resume.ckpt
```

Note that ```-load $resume.ckpt``` is only needed if you are resuming your training. 

### Inference

- We have three modes here:
  - Modality Agnostic (MA)
  - Multi-4, i.e., all 4 modalities getting used
  - Single (weights would be updated soon) 
- Populate a config file with required parameters. Examples:
  - MA: [test_params_ma.cfg](./Deep_BET/config/test_params_ma.cfg)
  - Multi-4: [test_params.cfg](./Deep_BET/config/test_params_multi_4.cfg)
- It is highly suggested that Multi-4 should be only run with some certain preprocessing steps (link goes here) mentioned below.
- Mode refers to the inference type that you wish to run which is necessary
- Invoke the following command:

```bash
deep_bet_run -params $test_params_ma.cfg -test True -mode MA -dev $device
```
```bash
deep_bet_run -params $test_params_multi_4.cfg -test True -mode Multi-4 -dev $device
```

```$device``` refers to the GPU device where you want your code to run or the CPU.

## Converting weights after training

- After training a custom model, you shall have a `.ckpt` file instead of a `.pt` file.
- The file (convert_ckpt_to_pt.py)[https://github.com/CBICA/Deep-BET/blob/master/Deep_BET/utils/convert_ckpt_to_pt.py] can be used  to convert the file. 
  - Example:
    ```bash
    ./env/python Deep_BET/utils/convert_ckpt_to_pt.py -i ${path_to_ckpt_file_with_filename} -o {path_to_pt_file_with_filename}
    ```
- Please note that the if you wish to use your own weights, you can use the ```-load``` option.

## Data File usage

If the data is organized according to the above [instructions](###Expected-Directory-structure-for-data), the `csv_provided` variable can be set to `False`, and CSV file would be generated according to what is inserted in the `modalities` [line](https://github.com/CBICA/Deep-BET/blob/ce0463dad1eeb73cc78a5ef2b266f630723e009b/Deep_BET/config/test_params_ma.cfg#L19).

For example:
```
modalities = ['T1', 'T2', 'T1ce', 'Flair']
```
would result in the CSV created in the following format:

- Training

  `Patient_ID,gt_path,T1_path,T2_path,T1ce_path,Flair_path`
- Testing

  `Patient_ID,T1_path,T2_path,T1ce_path,Flair_path`

Now, if the data isn't organized, a CSV can be created in the above mentioned format and the `csv_provided` variable can be set to `True` and the CSV files can be provided in the following locations:

- [`test_csv`](https://github.com/CBICA/Deep-BET/blob/ce0463dad1eeb73cc78a5ef2b266f630723e009b/Deep_BET/config/test_params_ma.cfg#L15)
- [`train_csv`](https://github.com/CBICA/Deep-BET/blob/ce0463dad1eeb73cc78a5ef2b266f630723e009b/Deep_BET/config/train_params.cfg#L17)
- [`validation_csv`](https://github.com/CBICA/Deep-BET/blob/ce0463dad1eeb73cc78a5ef2b266f630723e009b/Deep_BET/config/train_params.cfg#L18).

## Citation

If you use this package, please cite the following paper:

- Thakur, S.P., Doshi, J., Pati, S., Ha, S.M., Sako, C., Talbar, S., Kulkarni, U., Davatzikos, C., Erus, G. and Bakas, S., 2019, October. Skull-Stripping of Glioblastoma MRI Scans Using 3D Deep Learning. In International MICCAI Brainlesion Workshop (pp. 57-68). Springer, Cham. DOI:10.1007/978-3-030-46640-4_6

## Notes

- **IMPORTANT**: This application is neither FDA approved nor CE marked, so the use of this package and any associated risks are the users' responsibility.
- Using this software is pretty trivial as long as instructions are followed. 
- You can use it in any terminal on a supported system. 
- The ```deep_bet_run``` command gets installed automatically. 
- We provide CPU (untested as of 2020/05/31) as well as GPU support. 
  - Running on GPU is a lot faster though and should always be preferred. 
  - You need an GPU memory of ~5-6GB for testing and ~8GB for training.

## TO-DO

- In inference, rename ```model_dir``` to ```results_dir``` for clarity in the configuration and script(s)
- Add CCA for post-processing
- Add link to CaPTk as suggested mechanism for preprocessing (can refer to ```BraTSPipeline``` application after my [PR](https://github.com/CBICA/CaPTk/pull/1061) gets merged to master)
- Test on CPU
- Move all dependencies to ```setup.py``` for consistency 
- Put option to write logs to specific files in output directory
- Remove ```-mode``` parameter in ```deep_bet_run```
- Windows support (this currently works but needs a few work-arounds)
- Please post any requests as issues on this repository or send email to software@cbica.upenn.edu

## Contact

Please email software@cbica.upenn.edu with questions.
