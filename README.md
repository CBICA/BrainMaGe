# BrainMaGe (Brain Mask Generator)

## Introduction

The Brain Mask Generator (BrainMaGe) is a deep-learning (DL) generalizable robust brain extraction (skull-stripping) tool explicitly developed for application in brain MRI scans with apparent pathologies, e.g., tumors. BrainMaGe introduces a modality-agnostic training method rather than one that needs a specific set or combination of modalities, and hence forces the model to learn the spatial relationships between the structures in the brain and its shape, as opposed to texture, and thereby overriding the need for a particular modality. If you want to read more about BrainMaGe, please use the link in [Citations](##Citations) to read the full performance evaluation we conducted, where we have proved that such a model will have comparable (and in most cases better) accuracy to other DL methods while keeping minimal computational and logistical requirements.

## Citations

If you use this package, please cite the following paper:

1. S.Thakur, J.Doshi, S.Pati, S.Rathore, C.Sako, M.Bilello, S.M.Ha, G.Shukla, A.Flanders, A.Kotrotsou, M.Milchenko, S.Liem, G.S.Alexander, J.Lombardo, J.D.Palmer, P.LaMontagne, A.Nazeri, S.Talbar, U.Kulkarni, D.Marcus, R.Colen, C.Davatzikos, G.Erus, S.Bakas, "Brain Extraction on MRI Scans in Presence of Diffuse Glioma: Multi-institutional Performance Evaluation of Deep Learning Methods and Robust Modality-Agnostic Training, NeuroImage, Epub-ahead-of-print, 2020. [DOI: 10.1016/j.neuroimage.2020.117081](https://doi.org/10.1016/j.neuroimage.2020.117081)

The following citations are previous conference presentations of related results:  

2. S.P.Thakur, J.Doshi, S.Pati, S.M.Ha, C.Sako, S.Talbar, U.Kulkarni, C.Davatzikos, G.Erus, S.Bakas, "Skull-Stripping of Glioblastoma MRI Scans Using 3D Deep Learning". In International MICCAI BrainLes Workshop, Springer LNCS, 57-68, 2019. [DOI: 10.1007/978-3-030-46640-4_6](https://doi.org/10.1007/978-3-030-46640-4_6)

3. S.Thakur, J.Doshi, S.M.Ha, G.Shukla, A.Kotrotsou, S.Talbar, U.Kulkarni, D.Marcus, R.Colen, C.Davatzikos, G.Erus, S.Bakas, "NIMG-40. ROBUST MODALITY-AGNOSTIC SKULL-STRIPPING IN PRESENCE OF DIFFUSE GLIOMA: A MULTI-INSTITUTIONAL STUDY", Neuro-Oncology, 21(Supplement_6): vi170, 2019. [DOI: 10.1093/neuonc/noz175.710](https://doi.org/10.1093/neuonc/noz175.710)


## Installation Instructions

Please note that python3 is required and [conda](https://www.anaconda.com/) is preferred.

```bash
git clone https://github.com/CBICA/BrainMaGe.git
cd BrainMaGe
git lfs pull
conda env create -f requirements.yml # create a virtual environment named brainmage
conda activate brainmage # activate it
latesttag=$(git describe --tags) # get the latest tag [bash-only]
echo checking out ${latesttag}
git checkout ${latesttag}
python setup.py install # install dependencies and BrainMaGe
```

## Generating brain masks for your data using our pre-trained models

- This application currently has two modes (more coming soon):
  - Modality Agnostic (MA)
  - Multi-4, i.e., using all 4 structural modalities

### Steps to run application

1. Co-registration within patient to the [SRI-24 atlas](https://www.nitrc.org/projects/sri24/) in the LPS/RAI space.

    An easy way to do this is using the [```BraTSPipeline``` application](https://cbica.github.io/CaPTk/preprocessing_brats.html) from the [Cancer Imaging Phenomics Toolkit (CaPTk)](https://github.com/CBICA/CaPTk/). This pipeline currently uses a pre-trained model to extract the skull but the processed images (in the order defined above till registration) are also saved.

2. Make an Input CSV including paths to the co-registered images (prepared in the previous step) that you wish to make brain masks.

  - Multi-4 (use all 4 structural modalities): Prepare a CSV file with the following headers:
  `Patient_ID,T1_path,T2_path,T1ce_path,Flair_path`

  - Modality-agnostic (works with any structural modality): Prepare a CSV file with the following headers:
  `Patient_ID_Modality,image_path`


3. Make config files:

    Populate a config file with required parameters. Examples:
    - MA: [test_params_ma.cfg](./BrainMaGe/config/test_params_ma.cfg)
    - Multi-4: [test_params.cfg](./BrainMaGe/config/test_params_multi_4.cfg)

    Where `mode` refers to the inference type, which is a required parameter

    **Note**: Alternatively, you can use the diretory structure similar to the training as desribed in the next section.

4. Run the application:

    ```bash
    conda activate brainmage
    brain_mage_run -params $test_params_ma.cfg -test True -mode $mode -dev $device
    ```

    Where:
    - ```$mode``` can be ```MA``` for modality agnostic or ```Multi-4```.
    - ```$device``` refers to the GPU device where you want your code to run or the CPU.

### Steps to run application (Alternative)

1.Although this method is much slower, and runs for single subject at a time, it works flawlessly on CPU's and GPU's.

    conda activate brainmage
    brain_mage_single_run -i $path_to_input.nii.gz -o $path_to_output_mask.nii.gz
    \ -m  $path_to_output_brain.nii.gz -dev $device
    
    Where:
    - `$path_to_input.nii.gz` can be path to the input file as a nifti.
    - `$path_to_output_mask.nii.gz` is the output path to save the mask for the nifti
    - `$path_to_output_brain.nii.gz` is the output path to brain for the nifti

## [ADVANCED] Train your own model

1. Co-registration within patient in a common atlas space such as the [SRI-24 atlas](https://www.nitrc.org/projects/sri24/) in the LPS/RAI space. 

    An easy way to do this is using the [```BraTSPipeline``` application](https://cbica.github.io/CaPTk/preprocessing_brats.html) from the [Cancer Imaging Phenomics Toolkit (CaPTk)](https://github.com/CBICA/CaPTk/).

    **Note**: Any changes done in this step needs to be reflected during the inference process.

2. Arranging the Input Data, co-registered in the previous step, to the following folder structure. Please note files must be named exactly as below (e.g. ${subjectName}_t1, ${subjectName}_mask.nii.gz etc.) 

    ```
    Input_Data_folder -- patient_1 -- patient_1_t1.nii.gz
                            -- patient_1_t2.nii.gz
                            -- patient_1_t1ce.nii.gz
                            -- patient_1_flair.nii.gz
                            -- patient_1_mask.nii.gz
                  patient_2 -- ...
                  ...
                  ...
                  patient_n -- ...
    ```

3. Standardizing Dataset Intensities

    Use the following command to standardize intensities for both training and validation data:

    ```bash
    brain_mage_intensity_standardize -i ${inputSubjectDirectory} -o ${outputSubjectDirectory} -t ${threads}
    ```

    - ```${inputSubjectDirectory}``` needs to be structured as described in the previous step (Arranging Data).
    - `${threads}` are the maximum number of threads that can be used for computation and is generally dependent on the number of available CPU cores. Should be of type `int` and should satisfy: `0 < ${threads} < maximum_cpu_cores`. Depending on the type of CPU you have, it can vary from [1](https://ark.intel.com/content/www/us/en/ark/products/37133/intel-core-2-solo-processor-ulv-su3500-3m-cache-1-40-ghz-800-mhz-fsb.html) to [112](https://www.intel.com/content/www/us/en/products/processors/xeon/scalable/platinum-processors/platinum-9282.html) threads.

4. Prepare configuration file

    Populate a config file with required parameters. Example: [train_params.cfg](./BrainMaGe/config/train_params.cfg)

    Change the ```mode``` variable in the config file based on what kind of model you want to train (either modality agnostic or multi-4).

5. Run the training:

    ```bash
    brain_mage_run -params train_params.cfg -train True -dev $device -load $resume.ckpt
    ```

    Note that ```-load $resume.ckpt``` is only needed if you are resuming your training. 

6. [OPTIONAL] Converting weights after training

  - After training a custom model, you shall have a `.ckpt` file instead of a `.pt` file.
  - The file [convert_ckpt_to_pt.py](./BrainMaGe/utils/convert_ckpt_to_pt.py) can be used  to convert the file. For example:
      ```bash
      ./env/python BrainMaGe/utils/convert_ckpt_to_pt.py -i ${path_to_ckpt_file_with_filename} -o {path_to_pt_file_with_filename}
      ```
  - Please note that the if you wish to use your own weights, you can use the ```-load``` option.


## Notes

- **IMPORTANT**: This application is neither FDA approved nor CE marked, so the use of this package and any associated risks are the users' responsibility.
- Please follow instructions carefully and for questions/suggestions, post an issue or [contact us](##Contact). 
- The ```brain_mage_run``` command gets installed automatically in the virtual environment.
- We provide CPU (untested as of 2020/05/31) as well as GPU support. 
  - Running on GPU is a lot faster though and should always be preferred. 
  - You need an GPU memory of ~5-6GB for testing and ~8GB for training.
- Added support for hole filling and largest CCA post processing

## TO-DO

- Windows support (this currently works but needs a few work-arounds)
- Give example of skull stripping dataset 
- In inference, rename ```model_dir``` to ```results_dir``` for clarity in the configuration and script(s)
- Test on CPU
- Move all dependencies to ```setup.py``` for consistency 
- Put option to write logs to specific files in output directory
- Remove ```-mode``` parameter in ```brain_mage_run```

## Contact

Please email software@cbica.upenn.edu with questions.
