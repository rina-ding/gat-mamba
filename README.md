# gat-mamba
Combining Graph Neural Network and Mamba to Capture Local and Global Tissue Spatial Relationships in Whole Slide Images

## Instructions 
### Required packages
First, create a pytorch docker container using:
```
docker run  --shm-size=2g --gpus all -it --rm -v /:/workspace -v /etc/localtime:/etc/localtime:ro nvcr.io/nvidia/pytorch:24.02-py3
```
Then install all packages listed in `./requirements/requirements.txt`.

### Preprocessing
Download NLST data from [NLST](https://wiki.cancerimagingarchive.net/display/NLST/NLST+Pathology), download TCGA data from [TCGA-LUAD](https://portal.gdc.cancer.gov/projects/TCGA-LUAD).

Use [generate_tiles.py](./preprocessing/generate_tiles.py) to generate tiles of specified magnification level specified by `mag_level (string)`, the parent folder path containing all patients and slides `wsi_root_path`, destination folder path `wsi_tiles_root_dir`, and cohort name `cohort_name`('nlst', 'tcga').  

Each dataset's folder structure should be:
```
  ├── <patient_id>                   
  │   ├── <slide_id>   
```
### Feature extraction
Run [main_uni_and_luad_subtype.py](./feature_extraction/main_uni_and_luad_subtype.py) to extract all tile/node features using the tiled whole slide images from the previous step as input. The pretrained LUAD subtype classifier model weights can be accessed [here](https://github.com/rina-ding/ssl_luad_classification/tree/main/modeling/downstream_ensemble/model_weights).

### Graph data construction
Run [create_dataset_uni.py](./modeling/create_dataset_uni.py) to get the procesed graph dataset object named `graph_data.pt` that can be fed into the graph modeling code in the next step, using the extracted features from the previous step as input.

### Modeling
Run [main_gat_mamba.py](./modeling/main_gat_mamba.py) using `graph_data.pt` as input.