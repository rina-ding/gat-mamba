# gat-mamba
Combining Graph Neural Network and Mamba to Capture Local and Global Tissue Spatial Relationships in Whole Slide Images

In computational pathology, whole slide images (WSIs) are typically segmented into small tiles for analysis due to their large size. A critical aspect of this analysis is the method of aggregating information from these tiles to make predictions at the WSI level. This study introduces a model that treats each tile as a node in a graph and combines a message-passing graph neural network (GNN) with a state space model Mamba to capture both local and global spatial relationships among the tiles in WSIs. The effectiveness of the model was demonstrated on  early-stage lung adenocarcinomas (LUAD) progression-free suvival prediction. We  compared the model with other state-of-the-art methods for tile level information aggregation in WSIs, including tile-level information symmary statistics-based aggregation, multiple instance learning (MIL)-based aggregation, GNN-based aggregation, and GNN-transformer-based aggregation. Additional experiments showed the impact of different types of node features and different tile sampling strategies on the model performance. This work can be easily extended to any WSI-based analysis.
![overview](overview.png)

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

Input data structure:
```
  ├── <patient_id>                   
  │   ├── <slide_id1.svs>  
  │   ├── <slide_id2.svs>   
  │   ├── <...>    
```
Output data structure:
```
  ├── <patient_id>                   
  │   ├── <tiles>
  │   ├──│   ├── <tile_id1.png>
  │   ├──│   ├── <tile_id2.png>
  │   ├──│   ├── <...>

```
### Feature extraction
Run [main_uni_and_luad_subtype.py](./feature_extraction/main_uni_and_luad_subtype.py) to extract all tile/node features using the tiled whole slide images from the previous step as input. Set the input path in the code `parent_dir_for_tiles` to be the output path from the previous step. The pretrained LUAD subtype classifier model weights can be accessed [here](https://github.com/rina-ding/ssl_luad_classification/tree/main/modeling/downstream_ensemble/model_weights).

Output data structure:
```
  ├── <patient_id1.csv>                   
  ├── <patient_id2.csv> 
  ├── <...> 
```
where in each csv file, the first column is slide id, second column tile name, third column patient's time to event (or follow-up time), forth column patient's event status, fifth column the LUAD histologic subtype, and the rest of the 1024 columns are the UNI features.

### Graph data construction
Run [create_dataset_uni.py](./modeling/create_dataset_uni.py) to get the procesed graph dataset object named `graph_data.pt` that can be fed into the graph modeling code in the next step, using the extracted features from the previous step as input. Set the input path in the code `parent_dir_node_features` to be the output path from the previous step.

Output data structure:
```
  ├── <processed>                   
  │   ├── <graph_data.pt>
  │   ├── <pre_filter.pt> 
  │   ├── <pre_transform.pt> 

```
### Modeling
Run [main_gat_mamba.py](./modeling/main_gat_mamba.py) using the path of `processed` as input by changing `processed_data_path` in the code.