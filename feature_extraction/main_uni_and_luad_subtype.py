
import torch
import numpy as np
import os
from glob import glob
import pandas as pd
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from luad_subtype_classifier_model import ModifiedResNet, ResNetEncoder
import torch.nn.functional as F

from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
from natsort import natsort_keygen

class DataProcessor(Dataset):
    def __init__(self, imgs_dir, transformations=None):
        self.imgs_ids = imgs_dir
        self.transformations = transformations

    def __getitem__(self, i):
        img_file = self.imgs_ids[i]
        tile_name = os.path.basename(img_file).replace('.png', '')
        
        img = np.asarray(Image.open(img_file))

        if self.transformations:
            img_normed = self.transformations(img)

        combined_img = np.concatenate((img_normed, img_normed), axis = 0)

        return {'image_combined': combined_img, 'image_single': img_normed, 'label' : tile_name}

    def __len__(self):
        return len(self.imgs_ids)

class FeatureExtractor:
    def __init__(self):
        self.device = self._get_device()

    def _get_device(self):
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        return device

    # Code is based on https://github.com/mahmoodlab/uni
    def extract_features_uni(self, path_to_images):
    
        # # pretrained=True needed to load UNI weights (and download weights for the first time)
        # # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
        model_uni = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True).to(self.device)
        transform = create_transform(**resolve_data_config(model_uni.pretrained_cfg, model=model_uni))
        model_uni.eval()

        features, tile_names = [], []
      
        for tile_path in path_to_images:
            image_tile = Image.open(tile_path)
            
            image_tile_processed_uni = transform(image_tile).unsqueeze(dim=0).to(self.device) # Image (torch.Tensor) with shape [1, 3, 224, 224] following image resizing and normalization (ImageNet parameters)
            with torch.inference_mode():
                image_features_uni = model_uni(image_tile_processed_uni) # Extracted features (torch.Tensor) with shape [1,1024]
            
            image_features_uni = image_features_uni.cpu().numpy()
            all_image_features = image_features_uni

            features.extend(all_image_features)
            tile_names.append(os.path.basename(tile_path).replace('.png', ''))        

        features_array = np.asarray(features)
        df_features = pd.DataFrame(features_array)
        return df_features, tile_names
    
    def extract_features_luad(self, path_to_images):
       
        batch = 1
        dataset = DataProcessor(imgs_dir=path_to_images, transformations=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
        print("Number of images ", len(dataset))
        testloader = DataLoader(dataset, batch_size=batch, shuffle=False, drop_last=False)
        
        model1 = ModifiedResNet(self.num_classes).to(self.device)
        weights = torch.load(os.path.join(ssl1_model_path, 'resnet18_fold1.pth'), map_location=self.device)
        model1.load_state_dict(weights)

        model2 = ModifiedResNet(self.num_classes).to(self.device)
        weights = torch.load(os.path.join(ssl2_model_path, 'resnet18_fold1.pth'), map_location=self.device)
        model2.load_state_dict(weights)

        model3 = ResNetEncoder(self.num_classes).to(self.device)
        weights = torch.load(os.path.join(ssl3_model_path, 'resnet18_fold1.pth'), map_location=self.device)
        model3.load_state_dict(weights)

        # extract features
        with torch.no_grad():
            model1.eval()
            model2.eval()
            model3.eval()
            
            features, tile_names, predicted_class = [], [], []
            for data in testloader:
                image_combined, image_single, labels = data['image_combined'].to(self.device, dtype=torch.float), data['image_single'].to(self.device, dtype=torch.float), data['label']
                output1, hidden_layer1 = model1(image_combined)
                output_pb1 = F.softmax(output1.cpu(), dim=1)
                output_pb1 = output_pb1.numpy().tolist()

                output2, hidden_layer2 = model2(image_combined)
                output_pb2 = F.softmax(output2.cpu(), dim=1)
                output_pb2 = output_pb2.numpy().tolist()

                output3, hidden_layer3 = model3(image_single)
                output_pb3 = F.softmax(output3.cpu(), dim=1)
                output_pb3 = output_pb3.numpy().tolist()

                output_pb_ensemble = []
                for i in range(len(output_pb1)):
                    pb_one_patient_ssl1 = output_pb1[i]
                    pb_one_patient_ssl2 = output_pb2[i]
                    pb_one_patient_ssl3 = output_pb3[i]
                    pb_one_patient_updated = []
                    for j in range(len(pb_one_patient_ssl1)):
                        if j == 0: # Probability for lepidic
                            pb_one_patient_updated.append(pb_one_patient_ssl1[j] * 0.6 + pb_one_patient_ssl2[j] * 0.2 + pb_one_patient_ssl3[j] * 0.2)
                        elif j == 1: # Probability for acinar
                            pb_one_patient_updated.append(pb_one_patient_ssl1[j] * 0.6 + pb_one_patient_ssl2[j] * 0.2 + pb_one_patient_ssl3[j] * 0.2)
                        elif j == 2: # Probability for pap
                            pb_one_patient_updated.append(pb_one_patient_ssl1[j] * 0.6 + pb_one_patient_ssl2[j] * 0.2 + pb_one_patient_ssl3[j] * 0.2)
                        elif j == 3: # Probability for micropap
                            pb_one_patient_updated.append(pb_one_patient_ssl1[j] * 0.2 + pb_one_patient_ssl2[j] * 0.2 + pb_one_patient_ssl3[j] * 0.6)
                        elif j == 4: # Probability for solid
                            pb_one_patient_updated.append(pb_one_patient_ssl1[j] * 0.2 + pb_one_patient_ssl2[j] * 0.2 + pb_one_patient_ssl3[j] * 0.6)
                        elif j == 5: # Probability for nontumor
                            pb_one_patient_updated.append(pb_one_patient_ssl1[j] * 0.2 + pb_one_patient_ssl2[j] * 0.2 + pb_one_patient_ssl3[j] * 0.6)
                        
                    output_pb_ensemble.append(pb_one_patient_updated)
                
                top_ps, top_class = torch.from_numpy(np.asarray(output_pb_ensemble)).topk(1, dim=1)
                
                reshaped_feature1 = np.reshape(hidden_layer1.cpu().numpy(), (len(labels), 512))
                reshaped_feature2 = np.reshape(hidden_layer2.cpu().numpy(), (len(labels), 512))
                reshaped_feature3 = np.reshape(hidden_layer3.cpu().numpy(), (len(labels), 512))
        
                reshaped_features_combined = np.mean(np.vstack((reshaped_feature1, reshaped_feature2, reshaped_feature3)), axis = 0)
                reshaped_features_combined = np.reshape(reshaped_features_combined, (len(labels), 512))
                features.extend(reshaped_features_combined)
                tile_names.extend(list(labels))
                predicted_class.extend(list(top_class.flatten().numpy()))
            
            features_array = np.asarray(features)
            df_features = pd.DataFrame(features_array)
            df_features.insert(0, 'tile_name', tile_names)
            df_features.insert(1, 'predicted_class', predicted_class)
            return df_features
                                
if __name__ == "__main__":
    login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens # Code is based on https://github.com/mahmoodlab/uni

    FEATURE_DIR = 'dir to save the extracted features' 
    if not os.path.exists(FEATURE_DIR):
        os.makedirs(FEATURE_DIR)
    all_cases = natsorted(glob(os.path.join('parent dir to the tiled WSIs'))) 
    df_clinical = 'path to the clinical data that contains a patient outcome (event and days)'
    ssl1_model_path = 'path to luad subtype classification model1'
    ssl2_model_path = 'path to luad subtype classification model2'
    ssl3_model_path = 'path to luad subtype classification model3'
    for i in range(len(all_cases)):
        tiles_root_dir = glob(os.path.join(all_cases[j], 'tiles_*x_png', '*.png'))
        pid = os.path.basename(all_cases[i])           
        print('For patient ', pid)
        extractor = FeatureExtractor()
        df_features_uni, tile_names = extractor.extract_features_uni(tiles_root_dir)
        df_features_deep, tile_names = extractor.extract_features_luad(tiles_root_dir)

         # Adding slide id, tile name into the feature matrix
        df_features_uni.insert(0, 'sid', [x.split('-tile-')[0] for x in df_features_uni['tile_name']])
        df_features_uni.insert(1, 'tile_name', tile_names)

        # Add predicted luad subtype, survival event, and survival days
        event = df_clinical[df_clinical['pid'] == pid]['event']
        days = df_clinical[df_clinical['pid'] == pid]['days']
        df_features_uni.insert(df_features_uni.columns.get_loc('tile_name')+1, 'event', len(df_features_uni) * [event])
        df_features_uni.insert(df_features_uni.columns.get_loc('tile_name')+1, 'days', len(df_features_uni) * [days])
        df_features_uni = df_features_uni.sort_values(by = ['tile_name'], key=natsort_keygen())
        df_features_deep = df_features_deep.sort_values(by = ['tile_name'], key=natsort_keygen())
        df_features_uni.insert(df_features_uni.columns.get_loc('days')+1, 'predicted_class', df_features_deep['predicted_class'])

        # Saving the features into a csv file
        df_features_uni.to_csv(os.path.join(FEATURE_DIR, str(pid) + '.csv'))
