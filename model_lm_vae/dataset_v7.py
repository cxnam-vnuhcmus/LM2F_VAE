import json
from tqdm import tqdm
import torch
import torch.utils.data as td
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
import numpy as np
import random
import cv2
import librosa
from PIL import Image
import torchvision.transforms as T
from .utils import FACEMESH_ROI_IDX, extract_llf_features, ROI
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
import torchvision.transforms.functional as F


class Dataset(td.Dataset):

    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = kwargs["data_root"]
        self.data_file = kwargs["data_file"]
        self.audio_dataroot = os.path.join(self.data_root, kwargs["audio_dataroot"])
        self.audio_feature_dataroot = os.path.join(self.data_root, kwargs["audio_feature_dataroot"])
        self.visual_dataroot = os.path.join(self.data_root, kwargs["visual_dataroot"])
        self.visual_feature_dataroot = os.path.join(self.data_root, kwargs["visual_feature_dataroot"])
        self.visual_mask_feature_dataroot = os.path.join(self.data_root, kwargs["visual_mask_feature_dataroot"])
        self.lm_dataroot = os.path.join(self.data_root, kwargs["lm_dataroot"])
        self.lm_feature_dataroot = os.path.join(self.data_root, kwargs["lm_feature_dataroot"])
        
        self.train = kwargs["train"]
        n_folders = kwargs["n_folders"]

        
        self.transform = T.Compose([
            T.Resize((256 , 256 )),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])   
        self.inv_normalize = T.Compose([
            # T.Normalize(mean=[-1.0, -1.0, -1.0], std=[1.0/0.5, 1.0/0.5, 1.0/0.5]),
            T.ToPILImage()
        ]) 
        self.img_transform = T.Compose([
            T.Resize((256 , 256 )),
            T.ToTensor()
        ])   
        
        # if self.train:
        if os.path.isdir(self.data_file):
            persons = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.data_file))][:n_folders]
            data_path = os.path.join(self.data_file,'{p}.txt')
        else:
            persons, _ = os.path.splitext(os.path.basename(self.data_file))   
            persons = [persons] 
            data_path = self.data_file
        
        filelists = []
        for p in tqdm(persons, total=len(persons)):
            data_path_p = data_path.format(p=p)
            with open(data_path_p, 'r') as file:
                for line in file:
                    line = line.strip()
                    filelists.append(f"{p}\t{line}")
        
        random.seed(0)
        random.shuffle(filelists)
        if self.train:
            filelists = filelists[:int(len(filelists) * 0.9)]
        else:
            filelists = filelists[int(len(filelists) * 0.9):] 
                    
        self.all_datas = self.data_augmentation(filelists)
        random.shuffle(self.all_datas)            
    
    def data_augmentation(self, filelists):
        all_datas = []
        for fileline in tqdm(filelists, desc="Loading datas"):
            p,line = fileline.strip().split("\t")
            audio_p = self.audio_dataroot.format(p=p)
            audio_feat_p = self.audio_feature_dataroot.format(p=p)
            visual_p = self.visual_dataroot.format(p=p)
            lm_p = self.lm_dataroot.format(p=p)
            lm_feat_p = self.lm_feature_dataroot.format(p=p)
            visual_feat_p = self.visual_feature_dataroot.format(p=p)
            visual_mask_feat_p = self.visual_mask_feature_dataroot.format(p=p)
            
            audio_name = os.path.join(audio_feat_p, f'{line}.json')
            lm_folder = os.path.join(lm_p, line)
            lm_feat_folder = os.path.join(lm_feat_p, line)
            vs_feat_folder = os.path.join(visual_feat_p, line)
            vs_mask_feat_folder = os.path.join(visual_mask_feat_p, line)
            vs_folder = os.path.join(visual_p, line)
            
            n_frame = 3
            lm_paths = sorted(os.listdir(lm_feat_folder))
            for i in range(2, len(lm_paths)-10,n_frame):
                sequence_path = []
                for j in range(i, i+n_frame):
                    lm_path = os.path.join(lm_folder,lm_paths[j])
                    lm_feat_path = os.path.join(lm_feat_folder,lm_paths[j])
                    vs_feat_path = os.path.join(vs_feat_folder,lm_paths[j])
                    vs_mask_feat_path = os.path.join(vs_mask_feat_folder,lm_paths[j])
                    vs_path = os.path.join(vs_folder,lm_paths[j].replace("json","jpg"))
                    if os.path.exists(vs_path) and os.path.exists(vs_feat_path) and os.path.exists(vs_mask_feat_path) and os.path.exists(lm_path) and os.path.exists(lm_feat_path):
                        sequence_path.append((lm_path, lm_feat_path, vs_path, vs_feat_path, vs_mask_feat_path, audio_name, max(0,j-2)))
                if len(sequence_path) == n_frame:
                    all_datas.append(sequence_path)
        return all_datas
    
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def __len__(self):
        return len(self.all_datas)
        # return 1

    def __getitem__(self, idx):        
        lm_datas = []
        lm_features = []
        gt_img_features = []
        gt_img_mask_features = []
        vs_paths = []
        mfcc_features = []
        gt_imgs = []
        
        for item in self.all_datas[idx]:
            (lm_path, lm_feat_path, vs_path, vs_feat_path, vs_mask_feat_path, audio_name, seg_idx) = item
            
            #landmark_feat
            with open(lm_path, "r") as f:
                lm_data = json.load(f)
            lm_roi = []
            for i in FACEMESH_ROI_IDX:
                lm_roi.append(lm_data[i])
            lm_roi = np.asarray(lm_roi)
            lm_roi = torch.FloatTensor(lm_roi).unsqueeze(0)
            lm_roi = lm_roi / 256.0
            lm_datas.append(lm_roi)
            
            #landmark_feat
            with open(lm_feat_path, "r") as f:
                lm_feature = json.load(f)
            lm_feature = torch.FloatTensor(lm_feature) #(1, 4, 32, 32)  
            lm_features.append(lm_feature)
            
            #img_feature
            with open(vs_feat_path, "r") as f:
                img_feature = json.load(f)
            gt_img_feature = torch.FloatTensor(img_feature) #(1, 4, 32, 32)
            gt_img_features.append(gt_img_feature)
            
            #img_mask_feature
            with open(vs_mask_feat_path, "r") as f:
                img_mask_feature = json.load(f)
            gt_img_mask_feature = torch.FloatTensor(img_mask_feature) #(1, 4, 32, 32)
            gt_img_mask_features.append(gt_img_mask_feature)
            
            #gt_img
            vs_paths.append(vs_path)
        
            gt_img = Image.open(vs_path)
            gt_img = self.img_transform(gt_img)
            gt_img = gt_img.unsqueeze(0)
            gt_imgs.append(gt_img)
            
            #audio
            # with open(audio_name, "r") as f:
            #     data = json.load(f)
            # mfcc_db = torch.tensor(data["mfcc"])
            # mfcc_segment = mfcc_db[seg_idx:seg_idx + 5, :] #(5, 80)
            # mfcc_features.append(mfcc_segment)
            
        lm_datas = torch.cat(lm_datas, dim=0)
        lm_features = torch.cat(lm_features, dim=0)
        gt_img_features = torch.cat(gt_img_features, dim=0)
        gt_img_mask_features = torch.cat(gt_img_mask_features, dim=0)
        # mfcc_features = torch.stack(mfcc_features, dim=0)
        gt_imgs = torch.cat(gt_imgs, dim=0)
        
        ref_lm_features = []
        ref_img_features = []
        while True:
            ref_idx = random.randint(0, len(self.all_datas)-1)
            if ref_idx != idx:
                break
        for item in self.all_datas[ref_idx]:
            (lm_path, lm_feat_path, vs_path, vs_feat_path, vs_mask_feat_path, audio_name, seg_idx) = item
            
            #landmark_feat
            with open(lm_feat_path, "r") as f:
                lm_feature = json.load(f)
            lm_feature = torch.FloatTensor(lm_feature) #(1, 4, 32, 32)  
            ref_lm_features.append(lm_feature)
            
            #img_feature
            with open(vs_feat_path, "r") as f:
                img_feature = json.load(f)
            ref_img_feature = torch.FloatTensor(img_feature) #(1, 4, 32, 32)
            ref_img_features.append(ref_img_feature)
        
        ref_lm_features = torch.cat(ref_lm_features, dim=0)
        ref_img_features = torch.cat(ref_img_features, dim=0)
        
        return {
            "gt_lm_feature": lm_features,
            "gt_img_feature": gt_img_features,
            "ref_lm_feature": ref_lm_features,
            "ref_img_feature": ref_img_features,
            "gt_mask_img_feature": gt_img_mask_features,
            "vs_path": vs_paths,
            "gt_img": gt_imgs,
            "gt_lm": lm_datas
        }

    def collate_fn(self, batch):
        # Initialize an empty dictionary to store collated outputs
        collated_batch = {}
        
        batch_size = len(batch)
        for key in batch[0].keys():
            # Collect all values corresponding to the current key across the batch
            elements = [item[key] for item in batch]
            
            # Filter out None values to maintain consistent batch size
            keep_ids = [idx for idx, item in enumerate(elements) if item is not None]
            
            if len(keep_ids) != batch_size:
                raise ValueError("Batch elements have inconsistent batch sizes.")
                        
            # Keep only the valid elements
            filtered_elements = [elements[idx] for idx in keep_ids]
            
            # Stack tensors or keep lists based on the data type
            if isinstance(filtered_elements[0], torch.Tensor):
                collated_batch[key] = torch.stack(filtered_elements, dim=0)
            else:
                collated_batch[key] = filtered_elements
        return collated_batch
