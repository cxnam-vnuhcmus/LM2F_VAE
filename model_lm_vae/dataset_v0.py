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
import torch.nn.functional as F

class Dataset(td.Dataset):

    def __init__(self, 
                 data_root: str,
                 data_file: str,
                 n_folders: int,
                 audio_dataroot: str,
                 visual_dataroot: str,
                 visual_feature_dataroot: str, 
                 lm_dataroot: str,
                 fps: int,
                 img_size: int,
                 sample_rate: int,
                 n_mels: int,
                 n_fft: int,
                 win_length: int,
                 hop_length: int,
                 n_frames: int,
                 train: bool,
                 **_
                 ):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.data_file = data_file
        self.audio_dataroot = os.path.join(self.data_root, audio_dataroot)
        self.visual_dataroot = os.path.join(self.data_root, visual_dataroot)
        self.visual_feature_dataroot = os.path.join(self.data_root, visual_feature_dataroot)
        self.lm_dataroot = os.path.join(self.data_root, lm_dataroot)
        self.fps = fps
        self.img_size = img_size
        self.sr = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_frames = n_frames
        self.train = train    
        
        self.transform = T.Compose([
            T.Resize((self.img_size , self.img_size )),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])   
        self.inv_normalize = T.Compose([
            T.Normalize(mean=[-1.0, -1.0, -1.0], std=[1.0/0.5, 1.0/0.5, 1.0/0.5]),
            T.ToPILImage()
        ]) 
        self.img_transform = T.Compose([
            T.Resize((self.img_size , self.img_size )),
            T.ToTensor()
        ])   
        
        
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
    
    def data_augmentation(self, filelists):
        all_datas = []
        for fileline in tqdm(filelists, desc="Loading datas"):
            p,line = fileline.strip().split("\t")
            audio_p = self.audio_dataroot.format(p=p)
            visual_p = self.visual_dataroot.format(p=p)
            lm_p = self.lm_dataroot.format(p=p)
            visual_feat_p = self.visual_feature_dataroot.format(p=p)
            
            lm_folder = os.path.join(lm_p, line)
            vs_feat_folder = os.path.join(visual_feat_p, line)
            vs_folder = os.path.join(visual_p, line)
            
            lm_paths = sorted(os.listdir(lm_folder))
            for i in range(0, len(lm_paths), 2):
                lm_path = os.path.join(lm_folder,lm_paths[i])
                vs_feat_path = os.path.join(vs_feat_folder,lm_paths[i])
                vs_path = os.path.join(vs_folder,lm_paths[i].replace("json","jpg"))
                
                if os.path.exists(vs_path) and os.path.exists(vs_feat_path):
                    all_datas.append((lm_path, vs_feat_path, vs_path))
                
        return all_datas
        
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def __len__(self):
        return len(self.all_datas)
        # return 1

    def __getitem__(self, idx):        
        (lm_path, vs_feat_path, vs_path) = self.all_datas[idx]

        #landmark
        lm_roi = []
        with open(lm_path, "r") as f:
            lm_data = json.load(f)
            for i in range(len(lm_data)):
                lm_roi.append(lm_data[i])
        lm_roi = np.asarray(lm_roi)
        lm_roi = lm_roi / 256.0
        # lm_roi = ((lm_roi - 128.0) / 128.0) * 5.0 + 0.5

        lm_roi = torch.FloatTensor(lm_roi).unsqueeze(0)
                    
        #img_feature
        with open(vs_feat_path, "r") as f:
            img_feature = json.load(f)
        gt_img_feature = torch.FloatTensor(img_feature) #(1,4,32,32)

        #image
        # gt_img = Image.open(vs_path)
        # gt_img = self.img_transform(gt_img)
        # gt_img = gt_img.unsqueeze(0)
        
        return (lm_roi, gt_img_feature, None, vs_path)

    def collate_fn(self, batch):
        batch_landmark, batch_gt_img_feature, batch_gt_img, batch_vs_path = zip(*batch)
        keep_ids = [idx for idx, (_, _) in enumerate(zip(batch_landmark, batch_gt_img_feature))]
            
        if not all(img is None for img in batch_landmark):
            batch_landmark = [batch_landmark[idx] for idx in keep_ids]
            batch_landmark = torch.cat(batch_landmark, dim=0)
        else:
            batch_landmark = None
        
        if not all(img is None for img in batch_gt_img_feature):
            batch_gt_img_feature = [batch_gt_img_feature[idx] for idx in keep_ids]
            batch_gt_img_feature = torch.cat(batch_gt_img_feature, dim=0)
        else:
            batch_gt_img_feature = None
            
        # if not all(img is None for img in batch_gt_img):
        #     batch_gt_img = [batch_gt_img[idx] for idx in keep_ids]
        #     batch_gt_img = torch.cat(batch_gt_img, dim=0)
        # else:
        #     batch_gt_img = None
            
        if not all(img is None for img in batch_vs_path):
            batch_vs_path = [batch_vs_path[idx] for idx in keep_ids]
        else:
            batch_vs_path = None
            
        return batch_landmark, batch_gt_img_feature, None, batch_vs_path
