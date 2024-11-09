
import librosa
import numpy as np
import torch
from torch import nn


FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])

FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])

FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477),
                                (477, 474)])

FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])

FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])

FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])

FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472),
                                 (472, 469)])

FACEMESH_FACE_OVAL = frozenset([(389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162)])

FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
                           (4, 45), (45, 220), (220, 115), (115, 48),
                           (4, 275), (275, 440), (440, 344), (344, 278), ])

ALL_GROUPS = [FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, 
              FACEMESH_RIGHT_EYE, FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, 
              FACEMESH_NOSE]

ROI =  frozenset().union(*[FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, 
FACEMESH_RIGHT_EYE,FACEMESH_RIGHT_EYEBROW,FACEMESH_FACE_OVAL,FACEMESH_NOSE])            #131 keypoints

def get_indices_from_frozenset(frozenset_data):
    indices = set()
    for group in frozenset_data:
        for (start_idx, end_idx) in group:
            indices.add(start_idx)
            indices.add(end_idx)
    return sorted(indices)

FACEMESH_ROI_IDX = get_indices_from_frozenset(ALL_GROUPS)
FACEMESH_LIPS_IDX = get_indices_from_frozenset([FACEMESH_LIPS])
FACEMESH_FACES_IDX = get_indices_from_frozenset([FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, 
              FACEMESH_RIGHT_EYE, FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, 
              FACEMESH_NOSE])

def plot_landmark_connections(ax, landmarks, color):
    for group in ALL_GROUPS:
        for (start_idx, end_idx) in group:
            ax.plot([landmarks[FACEMESH_ROI_IDX.index(start_idx), 0], landmarks[FACEMESH_ROI_IDX.index(end_idx), 0]],
                    [landmarks[FACEMESH_ROI_IDX.index(start_idx), 1], landmarks[FACEMESH_ROI_IDX.index(end_idx), 1]],
                    color=color, lw=1)
            
            

emotion_labels = ["angry", "contempt", "disgusted", "fear", "happy", "neutral", "sad", "surprised"]
emotion_to_index = {label: idx for idx, label in enumerate(emotion_labels)}

def emotion_to_one_hot(emotion_label):
    one_hot_vector = np.zeros(len(emotion_labels))
    index = emotion_to_index[emotion_label]
    one_hot_vector[index] = 1
    return one_hot_vector

def extract_llf_features(audio_data, sr, n_fft, win_length, hop_length):
    # Rút trích đặc trưng âm thanh
    # Âm lượng
    rms = librosa.feature.rms(y=audio_data, frame_length=win_length, hop_length=hop_length, center=False)

    # Tần số cơ bản
    chroma = librosa.feature.chroma_stft(n_chroma=17,y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)

    # Tần số biên độ
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)

    # Mức độ biến đổi âm lượng và tần số
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_data, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    
    #Poly-features
    poly_features = librosa.feature.poly_features(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    
    # Compute zero-crossing rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=audio_data, frame_length=win_length, hop_length=hop_length, center=False)
    
    feats = np.vstack((chroma, #12
                spectral_contrast, #7
                spectral_centroid, #1
                spectral_bandwidth, #1
                spectral_flatness, #1
                spectral_rolloff, #1
                poly_features, #2
                rms, #1
                zcr #1
                )) 
    return feats

def calculate_LMD(pred_landmark, gt_landmark, norm_distance=1.0):
    euclidean_distance = torch.sqrt(torch.sum((pred_landmark - gt_landmark)**2, dim=(pred_landmark.ndim - 1)))
    norm_per_frame = torch.mean(euclidean_distance, dim=(pred_landmark.ndim - 2))
    q1 = torch.quantile(norm_per_frame, 0.25)
    q3 = torch.quantile(norm_per_frame, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_norm_per_frame = norm_per_frame[(norm_per_frame >= lower_bound) & (norm_per_frame <= upper_bound)]
    
    lmd = torch.divide(filtered_norm_per_frame, norm_distance)  
    return lmd.item()
    
