import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T
from diffusers import AutoencoderKL
import cv2
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
import math
import torchvision.transforms.functional as F

# Thiết lập thiết bị (CPU hoặc GPU nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hàm đọc file JSON và trả về tensor
def read_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        # Chuyển dữ liệu từ JSON thành numpy array rồi thành torch.Tensor
        tensor = torch.tensor(data)  # Để giữ dạng tensor để sử dụng PyTorch .to()
    return tensor

# Load an image
image = cv2.imread("/home/cxnam/Documents/MEAD/M003/images/front_angry_level_1/001/00001.jpg")
h, w, _ = image.shape

# Convert the image color space to RGB (required by MediaPipe)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize FaceMesh model
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    results = face_mesh.process(image_rgb)
    
    if results.multi_face_landmarks:
        # Extract face landmarks
        for face_landmarks in results.multi_face_landmarks:
            # Create an empty mask
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Get landmark points
            face_points = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                face_points.append([x, y])
            
            # Convert points to a numpy array
            points = np.array(face_points, np.int32)
            
            # Create a convex hull around the landmarks
            convex_hull = cv2.convexHull(points)
            
            # Fill the mask with the convex hull
            cv2.fillConvexPoly(mask, convex_hull, 255)
            
            # Apply the mask on the original image
            # face_masked = cv2.bitwise_and(image, image, mask=mask)
            
            # Save the resulting face-masked image
            cv2.imwrite("mask.jpg", mask)
            
            # mask = torch.from_numpy(mask).to(device)
            

# Đọc hai file JSON
face_tensor = read_json_file('/home/cxnam/Documents/MEAD/M003/image_features/front_angry_level_1/004/00010.json')  # Tensor shape (4, 32, 32)
face_mask_tensor = read_json_file('/home/cxnam/Documents/MEAD/M003/image_features/front_angry_level_1/001/00001.json')  # Tensor shape (4, 32, 32)

face_tensor = face_tensor[0]

face_mask_tensor = face_mask_tensor[0]
face_mask_tensor[:,4*4:7*4, 2*4:6*4] = 1


region = face_tensor[:, 4*4:7*4, 2*4:6*4]

# Define the affine transformation (rotation + translation)
angle = 0  # Rotation angle in degrees
translate_y = -1  # Shift upwards by 10 pixels
translate_x = 0  # No horizontal shift

transformed_region = F.affine(region, angle=angle, translate=(translate_x, translate_y), scale=1.0, shear=(0, 0))

face_tensor_tf = face_tensor.clone()
face_tensor_tf[:, 4*4:7*4, 2*4:6*4] = transformed_region


face_swap_tensor = face_mask_tensor.clone()
face_swap_tensor[:, 4*4:7*4, 2*4:6*4] = face_tensor[:,4*4:7*4, 2*4:6*4]

# Tạo inv_normalize để biến đổi output
inv_normalize = T.Compose([
    T.Normalize(mean=[-1.0, -1.0, -1.0], std=[1.0 / 0.5, 1.0 / 0.5, 1.0 / 0.5]),
    T.ToPILImage()
])

# Tạo plot với kích thước phù hợp (2 hàng x 5 cột)
fig, axs = plt.subplots(3, 5, figsize=(15, 10))


# Hiển thị 4 channels của face_mask.json ở hàng dưới
for i in range(4):
    axs[0, i].imshow(face_mask_tensor[i].cpu().numpy())  # Tương tự với face_mask_tensor
    axs[0, i].set_title(f'mask channel {i+1}')
    axs[0, i].axis('off')

# Hiển thị 4 channels của face.json ở hàng trên
for i in range(4):
    axs[1, i].imshow(face_tensor[i].cpu().numpy())  # Chuyển từ tensor thành numpy trước khi hiển thị
    axs[1, i].set_title(f'face channel {i+1}')
    axs[1, i].axis('off')
    
# Hiển thị 4 channels của face.json ở hàng trên
for i in range(4):
    axs[2, i].imshow(face_swap_tensor[i].cpu().numpy())  # Chuyển từ tensor thành numpy trước khi hiển thị
    axs[2, i].set_title(f'face channel {i+1}')
    axs[2, i].axis('off')

# Bước giải mã inv_image thông qua AutoencoderKL và thêm vào hình thứ 5 của mỗi hàng
with torch.no_grad():
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    
    # Làm tương tự cho face_mask_tensor (cột thứ 5 hàng dưới)
    pred_mask_features = face_mask_tensor.to(device)
    pmf = pred_mask_features.unsqueeze(0)
    mask_samples = vae.decode(pmf)
    mask_output = mask_samples.sample[0]
    
    # Chuẩn hóa và chuyển thành ảnh
    inv_mask_image = inv_normalize(mask_output)

    # Hiển thị inv_mask_image ở cột thứ 5 của hàng dưới
    axs[0, 4].imshow(inv_mask_image)
    axs[0, 4].set_title(f'Decoded mask')
    axs[0, 4].axis('off')
    
    # Giải mã và chuyển đổi inv_image từ pred_features
    pred_features = face_tensor.to(device)  # Chuyển tensor sang thiết bị (CPU/GPU)
    pf = pred_features.unsqueeze(0)  # Thêm batch dimension
    samples = vae.decode(pf)
    output = samples.sample[0]
    
    # Chuẩn hóa lại và chuyển thành hình ảnh
    inv_image = inv_normalize(output)

    # Hiển thị inv_image ở cột thứ 5 của hàng trên
    axs[1, 4].imshow(inv_image)
    axs[1, 4].set_title(f'Decoded face')
    axs[1, 4].axis('off')
    
    # Giải mã và chuyển đổi inv_image từ pred_features
    pred_features = face_swap_tensor.to(device)  # Chuyển tensor sang thiết bị (CPU/GPU)
    pf = pred_features.unsqueeze(0)  # Thêm batch dimension
    samples = vae.decode(pf)
    output = samples.sample[0]
    
    # Chuẩn hóa lại và chuyển thành hình ảnh
    inv_image = inv_normalize(output)
    inv_part = cv2.bitwise_and(np.array(inv_image), np.array(inv_image), mask=mask)    
    inv_mask = cv2.bitwise_not(mask)
    raw_part = cv2.bitwise_and(image_rgb, image_rgb, mask=inv_mask)
    face_masked = cv2.bitwise_or(inv_part, raw_part)

    # Hiển thị inv_image ở cột thứ 5 của hàng trên
    axs[2, 4].imshow(face_masked)
    axs[2, 4].set_title(f'Decoded face swap')
    axs[2, 4].axis('off')

    

# Chỉnh layout và lưu ảnh thành file
plt.tight_layout()
plt.savefig('test.jpg')

# Hiển thị ảnh đã lưu
plt.show()
