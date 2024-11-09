# from diffusers import AutoencoderTiny
from diffusers import AutoencoderKL
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo AutoencoderTiny
# autoencoder = AutoencoderTiny.from_pretrained("madebyollin/taesd")
autoencoder = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")
autoencoder = autoencoder.to(device)


# image = cv2.imread("/home/cxnam/Documents/MEAD/M003/images/front_angry_level_1/001/00001.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 256 )),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ]) 

# image_tensor = transform(image)
# image_tensor = image_tensor.unsqueeze(0)
# image_tensor = image_tensor.to(device)

# with torch.no_grad():  
#     latent_vectors = autoencoder.encode(image_tensor).latent_dist.sample()

filename = '/home/cxnam/Documents/MEAD/M003/image_features/front_angry_level_1/004/00010.json'

with open(filename, 'r') as file:
    data = json.load(file)
    latent_vectors = torch.tensor(data).to(device)

with torch.no_grad():
    reconstructed_images = autoencoder.decode(latent_vectors).sample

# print(f"Original images shape: {image.shape}")
print(f"Latent vectors shape: {latent_vectors.shape}")
print(f"Reconstructed images shape: {reconstructed_images.shape}")

inv_normalize = transforms.Compose([
    # transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[1/0.5, 1/0.5, 1/0.5]),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # transforms.ToPILImage()
])

reconstructed_image = reconstructed_images.squeeze(0)  # Bỏ chiều batch để còn [3, 256, 256]
min_value = reconstructed_image.min()
max_value = reconstructed_image.max()
reconstructed_image = (reconstructed_image - min_value)/(max_value - min_value)
# reconstructed_image = inv_normalize(reconstructed_image)
min_value = reconstructed_image.min()
max_value = reconstructed_image.max()

print(f'Min value: {min_value}, Max value: {max_value}')


# reconstructed_image_tensor = (reconstructed_image_tensor * 0.5) + 0.5  # Giải chuẩn hóa nếu cần thiết
# reconstructed_image_tensor = torch.clamp(reconstructed_image_tensor, 0, 1)  # Đảm bảo không vượt ngoài khoảng [0, 1]

to_pil = transforms.ToPILImage()
reconstructed_image = to_pil(reconstructed_image)
reconstructed_image.save("test.jpg")
