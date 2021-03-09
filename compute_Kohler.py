import numpy as np
import math
import cv2
import os
import matplotlib.pyplot as plt
import torch
import piq
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

deblu_root = './res_kohler1/'
sharp_root = './datasets/Kohler_multi1/'

deblu_list = os.listdir(deblu_root)
sharp_list = os.listdir(sharp_root)
sharp_list = sorted(sharp_list, key=str.lower)

num_imgs = len(deblu_list)
PSNR_all = []
SSIM_all = []
count_k = 1
psnr_k = []
ssim_k = []
vif_k = []
vsi_k = []
haar_k = []
kernel_p = []
kernel_s = []
kernel_f = []
kernel_v = []
kernel_h = []
VIF_all = []
VSI_all = []
Haar_all = []

for n, item in enumerate(sharp_list):
    if not item.startswith('.'):

        name = item[:2]
        name_sharp = sharp_list[n]
        name_deblu = item[:2] + "0001.png"

        path_deblu = os.path.join(deblu_root, name_deblu)
        path_sharp = os.path.join(sharp_root, name_sharp)

        img_deblu = cv2.imread(path_deblu, cv2.IMREAD_COLOR).astype(np.float)
        img_sharp = cv2.imread(path_sharp, cv2.IMREAD_COLOR).astype(np.float)

        _, _, img_sharp = np.split(img_sharp, 3, axis=1)

        # cv2.imwrite('img_blur.png', img_blur)
        #cv2.imwrite('img_deblu.png', img_deblu)
        #cv2.imwrite('img_sharp.png', img_sharp)

        img_sharp = img_sharp[:, :, [2, 1, 0]]
        img_deblu = img_deblu[:, :, [2, 1, 0]]

        # plt.figure()
        # plt.imshow(img_sharp/255)
        # plt.title('sharp')
        #
        # plt.figure()
        # plt.imshow(img_deblu/255)
        # plt.title('denoise')
        # plt.show()

        psnr_n = psnr(img_sharp, img_deblu, data_range=255)
        ssim_n = ssim(img_deblu / 255, img_sharp / 255, gaussian_weights=True, multichannel=True,
                      use_sample_covariance=False, sigma=1.5)

        sharp = Image.fromarray(np.uint8(img_sharp))
        deblu = Image.fromarray(np.uint8(img_deblu))
        sharp_ts = TF.to_tensor(sharp).unsqueeze(0)
        deblu_ts = TF.to_tensor(deblu).unsqueeze(0)
        sharp_ts, deblu_ts = sharp_ts/255.0, deblu_ts/255.0

        vif_n = piq.vif_p(deblu_ts, sharp_ts)
        vsi_n = piq.vsi(deblu_ts, sharp_ts)
        haar_n = piq.haarpsi(deblu_ts, sharp_ts)

        # For Kohler dataset
        if count_k < 198:
            psnr_k.append(psnr_n)
            ssim_k.append(ssim_n)
            vif_k.append(vif_n)
            vsi_k.append(vsi_n)
            haar_k.append(haar_n)
            count_k += 1
        elif count_k == 198:
            psnr_k.append(psnr_n)
            ssim_k.append(ssim_n)
            vif_k.append(vif_n)
            vsi_k.append(vsi_n)
            haar_k.append(haar_n)
            kernel_p.append(max(psnr_k))
            kernel_s.append(max(ssim_k))
            kernel_f.append(max(vif_k))
            kernel_v.append(max(vsi_k))
            kernel_h.append(max(haar_k))

            psnr_k = []
            ssim_k = []
            vif_k = []
            vsi_k = []
            haar_k = []

            count_k = 1

        # For other datasets
        # PSNR_all.append(psnr_n)
        # SSIM_all.append(ssim_n)
        # VIF_all.append(vif_n)
        # VSI_all.append(vsi_n)
        # Haar_all.append(haar_n)
    else:
        continue

# For other dataset
# PSNR = np.mean(PSNR_all)
# SSIM = np.mean(SSIM_all)
# VIF = np.mean(VIF_all)
# VSI = np.mean(VSI_all)
# Haar = np.mean(Haar_all)
#
# print("ave PSNR = ", PSNR)
# print("ave SSIM = ", SSIM)
# print("ave VIF = ", VIF)
# print("ave VSI = ", VSI)
# print("ave Haar = ", Haar)

# For Kohler dataset
print(kernel_p)
print("ave PSNR = ", np.mean(kernel_p))
print(kernel_s)
print("ave SSIM = ", np.mean(kernel_s))
print(kernel_f)
print("ave VIF = ", np.mean(kernel_f))
print(kernel_v)
print("ave VSI = ", np.mean(kernel_v))
print(kernel_h)
print("ave Haar = ", np.mean(kernel_h))