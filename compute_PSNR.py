import numpy as np
import math
import cv2
import os
import matplotlib.pyplot as plt
import torch
import piq
from PIL import Image
from skimage.measure import compare_ssim as ssim


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# deblu_root = './test'  # _all_deblurred'
# sharp_root = './datasets/Kohler_multi4'  # _all'

deblu_root = './res_GOPRO_NB'
sharp_root = './datasets/test_poisson'

deblu_list = os.listdir(deblu_root)
sharp_list = os.listdir(sharp_root)
sharp_list = sorted(sharp_list, key=str.lower)
print(sharp_list)

num_imgs = len(deblu_list)
PSNR_all = []
SSIM_all = []
count_k = 1
psnr_k = []
ssim_k = []
kernel_p = []
kernel_s = []
VIF_all = []
VSI_all = []
Haar_all = []

for n, item in enumerate(sharp_list):
    if not item.startswith('.'):

        name_sharp = sharp_list[n]
        name_deblu = name_sharp

        path_deblu = os.path.join(deblu_root, name_deblu)
        path_sharp = os.path.join(sharp_root, name_sharp)

        img_deblu = cv2.imread(path_deblu, cv2.IMREAD_COLOR).astype(np.float)
        img_sharp = cv2.imread(path_sharp, cv2.IMREAD_COLOR).astype(np.float)

        _, _, img_sharp = np.split(img_sharp, 3, axis=1)

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

        psnr_n = psnr(img_deblu, img_sharp)
        ssim_n = ssim(img_deblu / 255, img_sharp / 255, gaussian_weights=True, multichannel=True,
                      use_sample_covariance=False, sigma=1.5)
        if name_sharp[-3:] == "001":
            print('PSNR=%f, SSMI=%f', (psnr_n, ssim_n))

        sharp = Image.fromarray(np.uint8(img_sharp))
        deblu = Image.fromarray(np.uint8(img_deblu))
        sharp_ts = TF.to_tensor(sharp/255.0).unsqueeze(0)
        deblu_ts = TF.to_tensor(deblu/255.0).unsqueeze(0)

        vif_n = piq.vif_p(deblu_ts, sharp_ts)
        vsi_n = piq.vsi(deblu_ts, sharp_ts)
        haar_n = piq.haarpsi(deblu_ts, sharp_ts)

        #
        # if count_k < 198:
        #     psnr_k.append(psnr_n)
        #     ssim_k.append(ssim_n)
        #     count_k += 1
        # elif count_k == 198:
        #     psnr_k.append(psnr_n)
        #     ssim_k.append(ssim_n)
        #     kernel_p.append(max(psnr_k))
        #     kernel_s.append(max(ssim_k))
        #     psnr_k = []
        #     ssim_k = []
        #     count_k = 1

        PSNR_all.append(psnr_n)
        SSIM_all.append(ssim_n)
        VIF_all.append(vif_n)
        VSI_all.append(vsi_n)
        Haar_all.append(haar_n)
    else:
        continue

PSNR = np.mean(PSNR_all)
SSIM = np.mean(SSIM_all)
VIF = np.mean(VIF_all)
VSI = np.mean(VSI_all)
Haar = np.mean(Haar_all)
# print(PSNR_all)
# print(SSIM_all)
print("ave PSNR = ", PSNR)
print("ave SSIM = ", SSIM)
print("ave VIF = ", VIF)
print("ave VSI = ", VSI)
print("ave Haar = ", Haar)
# print('For Kohler dataset')
#
# print(np.mean(kernel_p))
# print(np.mean(kernel_s))