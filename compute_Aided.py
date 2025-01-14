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


# def psnr(img1, img2):
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 255.0
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


deblu_root = './test_aided'  #
sharp_root = './dataset/AidedDeblur/test/'  # _

f_test = open("./dataset/AidedDeblur/test_instance_names.txt", "r")
test_data = f_test.readlines()
sharp_list = [line.rstrip() for line in test_data]
f_test.close()
deblu_list = os.listdir(deblu_root)
sharp_list = sorted(sharp_list, key=str.lower)

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

sample_img_names = set(["010221", "024071", "033451", "051271", "060201",
                         "070041", "090541", "100841", "101031", "113201"])

for n, item in enumerate(sharp_list):
    if True:

        name_sharp = item[-6:]
        name_deblu = name_sharp + '.png'

        path_deblu = os.path.join(deblu_root, name_deblu)
        path_sharp = item + '_ref.png'

        img_deblu = cv2.imread(path_deblu, cv2.IMREAD_COLOR).astype(np.float)
        img_sharp = cv2.imread(path_sharp, cv2.IMREAD_COLOR).astype(np.float)

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

        if item[-3:] == '001' or name_sharp in sample_img_names:
            print("Test Image {}, PSNR = {}, SSIM = {}".format(name_sharp, psnr_n, ssim_n))

        sharp = Image.fromarray(np.uint8(img_sharp))
        deblu = Image.fromarray(np.uint8(img_deblu))
        sharp_ts = TF.to_tensor(sharp).unsqueeze(0)
        deblu_ts = TF.to_tensor(deblu).unsqueeze(0)
        sharp_ts, deblu_ts = sharp_ts/255.0, deblu_ts/255.0

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