import numpy
import math
import cv2
import os
import numpy as np
import argparse


def psnr(img1, img2):
    mse = numpy.mean( (img1/255. - img2/255.) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def test_psnr(path):
	result = []
	for roots, dirs, files in os.walk(path):
		for file in sorted(files):
			if "fake_B" in file:
				fake_img = cv2.imread(path+file)
				real_img = cv2.imread(path+file.replace("fake", "real"))
				num = psnr(fake_img, real_img)
				print(file, num)
				result.append(num)
	mean_num = np.mean(result)
	return mean_num


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, default='./results/Unet_unpair_add_l1_10_compete_LOL/test_100/images/', help='test path')
	opt = parser.parse_args()
	result = []
	path = opt.path
	for roots, dirs, files in os.walk(path):
		for file in sorted(files):
			if "fake_B" in file:
				fake_img = cv2.imread(path+file)
				real_img = cv2.imread(path+file.replace("fake", "real"))
				num = psnr(fake_img, real_img)
				print(file, num)
				result.append(num)

	mean_num = np.mean(result)
	print("num: " + str(mean_num))
