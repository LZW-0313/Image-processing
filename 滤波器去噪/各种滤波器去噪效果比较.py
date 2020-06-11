# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:15:12 2020

@author: lx
"""





import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
os.getcwd()
os.chdir('C:\\Users\\lx\\Desktop')       ##更改路径至桌面
img = cv2.imread("test2.png")#读取目标图片

########     四个不同的滤波器    #########


# 均值滤波
img_mean = cv2.blur(img, (5,5))

# 高斯滤波
img_Guassian = cv2.GaussianBlur(img,(5,5),0)

# 中值滤波
img_median = cv2.medianBlur(img, 5)


# 展示不同的图片
titles = ['Noise-Img','mean', 'Gaussian', 'median']
imgs = [img, img_mean, img_Guassian, img_median]

for i in range(4):
    plt.subplot(2,3,i+1)#注意，这和matlab中类似，没有0，数组下标从1开始
    plt.imshow(imgs[i])
    plt.title(titles[i])
plt.show()
