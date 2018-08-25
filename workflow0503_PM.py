# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 19:04:30 2018

@author: liangjian
"""
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import os
import numpy as np
import cv2
from keras.models import load_model
from keras.models import Model
from keras import backend as keras
from keras import backend as K
from aprilmodel0514B import MCPN


def workflow(file,pathout,img):

    boxlist = []
    boxlist_inner = []

    w,h = img.size
    print (w,h)
    padding = 80
    overlap = 2*padding
    step = 320 - overlap
    padding_w = w + 2*padding
    padding_h = h + 2*padding   
    
    new_im=Image.new('RGB',(padding_w,padding_h))
    new_im.paste(img,(padding,padding,w+padding,h+padding))

 
    smallbox = (padding,padding,320-padding,320-padding)
    finalbox = (padding,padding,padding_w-padding,padding_h-padding)
    
    number = int(np.ceil(padding_w/step)*np.ceil(padding_h/step)) 
    patches = np.ndarray((number,320,320,3),dtype = np.uint8)

    count = 0;
    for i in range (0,padding_w,step):
        for j in range (0,padding_h,step):
            box = (i,j,i+320,j+320)
            box_inner = (i+padding,j+padding,i+320-padding,j+320-padding)
            boxlist.append(box)                  #record box location
            boxlist_inner.append(box_inner)      #record innerbox location 
            imgpatch = new_im.crop(box)          #crop this patch
            imgpatch = img_to_array(imgpatch)    #turn img to array
            patches[count] = imgpatch           #save patch array into patches_ndarray
            print ("patch",count,end='done...')   
            count = count+1
    print ("\n"+"totally ",count,"patches cropped.") 
    imglist = createlblimg(patches,smallbox)
    final_segmentation = recover_imagery(imglist,boxlist_inner,padding_w,padding_h,pathout,finalbox)
    
    outfile = (filename+".tif")
    final_segmentation.save(pathout+"\\"+outfile)  
    print ("segmentation finished")

  
  
def createlblimg(patches,smallbox):

    patches = patches.astype('float32')
    patches /= 255
#    mean = np.mean(patches)  # mean for data centering
#    std = np.std(patches)  # std for data normalization
#    patches -= mean
#    patches /= std
#    mymodel = MCPN()
#    model = mymodel.get_model()
#
#    model.load_weights('50_va_MASH0514.hdf5')
    model = load_model('50_filterfulldense_0517.h5')
    imgs_mask_test = model.predict(patches,batch_size=1, verbose=1)
    np.save('patches.npy', imgs_mask_test)
    K.clear_session()
    print('-'*30)
    print('loading masks...')
    imgs = np.load("patches.npy")
    print (imgs.shape)
    imgs = np.argmax(imgs,axis=3)
    print (imgs.shape)
    imgs = imgs.reshape(imgs.shape[0],imgs.shape[1],imgs.shape[2],1)
    print (imgs.shape)
    imgs[imgs==0] = 0
    imgs[imgs==1] = 255
    print('-'*30)
    imglist =[]
    for i in range(0, len(imgs)):
        imgtest=imgs[i]
        imgtest = array_to_img(imgtest,'channels_last')
        imgtest = imgtest.crop(smallbox)
        imglist.append(imgtest)
    return imglist
    
def recover_imagery(imglist,boxlist_inner,padding_w,padding_h,pathout,finalbox):
    final_segmentation=Image.new('RGB',(padding_w,padding_h))
    print ("recovering the imagery")
    for i in range(0,len(imglist)):
        final_segmentation.paste(imglist[i],boxlist_inner[i])
        print (end='*')   
    final_segmentation =final_segmentation.crop(finalbox)
    print ("Imagery just recovered")
    return final_segmentation
  

path = "5A"
pathout = "50_filterfulldense_0517_80"
filelist=os.listdir(path)
filenum = 0    
for files in filelist:
    filenum =filenum +1
    filename=os.path.splitext(files)[0]
    print ("now processing imagery" + filename)
    midname = filename + ".tiff"
    img = load_img(path+"/"+midname,grayscale= False)
    workflow(filename,pathout,img)    
