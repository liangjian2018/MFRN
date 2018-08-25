# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:39:43 2017

@author: Administrator
"""
#import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import glob
import os
import cv2
import numpy as np 
from keras import backend as K
#from skimage import transform,io
#import matplotlib.pyplot as plt


def rename():
    #path='/home/liangjian/Mylab/Desktop0928spare/DATA'
    path='DATA'
    count=1;
#    path2='E:\\MYpython\\pics\\preview'
    filelist=os.listdir(path)
    for files in filelist:
        print (str(count))
        count=count+1
        Olddir=os.path.join(path,files);
#        if os.path.isdir(Olddir):
#            continue
        filename=os.path.splitext(files)[0]
        print (filename)
        filetype=os.path.splitext(files)[1]
        print (filetype)
        if filename.find('tif')>=0:
            print ("!!"*5)
            Newdir=os.path.join(path,filename.split('tif')[1]+filetype)

            if not os.path.isfile(Newdir):
                os.rename(Olddir,Newdir)
#########################################################################                
             
                
def checksize():
    path='E:\\MYpython\\0914rawjpg'
#    newpath = 'E:\\MYpython\\doc'
    img_type="jpg"
    imgs = glob.glob(path+"/*."+img_type)
    count = 0;
    for imgname in imgs:
        midname = os.path.basename(imgname)
#        filename=os.path.splitext(imgname)[0]#文件名
        filetype=os.path.splitext(imgname)[1]#文件扩展名
        img_temp = Image.open(path+"/"+midname)
        print (midname)
        count=count+1
        Olddir=os.path.join(path,imgname);#原来的文件路径
        (width,height)=img_temp.size
        if (width < 438 or height < 406):
            print (midname+"'s size is wrong!!!")
            Newdir=os.path.join(path,"A"+str(count)+filetype)
            img_temp.close()
            os.rename(Olddir,Newdir);
                     
def checkcompanion():
    path1='problem_pics_of_2_sc'
    path2='problem_pics_of_1_sc'
#    thetype = 'tif'
    filelist=os.listdir(path1)#该文件夹下所有的文件（包括文件夹）
#    filelist2=os.listdir(path2)
    count = 0
    for files in filelist:#遍历所有文件
#        print str(count)
        count=count+1
        filename=os.path.splitext(files)[0]#文件名
#        print filename
        filetype=os.path.splitext(files)[1]#文件扩展名
        Olddir=os.path.join(path1,files);              
        if not os.path.exists(path2+"\\"+filename+filetype):
            print (str(count) + "doesn't exist!")
            Newdir=os.path.join(path1,"Bbbb"+str(count)+filetype)
            os.rename(Olddir,Newdir)
            
def imgresize():
    path='aa'
    pathout='TE4'
    img_type="tif"
    imgs = glob.glob(path+"/*."+img_type)
    count = 0;
    for imgname in imgs:
        midname = os.path.basename(imgname)
        imgt = load_img(path+"/"+midname,target_size=(512,512))
        print (midname)
        count=count+1
        print (str(count))
        imgt.save(pathout+"\\"+midname)
        
def turnformat():
    path='sl'
    pathout='T2'
    #img_type="tif"
    count = 0;
    filelist=os.listdir(path)#该文件夹下所有的文件（包括文件夹）
    for files in filelist:#遍历所有文件
        #midname = os.path.basename(files)
        print (str(count))
        count=count+1
        print (str(count))
        filename=os.path.splitext(files)[0]#文件名
        print (filename)
#        filetype=os.path.splitext(files)[1]#文件扩展名
        outfile = filename + ".tif"
        Image.open(path+"\\"+files).save(pathout+"\\"+outfile)
        
def splitband():
    path='problem_pics_of_2'
    pathout = 'problem_pics_of_2_1b'
    img_type="tif"
    imgs = glob.glob(path+"/*."+img_type)
    for imgname in imgs:
        midname = os.path.basename(imgname)
        img = cv2.imread(imgname)
        imgband1 = img[:,:,2]
        print(imgband1)
        cv2.imwrite(pathout+"/"+midname,imgband1)
#
    
def checkresult():
    path='kk'
    pathout='roundUnet'
    print('-'*30)
    print('load masks...')
    imgs = np.load(path+"\\imgs_mask_test.npy")
#    imgs[imgs > 0.5] = 1
#    imgs[imgs <=0.5]=0
    imgs = np.round(imgs)
    print (len(imgs))
    print('-'*30)
    for i in range(0, len(imgs)):
        print (i )
        imgtest=imgs[i]
        imgtest = array_to_img(imgtest,'channels_last')
        outfile = "filename"+str(i) + ".tif"
        imgtest.save(pathout+"\\"+outfile)

def multytoOne():
    path1 = 'testlabel'
    path2 = '1band-testlabel'

    img_type = 'tif'
    imgs = glob.glob(path1+"/*."+img_type)
    for imgname in imgs:
        midname = os.path.basename(imgname)
        img = cv2.imread(imgname)
        img[img != 255] = 0
        imgband1 = img[:, :, 0]
        imgband2 = img[:, :, 1]
        imgband3 = img[:, :, 2]

        imgband1[imgband1 == 255] = 10
        imgband2[imgband2 == 255] = 50
        imgband3[imgband3 == 255] = 100
        imgband1=imgband1+imgband2+imgband3

        imgband1[imgband1==160] = 1
        imgband1[imgband1==10] = 2
        imgband1[imgband1==60] = 3
        imgband1[imgband1==50] = 4
        imgband1[imgband1==150] = 5
        imgband1[imgband1==100] = 6
        print(imgband1)
#If want one class label,  use it.
#
#        imgband1[imgband1 != 10]=0
#        imgband1[imgband1 == 10]=255
        cv2.imwrite(path2+"/"+midname,imgband1)

def changecolor():
    path1 = 'testlabel'
    path2 = 'npytest'
    img_type = 'tif'
    imgs = glob.glob(path1+"/*."+img_type)
    for imgname in imgs:
        midname = os.path.basename(imgname)
        img = cv2.imread(imgname)
#        img[img == 0] = 0
        img[img == 1] = 255
#        img[img == 3] = 2
#        img[img == 4] = 3
#        img[img == 5] = 4
#        img[img == 6] = 5
#        img[img == 7] = 6
        print(img)
#If want one class label,  use it.
#
#        imgband1[imgband1 != 10]=0
#        imgband1[imgband1 == 10]=255
        cv2.imwrite(path2+"/"+midname,img)
        
    
 
def checkmatch():
    path1='bigpic'
    path2='bigmask'
    path3='bigmatch'
    img_type='tif'
    imgs = glob.glob(path1+"/*."+img_type)
    for imgname in imgs:
        midname = os.path.basename(imgname)
        img_t=load_img(path1+"/"+midname)
        img_l=load_img(path2+"/"+midname)
        xt=img_to_array(img_t)
        xl=img_to_array(img_l)
        xl[xl==255]=254
        xl[xl==0]=255
        xl[xl==254]=0
        print(xl[:,:,0])
        xt[:,:,0]=xt[:,:,0]+xl[:,:,0]
        xt[:,:,0][xt[:,:,0]>255]=254
#        xt[:,:,2]=xl[:,:,0]
        xl = array_to_img(xl)
        outfile = "filename"+"2200" + ".tif"
        xl.save(path3+"\\"+outfile)
        img_temp = array_to_img(xt)
        img_temp.save(path3+"/"+midname)
        
def rotateImages():
    path='problem_pics_of_2_sc'
    pathout='train_label'
    img_type="tif"
    imgs = glob.glob(path+"/*."+img_type)
    count = 0;
    for imgname in imgs:
        midname = os.path.basename(imgname)
        img = load_img(path+"/"+midname,target_size=(320,320))
        imgR = img.transpose(Image.ROTATE_90)
        print (midname)
        count=count+1
        print (str(count))
        img.save(pathout+"\\"+midname)
        imgR.save(pathout+"\\"+"r"+midname)
        
def cutImage(width):
    w=width
    path1='1'
    path2='160finaltest_labels'
#    imgs = glob.glob(path1+"/*."+img_type)
    filelist=os.listdir(path1)
    filenum = 0
    for files in filelist:
        filenum =filenum +1
        filename=os.path.splitext(files)[0]
        print (filename)
        midname = filename + ".tif"
        img = load_img(path1+"/"+midname)
        count = 0;
        for i in range(0,2):
            for j in range(0,2):
                count = count+1
                box = (w*i,w*j,w*(i+1),w*(j+1))
                imgpatch=img.crop(box)
                outname=filename+"P"+"i"+str(i)+"j"+str(j)+".tif"
                imgpatch.save(path2+"/"+outname)
                print (str(count))
        print ("Number",filenum," finished")
    print ("finished")

def cutlabel(width):
    w=width
    path1='R2'
    path2='smalllbl'
#    imgs = glob.glob(path1+"/*."+img_type)
    filelist=os.listdir(path1)
    filenum = 0
    for files in filelist:
        filenum =filenum +1
        filename=os.path.splitext(files)[0]
        print (filename)
        midname = filename + ".tif"
        img = load_img(path1+"/"+midname)
        count = 0;
        for i in range(0,60):
            for j in range(0,47):
                count = count+1
                box = (w*i,w*j,w*(i+1),w*(j+1))
                imgpatch=img.crop(box)
                outname=filename+"i"+str(i)+"j"+str(j)+".tif"
                imgpatch.save(path2+"/"+outname)
                print (str(count))
        print ("Number",filenum," finished")
    print ("finished")

    
def uniformlbl():
    path='U'
    pathout='TRUE'
#    img_type='tif'
    print('-'*30)
    print('load masks...')
    imgs_mask_train = np.load(path+"/imgs_mask_test.npy")
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255
    imgs_mask_train[imgs_mask_train <1] = 0
#    imgs_mask_train[imgs_mask_train ==1] = 0
#    imgs_mask_train[imgs_mask_train ==0.5] = 1
#    imgs_mask_train[imgs_mask_train >=0.99] = 1
    print (len(imgs_mask_train))
    print('-'*30)
    for i in range(0, len(imgs_mask_train)):
        print (i)
        mask=imgs_mask_train[i]
        mask = array_to_img(mask,'channels_last')
        outfile = "filename"+str(i) + ".tif"
        mask.save(pathout+"\\"+outfile)
        
def killsame():
    path1='newtest' #bigdataset
    path2='test' #smalldataset whitelist
#    thetype = 'tif'
    filelist=os.listdir(path1)#该文件夹下所有的文件（包括文件夹）
#    filelist2=os.listdir(path2)
    count = 0
    for files in filelist:#遍历所有文件
#        print str(count)
        count=count+1
        filename=os.path.splitext(files)[0]#文件名
#        print filename
        filetype=os.path.splitext(files)[1]#文件扩展名
        Olddir=os.path.join(path1,files);              
        if os.path.exists(path2+"\\"+filename+filetype):
            print (filename + "already exist!")
            print ("number"+str(count))
            Newdir=os.path.join(path1,"aBadBadBadBad"+str(count)+filetype)
            os.rename(Olddir,Newdir)

def cutImageR():

    w=19350
    path1='H1'
    path2='T1'
#    imgs = glob.glob(path1+"/*."+img_type)
    filelist=os.listdir(path1)
    filenum = 0
    for files in filelist:
        filenum =filenum +1
        filename=os.path.splitext(files)[0]
        print (filename)
        midname = filename + ".tif"
        img = load_img(path1+"/"+midname)
        box = (0,0,w,15040)
        imgpatch=img.crop(box)
        outname=filename+".tif"
        imgpatch.save(path2+"/"+outname)
    print ("Number",filenum," finished")
    print ("finished") 
    
def cutlabelR():

    w=19350
    path1='sl'
    path2='T2'
#    imgs = glob.glob(path1+"/*."+img_type)
    filelist=os.listdir(path1)
    filenum = 0
    for files in filelist:
        filenum =filenum +1
        filename=os.path.splitext(files)[0]
        print (filename)
        midname = filename + ".png"
        img = load_img(path1+"/"+midname)
        box = (0,0,w,15040)
        imgpatch=img.crop(box)
        outname=filename+".tif"
        imgpatch.save(path2+"/"+outname)
    print ("Number",filenum," finished")
    print ("finished")   

def findnodata():
    path='testrecover'
    pathw='1116m'
    img_type = 'tif'
    pathout='whitelist'
    imgwhite = load_img(pathw+"/"+'filename2.tif',target_size=(320,320))
    imgwhite = img_to_array(imgwhite)
    print (imgwhite.sum())
    imgs = glob.glob(path+"/*."+img_type)
    count = 0;
    for imgname in imgs:
        midname = os.path.basename(imgname)
        imgt = load_img(path+"/"+midname,target_size=(320,320))
        imgt = img_to_array(imgt,'channels_last')
        if imgt.sum()==imgwhite.sum():
            print (midname,"is a fucking white pic!!!!!!!!!!!!!!!!!!!!")
            count=count+1
            print (str(count))  
            img_temp = array_to_img(imgt)
            img_temp.save(pathout+"/"+midname)



def checkimgvalue(pathIn):
    pathin = pathIn
#    path2 = 'testbandlbl'
    img_type ='tif'
    imgs = glob.glob(pathin+"/*."+img_type)
    for imgname in imgs:
        midname = os.path.basename(imgname)
        img = cv2.imread(imgname)
        print (midname,img.shape[0],img.shape[1]," violet evergarden"*3)
        print (img)


def WImage():
    path1='thu'
    path2='thu'
#    imgs = glob.glob(path1+"/*."+img_type)
    filelist=os.listdir(path1)
    filenum = 0
    for files in filelist:
        filenum =filenum +1
        filename=os.path.splitext(files)[0]
        print (filename)
        midname = filename + ".tif"
        img = load_img(path1+"/"+midname)
        box = (45,100,205,260)
        imgpatch=img.crop(box)
        outname=filename+"P"+".tif"
        imgpatch.save(path2+"/"+"filename"+outname)
        print ("Number",filenum," finished")
    print ("finished")

#WImage();

def checkimg():
    path = '2'
    filelist=os.listdir(path)
    filenum = 0
    for files in filelist:
        filenum =filenum +1
        filename=os.path.splitext(files)[0]
        print (filename)
        midname = filename + ".tif"
        img = load_img(path+"/"+midname)
        img = img_to_array(img)
        
        print (img)
    print ("done")
    

#checkimg()


def cutImageAuto(width,pathin,pathout):
    w=width
    path1= pathin
    path2= pathout
#    imgs = glob.glob(path1+"/*."+img_type)
    filelist=os.listdir(path1)
    filenum = 0
    for files in filelist:
        filenum =filenum +1
        filename=os.path.splitext(files)[0]
        print (filename)
        midname = filename + ".tif"
        img = load_img(path1+"/"+midname)
        imgsizei = img.size[0]
        imgsizej = img.size[1]

        ic = imgsizei/w
        jc = imgsizej/w
        ri = np.round(ic)
        rj = np.round(jc)
        if ri <= imgsizei:
            inum = int(np.ceil(imgsizei/w))
            print ("ceil",ri,ic,inum)
        else:
            inum = int(np.floor(imgsizei/w))
            print ("floor",ri,ic,inum)
        if rj <=imgsizej:
            jnum = int(np.ceil(imgsizej/w))
            print ("ceil",rj,jc,jnum)
        else:
            jnum = int(np.floor(imgsizej/w))
            print ("floor",rj,jc,jnum)

        inum = int(np.ceil(imgsizei/320))
        jnum = int(np.ceil(imgsizej/320))
        print (inum,jnum)
        count = 0;
        for i in range(0,inum):
            for j in range(0,jnum):
                count = count+1
                box = (w*i,w*j,w*(i+1),w*(j+1))
                imgpatch=img.crop(box)
                outname=filename+"P"+"i"+str(i)+"j"+str(j)+".tif"
                imgpatch.save(path2+"/"+outname)
                print (str(count))
        print ("Number",filenum," finished")
    print ("finished")


def transfer1and0():
    path = '4band1'
    path2 = 'four'
    filelist=os.listdir(path)
    filenum = 0
    for files in filelist:
        filenum =filenum +1
        filename=os.path.splitext(files)[0]
        print (filename)
        midname = filename + ".tif"
        img = load_img(path+"/"+midname)
        img = img_to_array(img)
        img /= 255
        img[img==255] = 50
        img[img==0]=255
        img[img==50]=0
        img = array_to_img(img)
        outname =filename + ".tif"
        img.save(path2+"/"+outname)
    print ("done")
    
def createIDs():
    IDs = []
    path = 'test'
    filelist = os.listdir(path)
    filenum = 0
    for files in filelist:
        filenum = filenum + 1
        filename = os.path.splitext(files)[0]
        IDs.append(filename)
        print (files,end='>>>>]]]}}}}}....>')
    print (len(IDs))
#    print (IDs)
    np.save("IDlist_test.npy",IDs)

def slidecut(width,overlap):
    w=width
    o = overlap
    path1 = 'problem_pics_of_2_1b_label'
    path2 = 'problem_pics_of_2_sc'
#    imgs = glob.glob(path1+"/*."+img_type)
    filelist=os.listdir(path1)
    filenum = 0
    for files in filelist:
        filenum =filenum +1
        filename=os.path.splitext(files)[0]
        print (filename)
        midname = filename + ".tif"
        img = load_img(path1+"/"+midname)
        imgw = img.size[0]
        imgh = img.size[1]
        print (imgh, imgw)
        count = 0;
        for i in range(0,(imgw),(w-o)):
            for j in range(0,(imgh),(w-o)):
                count = count+1
                box = (i,j,(i+w),(j+w))
                imgpatch=img.crop(box)
                outname=filename+"P"+"i"+str(i)+"j"+str(j)+".tif"
                imgpatch.save(path2+"/"+outname)
                print (str(count),"i:",i,"j",j)
        print ("Number",filenum," finished")
    print ("finished")
    

#transfer1and0();
#rename();
#checksize();
#checkcompanion();
#imgresize();
#reversecolor();
#turnformat();
#splitband();
#checkresult()
#checkmatch();
#rotateImages();
#cutImage(160);
#cutImage(320)
#cutlabel(320)
#uniformlbl();
#killsame()
#cutImageR()
#cutlabelR()
#findnodata()
#multytoOne();
#slidecut(320, 100)
changecolor();
#checkimgvalue('transL');  
#cutImageAuto(320,'4','patchlabels');



#x = np.load("npydata/"+"imgs_test"+'.npy')
#y= load_img("smalls/smalltrain" + "/" + 'top_mosaic_09cm_area11Pi0j0' +'.tif',grayscale = False)
#y = img_to_array(y)
#print (y.shape)
#createIDs();

# f = np.load("IDlist.npy")
# for i in range(len(f)):
#
#     print (i,"^^^^^", f[i])
#     i +=1
#
