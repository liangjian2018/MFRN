# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:00:19 2018

@author: LiangJian
"""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2

class MyDataGenerator(object):
    
    def __init__(self, dim_x=320,dim_y = 320, batch_size = 4, shuffle = True):
        'initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __get_exploration_order(self,list_IDs):
        'Generates order of exploration'
        #Find exploration order
        indexes =np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)
            
        return indexes
        
    def one_hot_it(self,y):
        sx = np.zeros([320,320,2])
        for i in range(320):
            for j in range(320):
                sx[i,j,y[i][j]]=1
        return sx 

       
    def sparsify(y):
        'Returns labels in binary NumPy array'
        n_classes = 2 # Enter number of classes
        return np.array([[1 if y[i] == j else 0 for j in range(n_classes)] for i in range(y.shape[0])])
#Please note that this sparsify function is adapted for labels starting at 0. If your labels start at 1, 
#simply change the expression y[i] == j to y[i] == j+1 in the piece of code above.     
    
#    def __data_generation(self, labels, list_IDs_temp):
    def __data_generation(self, list_IDs_temp):
        'Generates data of batch_size samples' # X: (n_samples, v_size, v_size, n_channells)
        # Initialization 
        X = np.empty((self.batch_size, self.dim_x,self.dim_y,3), dtype=np.uint8)
        y = np.empty((self.batch_size, self.dim_x,self.dim_y,2), dtype=np.uint8)
        
        #Generate data
        for i, ID in enumerate(list_IDs_temp):          
            img = load_img("../MashData" + "/" + "validate" + "/" + ID +'.tif',grayscale = False)
#            label = load_img("dataset" + "/" + "test_L" + "/" + ID + '.tif',grayscale = True) 
            imgname =  "../MashData" + "/" + "validate_label" + "/" + ID + ".tif"
            label = cv2.imread(imgname)
            img = img_to_array(img) 

            X[i] = img
            y[i] = self.one_hot_it(label) 
                      

        X = X.astype('float32')
        X /= 255
#        mean = np.mean(X)
#        std = np.std(X)
#        X -=mean
#        X /= std
        return X, y
        
#        return X, sparsify(y)



        
#    def generate(self,labels,list_IDs):
    def generate(self,list_IDs):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)
            
            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                #Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                                 
                # Generate data
#                X, y = self.__data_generation(labels,list_IDs_temp)
                X, y = self.__data_generation(list_IDs_temp)                
                yield X, y
               
#params = {'dim_x': 32,
#          'dim_y': 32,
#          'dim_z': 32,
#          'batch_size': 32,
#          'shuffle': True}
#
## Datasets
#partition = # IDs
#labels = # Labels
                
# Generators
#training_generator = DataGenerator(**params).generate(labels, partition['train'])
#validation_generator = DataGenerator(**params).generate(labels, partition['validation'])
                
                
#model.fit_generator(generator = training_generator,
#                    steps_per_epoch = len(partition['train'])//batch_size,
#                    validation_data = validation_generator,
#                    validation_steps = len(partition['validation'])//batch_size)             
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                