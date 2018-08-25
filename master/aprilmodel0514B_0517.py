import numpy as np
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,Conv2DTranspose,concatenate,Activation,BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping
from keras import backend as keras
#from data import dataProcess
from keras.optimizers import Adam
from scipy.misc import imsave
import time
from keras import backend as K
from sklearn.metrics import recall_score
from my_classes_fortest import MyDataGenerator
from my_classes_fortrain import MytrainDataGenerator
from lib_0514 import denseblockC,denseblockup,dense_block,transition_up_block,transition_down_block,filter_block

params = {'dim_x': 320,
          'dim_y': 320,
          'batch_size': 4,
          'shuffle': True}
          
partition = np.load("../IDlist/"+'IDlist_validate.npy') # IDs
trainpartition = np.load("../IDlist/"+'IDlist_train_90.npy')

class MCPN(object):
    def __init__(self, img_rows = 320, img_cols = 320, growth_rate = 12):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.growth_rate = growth_rate
        
#    def load_data(self):
#        mydata = dataProcess(self.img_rows, self.img_cols)
#        imgs_train, imgs_mask_train = mydata.load_train_data()
#        imgs_test,imgs_mask_test = mydata.load_test_data()
#        return imgs_train, imgs_mask_train, imgs_test, imgs_mask_test
        
    def get_model(self):
        inputs = Input((self.img_rows, self.img_cols,3))
        k = self.growth_rate
        ch=0.5
        reduction = 0.5
#        reduction2 = 1
        
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1     
        conv0 = Conv2D(48, (3, 3), use_bias=False,padding='same', name='conv0')(inputs)
        conv0 = BatchNormalization(axis=bn_axis, scale=False, name ='conv0_bn')(conv0)
        conv0 = Activation('relu', name ='conv0_relu')(conv0)
        
        # downsampling path
        conv1d = dense_block(conv0,2, k,1,'db1')
        conv1t = transition_down_block(conv1d,1,'tb1')        
        conv2d = dense_block(conv1t,4, k,1,'db2')
        conv2t = transition_down_block(conv2d,1,'tb2')    
        conv3d = dense_block(conv2t,4, k,1,'db3')
        conv3t = transition_down_block(conv3d,1,'tb3')
        conv4d = dense_block(conv3t,4, k,1,'db4')
        conv4t = transition_down_block(conv4d,1,'tb4')
        conv5d = dense_block(conv4t,4, k,1,'db5')
        conv5t = transition_down_block(conv5d,1,'tb5')   
        
     
        #bottom block
        conv6 = dense_block(conv5t,4, k,1,'db6')
        
 #upsampling path    
         
        up7 = transition_up_block(conv6,reduction,'tb6')
#        _,r,c,ch = up7.get_shape().as_list()
##        ch = int(0.5*ch)
        filter_conv5d = filter_block(conv5d,ch,'filter1')
        up7 = concatenate([up7, filter_conv5d], axis=-1)
        conv7 = dense_block(up7,4, k,1,'db7') 
        
        
        up8 = transition_up_block(conv7,reduction,'tb7')
#        _,r,c,ch = up8.get_shape().as_list()
##        ch = int(0.5*ch)
        filter_conv4d = filter_block(conv4d,ch,'filter2')
        up8 = concatenate([up8, filter_conv4d], axis=-1)
        conv8 = dense_block(up8,4, k,1,'db8')    
        
        up9 = transition_up_block(conv8,reduction,'tb8')
#        _,r,c,ch = up9.get_shape().as_list()
##        ch = int(0.5*ch)
        filter_conv3d = filter_block(conv3d,ch,'filter3')
        up9 = concatenate([up9,  filter_conv3d], axis=-1)
        conv9 = dense_block(up9,4, k,1,'db9')    
        
        up10 = transition_up_block(conv9,reduction,'tb9')
#        _,r,c,ch = up10.get_shape().as_list()
##        ch = int(0.5*ch)
        filter_conv2d = filter_block(conv2d,ch,'filter4')
        up10 = concatenate([up10, filter_conv2d], axis=-1)
        conv10 = dense_block(up10,4, k,1,'db10')  
        
        up11 = transition_up_block(conv10,reduction,'tb10')
#        _,r,c,ch = up11.get_shape().as_list()
##        ch = int(0.5*ch)
        filter_conv1d = filter_block(conv1d,ch,'filter5')
        up11 = concatenate([up11, filter_conv1d], axis=-1)
        conv11 = dense_block(up11,2, k,1,'db11') 

        conv11 = Conv2D(2, (1, 1), activation='softmax')(conv11)
            
        model = Model(inputs=[inputs], outputs=[conv11])
        model.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

        return model

    def checkmodel(self):
        model = self.get_model()
        model.summary()

    def train(self):

        model = self.get_model()
        print("got model")
        start_time = time.time()
  
#        EarlyS = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='max')
        model_checkpoint = ModelCheckpoint('50_va_MASH0517.hdf5', monitor='val_acc',verbose=1, save_best_only=True)
        model_checkpoint1 = ModelCheckpoint('50_vl_MASH0517.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
        model_checkpoint2 = ModelCheckpoint('50_loss_MASH0517.hdf5', monitor='loss',verbose=1, save_best_only=True)
        model_checkpoint3 = ModelCheckpoint('50_acc_MASH0517.hdf5', monitor='acc',verbose=1, save_best_only=True)
        model.summary()
        print('Fitting model...')
        
        training_generator = MytrainDataGenerator(**params).generate(trainpartition)
        validation_generator = MyDataGenerator(**params).generate(partition)
        
        history = model.fit_generator(generator = training_generator, steps_per_epoch = len(trainpartition)//4,epochs = 50,verbose=1,
                                      validation_data = validation_generator,validation_steps = len(partition)//4, shuffle = True, callbacks=[model_checkpoint,model_checkpoint1,model_checkpoint2,model_checkpoint3])
  
        model.save('50_filterfulldense_0517.h5')
        print(time.time()-start_time, "is the time of training model")

  
        import matplotlib.pyplot as plt
		# list all data in history
        print(history.history.keys())
        print('val_acc:',history.history['val_acc'])
        print('val_loss:',history.history['val_loss'])
        print('loss',history.history['loss'])
        print('acc',history.history['acc'])
		# summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
		# summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
  


if __name__ == '__main__':
	mymodel = MCPN()
	mymodel.checkmodel()
#	mymodel.train()









