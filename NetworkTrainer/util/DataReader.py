from PIL import Image
import numpy as np
from scipy.misc import toimage
import glob
from sklearn.preprocessing import StandardScaler

class DataReader():
    folder = '../data'        
    pathTrain = folder +'/character/*.png'
    pathTest = folder +'/character0/*.png'
    ext = '.png'
    h = 16
    w = 12
            
    def __init__(self):
        print ('DataReader.py __init__') 
    
    def ReadFolder(self, path):
        
        list_path = glob.glob(path)                                
        count = len(list_path)
        print ('ReadFolder() count', len(list_path))
        list_image = []
                
        for n in range(count):            
            list_image.append(Image.open(list_path[n]))            
    
        return list_image

    def ResizeImages(self, list_image):
        count = len(list_image)
        setIn = np.zeros(shape=(count,self.h,self.w), dtype=np.float32)                        
        print ('Resize Image',self.h,self.w)

        for n in range(count):
            image = list_image[n]
            image_resize = image.resize((self.w,self.h), Image.ANTIALIAS)
            arr = np.asarray(image_resize)            
            setIn[n] = arr            

        return setIn

    def GetData(self):
        
        list_image_train = self.ReadFolder(self.pathTrain)
        list_image_test = self.ReadFolder(self.pathTest)

        trainIn = self.ResizeImages(list_image_train)
        trainOut = np.arange(len(list_image_train))

        testIn = self.ResizeImages(list_image_test)
        testOut = np.arange(len(list_image_test))

        return [trainIn,trainOut,testIn,testOut]

    def Augment(self, src0, src1, aug):
        if aug ==1: return [src0, src1 ]
        if aug > 4: aug=4
        n = src0.shape[0]
        h = src0.shape[1]
        w = src0.shape[2]
        c = src0.shape[3]
        setIn = np.zeros(shape=(n*aug,h,w,c), dtype=np.float32)
        setOut = np.zeros(shape=(n*aug,h,w), dtype=np.float32) 
        setIn[:n,:] = src0
        setOut[:n,:] = src1
        for i in range(0, n):            
            print ('augment ',i,'/',n)
            setIn_one = src0[i,:]
            setOut_one = src1[i,:]                
            if aug > 1:
                n1 = i+n
                setIn[n1,:]= np.fliplr(setIn_one)
                setOut[n1,:]= np.fliplr(setOut_one) 
            if aug > 2:
                n2 = i+n*2              
                setIn[n2,:]= np.flipud(setIn_one)
                setOut[n2,:]= np.flipud(setOut_one)
            if aug > 3:
                n3 = i+n*2              
                setIn[n3,:]= np.flipud(setIn[n1,:])
                setOut[n3,:]= np.flipud(setOut[n1,:]) 
        return [setIn, setOut]

    def GetDataAug(self, n, aug):        
        setIn, setOut = self.GetData(n)           
        return self.Augment(setIn,setOut, aug) 

    def GetDataTrainTest(self, count, aug, ensemble):        
        setIn, setOut = self.GetData(count)        
        count = setIn.shape[3]
        
        count_train = count - ensemble
        train0 = setIn[:,:,:,0:count_train]
        train1 = setOut
        test0 = setIn[:,:,:,count_train:]
        test1 = setOut

        train0,train1 = self.Augment(train0,train1, aug)
        print ('train_in',train0.shape)
        print ('train_out',train1.shape)
        print ('test_in',test0.shape)
        print ('test_out',test1.shape)        
        return train0,train1,test0,test1
    
    def Append_ensemble(self,data_in,data_out, ensemble):   
        data_in_a = data_in[:,:,:,0:ensemble]
        data_out_a = data_out
        for i in range(data_in.shape[3] - ensemble):
            data_in_b = data_in[:,:,:,1+i:1+i+ensemble]
            data_in_a = np.append(data_in_a, data_in_b, axis=0)
            data_out_a = np.append(data_out_a, data_out, axis=0)
        return data_in_a,data_out_a

    def GetData3(self, count, aug, ensemble):        
        setIn, setOut = self.GetData(count)        
        count = setIn.shape[3]
        
        offset0 = count - ensemble * 2
        offset1 = count - ensemble * 1

        in_train = setIn[:,:,:,0:offset0]
        in_val = setIn[:,:,:,offset0:offset1]
        in_test = setIn[:,:,:,offset1:]
                
        out_train = out_val = out_test = setOut        
        in_train,out_train = self.Augment(in_train,out_train, aug)

        half_offset = (int)(offset0/2)
        in_train_0 = in_train[:,:,:,0:half_offset]
        in_train_1 = in_train[:,:,:,half_offset:]
        in_train = np.append(in_train_0,in_train_1,axis=0)
        out_train = np.append(out_train,out_train,axis=0)
        return in_train,out_train,in_val,out_val, in_test, out_test

    def GetDataS(self, count, aug, ensemble):        
        setIn, setOut = self.GetData(count)        
        count = setIn.shape[3]
        
        offset0 = count - ensemble * 3
        offset1 = count - ensemble * 1

        in_train = setIn[:,:,:,0:offset0]
        in_val = setIn[:,:,:,offset0:offset1]
        in_test = setIn[:,:,:,offset1:]
                
        out_train = out_val = out_test = setOut
        half_offset = (int)(offset0/2)
        in_train_0 = in_train[:,:,:,0:half_offset]
        in_train_1 = in_train[:,:,:,half_offset:]
        in_train = np.append(in_train_0,in_train_1,axis=0)
        out_train = np.append(out_train,out_train,axis=0)

        in_val,out_val = self.Append_ensemble(in_val,out_val,ensemble)
        in_test,out_test = self.Append_ensemble(in_test,out_test,ensemble)

        in_train,out_train = self.Augment(in_train,out_train, aug)
        return in_train,out_train,in_val,out_val, in_test, out_test

    def GetNextBatch(self):

        return 0
    
    def SaveAsImage(self, src, filePath, count = 1):
        ext = '.png'        
        print ('SaveAsImage','count:', count, src.shape, filePath, ext)
        src = np.reshape(src, [count,self.dstH,self.w])
        for i in range(0, count):
            
            img = toimage(src[i,:])
            fileName =filePath+ str(i) +ext
            img.save( fileName )    
            
    def SaveAsImageByChannel(self, src, filePath, count = 1):
        
        print ('SaveAsImageByChannel','count:', count, src.shape, filePath,ext   )
        
        for i in range(0, count):          
            for c in range(0, src.shape[3]):
                inChannel = np.abs(src[i,:,:,c])
                
                #img = toimage(inChannel/np.max(inChannel)*255)                                
                img = toimage(inChannel*255)
                fileName =filePath+ str(i)+'_'+ str(c)+'_tri'+self.ext  
                img.save( fileName ) 
    
    def SaveImage(self, src, filePath):                
        print ('SaveTensorImage',  src.shape, filePath)
          
        for i in range(0, src.shape[0]):  
            img = toimage( src[i,:])
            fileName =filePath+'_t_'+ str(i)+self.ext 
            img.save( fileName )
             
    def SaveImageNormalize(self, src, filePath):
        print ('SaveTensorImage',  src.shape, filePath)
          
        for i in range(0, src.shape[0]):  
            data = src[i,:]
            src_shape = data.shape
            data_2d = np.reshape(data, (-1, data.shape[np.ndim(data)-1]) )            
            data_normal = data_2d/np.max(data_2d,0)
            data_normal_2d = np.reshape(data_2d, src_shape)
            img = toimage(data_normal_2d)
            fileName =filePath+'_t_'+ str(i)+self.ext  
            img.save( fileName ) 
                     