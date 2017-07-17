from PIL import Image
import PIL.ImageOps 
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

class TrainCharacterExtractor():
    grayImg = 0
    grayArr = 0
    cell_hw = [0,0]
    font_offsets_y = []
    font_offsets_x = []
    isFigure = False

    def __init__(self):
        print ("TrainCharacterExtractor __init__")

    def Reset(self):
        self.font_offsets_y.clear()
        self.font_offsets_x.clear()

    def Read(self, path):
        self.Reset()
        img = Image.open(path)
        grayImg = img.convert('L')
        self.grayImg = PIL.ImageOps.invert(grayImg)
        self.grayArr = np.asarray(self.grayImg)

    def GetRotationAngle(self):    
        image = self.grayImg
        maxAngle = 0
        maxMean = 0
        ratate_step = 0.02
        angle_min = -1
        angle_max = 1
        angles = []
        for i in range(angle_min,angle_max):
            angle = i*ratate_step            
            image_rot = image.rotate(angle)
            src = np.asarray(image_rot)
            mean_1 = np.mean(np.std(src, axis=1))
            mean_2 = np.mean(np.std(src, axis=0))
            elsilon = 0.1
            mean = 1/(mean_1+elsilon) * 1/(mean_2+elsilon)
            if mean > maxMean :
                print ('rotate angle',angle,'mean',mean)
                maxMean  = mean
                maxAngle = angle

        print ('max Angle',maxAngle,'maxMean',maxMean)
        if maxAngle!=0:
            self.grayImg = image.rotate(maxAngle)
            self.grayArr = np.asarray(self.grayImg)

        if self.isFigure:
            plt.figure(0)            
            plt.subplot(1,2,1)
            plt.title('original')
            plt.imshow(image, cmap = plt.get_cmap('gray'))

            plt.subplot(1,2,2)    
            img_rotate = image.rotate(maxAngle)
            plt.title('rotate'+str(maxAngle))
            plt.imshow(img_rotate, cmap = plt.get_cmap('gray'))
        return maxAngle

    def GetLetterSize(self, axis):
        list = 0
        if axis==1: list = self.font_offsets_y
        else: list = self.font_offsets_x

        src = self.grayArr    
        lpf = [0,0.1,0.2,0.3,0.4,0.3,0.2,0.1,0]
        sum_col = np.sum(src, axis=axis)        
        sum_col = np.convolve(sum_col,lpf,'valid')
        under = np.zeros_like(sum_col)
        under[:-1] = sum_col[1:]
        deriv = under-sum_col    
        sum_col_mean = np.mean(sum_col)
        print ('sum_col mean',np.mean(sum_col))
        i0 = 0
        pillas = []        
        candidate2Index = 0                
                        
        for i in range(len(deriv)-1):            
            if deriv[i]== 0 and deriv[i+1]==0:
                continue
            elif sum_col[i]<sum_col_mean/1 and (deriv[i]<= 0 and deriv[i+1]>0):
                pilla_w = i-i0                
                pillas.append(pilla_w)
                list.append(i)
                print ('axis',axis,candidate2Index, i0,'~', i,'w:', pilla_w)
                i0 = i
                candidate2Index+=1
            if i== len(deriv)-1-1:
                pilla_w = i-i0
                pillas.append(pilla_w)
                list.append(i)
        
        print ('size ',self.grayArr.shape)
        print ('pillar_count ',len(pillas))
        pilla_arr = np.array(pillas)
        pilla_std = np.std(pilla_arr)
        pilla_mean = np.mean(pilla_arr)
                
        letterH = (int)(pilla_mean)
        print ('cell_h',letterH)
        self.cell_hw[axis-1] = letterH
        
        if self.isFigure:
            plt.figure(1)
            plt.plot(sum_col)    
            plt.grid(which='both', axis='both')
            plt.show()     
    
    def GetGrid(self):  
        self.GetLetterSize(0)
        self.GetLetterSize(1)

