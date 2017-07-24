from PIL import Image
import PIL.ImageOps 
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os

class TrainCharacterExtractor():
    grayImg = 0
    grayArr = 0
    gridArr = 0
    cell_hw = [0,0]
    font_offsets_y = []
    font_offsets_x = []
    isFigure = False
    index_figure = 0
    list_character = []

    def __init__(self):
        print ("TrainCharacterExtractor __init__")

    def Reset(self):
        self.font_offsets_y.clear()
        self.font_offsets_x.clear()
        self.list_character.clear()

    def Binarize(self,src):
        shp = src.shape
        delete_count = 0
        for y in range(shp[0]):
            for x in range(shp[1]):
                v = 0
                if src[y,x]>128: v = 255                    
                src[y,x] = v

    def Read(self, path):
        self.Reset()
        img = Image.open(path)
        grayImg = img.convert('L')
        print ('Read invert', path)
        pad_w = grayImg.width/20
        pad_h = grayImg.height/30
        
        grayImg_cut = grayImg.crop((pad_w,pad_h, grayImg.width - pad_w,grayImg.height- pad_h))
        print ('cut padding', grayImg.width ,grayImg.height,'->', grayImg_cut.width ,grayImg_cut.height)
        self.grayImg = PIL.ImageOps.invert(grayImg_cut)        
        self.grayArr = np.asarray(self.grayImg).copy()              
        print ('Read Binarize')
        self.Binarize(self.grayArr) 

    def ClearByThreshold(self):
        shp = self.grayArr.shape
        delete_count = 0
        for y in range(shp[0]):
            for x in range(shp[1]):
                if self.grayArr[y,x]< 1:
                    self.grayArr[y,x] = 0;
                    delete_count+=1
        print ('ClearByThreshold', shp,'delete_count',delete_count)

    def GetRotationAngle(self):    
        image = self.grayImg
        maxAngle = 0
        maxMean = 0
        angle = 1.5
        ratate_step = 0.05
        angles = []
        for angle in np.arange(-angle, angle, ratate_step):
        #for i in range(angle_min,angle_max):
            #angle = i*ratate_step            
            image_rot = image.rotate(angle)
            
            #dx = ndimage.sobel(image_rot, 0)
            #dy = ndimage.sobel(image_rot, 1)
            #mag = np.hypot(dx, dy)  # magnitude
            #image_rot *= 255.0 / np.max(mag)
            src = np.asarray(image_rot)
            mean_1 = np.mean(np.std(src, axis=1))
            mean_2 = np.mean(np.std(src, axis=0))
            elsilon = 0.0001
            mean = (1/(mean_1+elsilon)) * (1/(mean_2+elsilon))
            mean = 1/ (mean_1+mean_2+elsilon )
            if mean > maxMean :
                print ('rotate angle',angle,mean_1,mean_2,'mean',mean)
                maxMean  = mean
                maxAngle = angle

        print ('max Angle',maxAngle,'maxMean',maxMean)
        if maxAngle!=0:
            self.grayImg = image.rotate(maxAngle, resample=Image.BICUBIC)
            self.grayArr = np.asarray(self.grayImg).copy()
            #self.Binarize(self.grayArr) # No!
            #self.ClearByThreshold()

        self.gridArr = np.asarray(self.grayImg).copy()

        if self.isFigure:
            plt.figure(self.index_figure)            
            self.index_figure+=1
            plt.subplot(1,2,1)
            plt.title('original')
            plt.imshow(image, cmap = plt.get_cmap('gray'))

            plt.subplot(1,2,2)    
            img_rotate = image.rotate(maxAngle)
            plt.title('rotate'+str(maxAngle))
            plt.imshow(img_rotate, cmap = plt.get_cmap('gray'))
            plt.get_current_fig_manager().window.setGeometry(450,50,900,400)
        return maxAngle

    def GetLetterSize(self, axis):
        list = 0
        if axis==1: list = self.font_offsets_y
        else: list = self.font_offsets_x

        src = self.grayArr    
        lpf = [0,0.1,0.2,0.3,0.4,0.3,0.2,0.1,0]
        sum_col = np.sum(src, axis=axis)        
        sum_col = np.convolve(sum_col,lpf,'same')
        under = np.zeros_like(sum_col)
        under[:-1] = sum_col[1:]
        deriv = under-sum_col    
        sum_col_mean = np.mean(sum_col)
        print ('sum_col mean',np.mean(sum_col))
        i0 = 0
        pillas = []                
        list.append(0)
        for i in range(1,len(deriv)):    
            if deriv[i-1]== 0 and deriv[i]==0:
                continue
            elif (sum_col[i]<sum_col_mean*0.75 and (deriv[i-1]<= 0 and deriv[i]>0)) or i== len(deriv)-1:
                pilla_w = i-i0                
                pillas.append(pilla_w)
                list.append(i)
                i0 = i
                if axis==1: self.gridArr[i,::2] = 255
                else : self.gridArr[::2,i] = 255
                
        print ('size ',self.grayArr.shape)
        print ('pillar_count ',len(pillas))        
        pilla_mean = np.mean(np.array(pillas))
                
        letterH = (int)(pilla_mean)
        print ('cell_h',letterH)
        self.cell_hw[axis-1] = letterH
        
        if self.isFigure:
            plt.figure(self.index_figure)            
            self.index_figure+=1
            plt.subplot(1,2,1)
            plt.plot(sum_col)    
            plt.title('axis:'+str(axis) +', '+ str(len(list))+', '+ str(letterH)+'m:'+str(sum_col_mean))
            plt.grid(which='both', axis='both')
                                    
            plt.subplot(1,2,2)
            plt.title('original')
            img = Image.fromarray(self.gridArr)            
            plt.imshow(img, cmap = plt.get_cmap('gray'))  
              
    
    def GetGrid(self):  
        self.GetLetterSize(0)
        self.GetLetterSize(1)

    def RemovePadding(self, src):
        sumRow = np.sum(src, axis=1)
        sumCol = np.sum(src, axis=0)
        
        h = len(sumRow)
        w = len(sumCol)
        y0 = 0
        y1 = h
        x0 = 0
        x1 = w

        threshold = np.mean(sumRow)/2
        for y in range(h):            
            weight = 1 + (abs(h/2 - y)/ (h/2))
            if sumRow[y] * weight > threshold : 
                y0 = y
                break;
        for y in range(len(sumRow)):
            index = len(sumRow)-1-y
            center_dist = abs(h/2 - y)
            weight = 1 + (center_dist / (h/2))
            if sumRow[index]*weight> threshold : 
                y1 = index
                break;        
        threshold = np.mean(sumCol)/2
        for x in range(len(sumCol)):
            center_dist = abs(w/2 - x)
            weight = 1 + (center_dist / (w/2))
            if sumCol[x]*weight> threshold : 
                x0 = x
                break;
        for x in range(len(sumCol)):
            index = len(sumCol)-1-x
            center_dist = abs(w/2 - x)
            weight = 1 + (center_dist / (w/2))
            if sumCol[index] * weight> threshold : 
                x1 = index
                break;
        dst = src[y0:y1+1,x0:x1+1]
        #print ('sum axis=1',y0,'~',y1,sumRow)
        #print ('sum axis=0',x0,'~',x1,sumCol)
        #print ('cut', y0,y1,x0,x1)
        return dst
    
    def ShowGridCell(self):
        rows = len(self.font_offsets_y)
        cols = len(self.font_offsets_x)
        print('rows:',rows,'cols',cols)
        if self.isFigure:
            plt.figure(self.index_figure)            
            self.index_figure+=1
            subRow = 4
            subCol = 4
            lpf = [0,0.1,0.2,0.3,0.4,0.3,0.2,0.1,0]
            #lpf = [0,0.1,0.2,0.3,0.2,0.1,0]
            for iter in range(10):
                subplot_index = 1
                for i in range(subRow):
                    y = np.random.randint(0, rows-1)
                    x = np.random.randint(0, cols-1)                    
                    y0 = self.font_offsets_y[y]
                    y1 = self.font_offsets_y[y+1] + 1
                    x0 = self.font_offsets_x[x]
                    x1 = self.font_offsets_x[x+1] + 1
                    print('y0,y1,x0,x1',y0,y1,x0,x1)
                    gridCell = self.grayArr[y0:y1,x0:x1]
                    cellChar = self.RemovePadding(gridCell)
                    print ('cellChar',cellChar.shape)
                    imgCell = Image.fromarray(gridCell)            
                    imgChar = Image.fromarray(cellChar)
                    plt.subplot(subRow,subCol,subplot_index)
                    subplot_index+=1
                    gridCell_mean = np.mean(gridCell)
                    plt.title(str(y)+'/'+str(x)+str(gridCell.shape)+' '+gridCell_mean)
                    plt.imshow(imgCell, cmap = plt.get_cmap('gray'))

                    plt.subplot(subRow,subCol,subplot_index)
                    subplot_index+=1
                    plt.title('cell'+str(y)+'/'+str(x)+str(cellChar.shape))
                    plt.imshow(imgChar, cmap = plt.get_cmap('gray'))

                    plt.subplot(subRow,subCol,subplot_index)
                    subplot_index+=1
                    sum_row = np.sum(gridCell, axis=0)
                    sum_row_conv = np.convolve(sum_row,lpf,'same')
                    
                    plt.title('sum_row'+str(np.mean(sum_row)/3))
                    #plt.axis([0, len(sum_row), np.mean(sum_row)/3, np.max(sum_row)])                    
                    plt.plot(sum_row)
                    plt.plot(sum_row_conv)

                    plt.subplot(subRow,subCol,subplot_index)
                    subplot_index+=1                    
                    sum_col = np.sum(gridCell, axis=1)
                    sum_col_conv = np.convolve(sum_col,lpf,'same')
                    plt.plot(sum_col)
                    plt.plot(sum_col_conv)
                plt.show()

        return 0

    def RemoveSmallArray(self, list):
        #check shape mean
        sum_h = 0
        sum_w = 0
        for i in range(len(list)):
            shape = list[i].shape
            sum_h += shape[0]
            sum_w += shape[1]
        
        mean_h = sum_h/len(list)
        mean_w = sum_w/len(list)
        print ('mean shape',sum_h/len(list),sum_w/len(list))
        subList = []
        for i in range(len(list)):
            target = list[i]
            shape = target.shape
            if shape[0] < mean_h/2 or shape[1] < mean_w/2 : 
                continue
            subList.append(target)
        return subList

    def GetCharacters(self):

        rows = len(self.font_offsets_y)
        cols = len(self.font_offsets_x)
        print('rows:',rows,'cols',cols)
                
        list_candiate_character = []
        for y in range(rows-1):
            for x in range(cols-1):
                #print (len(list_candiate_character),'y:',y,'x:',x)                
                y0 = self.font_offsets_y[y]
                y1 = self.font_offsets_y[y+1] + 1
                x0 = self.font_offsets_x[x]
                x1 = self.font_offsets_x[x+1] + 1
                #print('y0,y1,x0,x1',y0,y1,x0,x1)
                gridCell = self.grayArr[y0:y1,x0:x1]
                gridCell_mean = np.mean(gridCell)
                if gridCell_mean > 2:
                    cellChar = self.RemovePadding(gridCell)
                    list_candiate_character.append(cellChar)
        print ('list_candiate_character',len(list_candiate_character))

        self.list_character = self.RemoveSmallArray(list_candiate_character)        
        print ('character count',len(self.list_character))
        return len(self.list_character)
        
    def SaveCharacters(self, folder):
        
        print ('SaveCharacters',len(self.list_character), folder)
        if not os.path.exists(folder):os.makedirs(folder)
        for i in range(len(self.list_character)):
            imgChar = Image.fromarray(self.list_character[i])
            fileName = folder+'/'+str(i)+".png"

            imgChar.save(fileName)
