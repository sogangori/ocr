from PIL import Image
import PIL.ImageOps 
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

class TrainCharacterExtractor():
    grayImg = 0
    grayArr = 0
    gridArr = 0
    cell_hw = [0,0]
    font_offsets_y = []
    font_offsets_x = []
    isFigure = False
    index_figure = 0

    def __init__(self):
        print ("TrainCharacterExtractor __init__")

    def Reset(self):
        self.font_offsets_y.clear()
        self.font_offsets_x.clear()

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
        self.grayImg = PIL.ImageOps.invert(grayImg)
        self.grayArr = np.asarray(self.grayImg).copy()       
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
        ratate_step = 0.02
        angle_min = -1.0
        angle_max = 1.0
        angles = []
        for angle in np.arange(angle_min,angle_max, ratate_step):
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
        candidate2Index = 0                
        list.append(0)
        for i in range(1,len(deriv)-1):            
            if i== len(deriv)-2:
                pilla_w = i-i0
                pillas.append(pilla_w)
                list.append(i)
                if axis==1: self.gridArr[i,:] = 255
                else : self.gridArr[:,i] = 255
            elif deriv[i-1]== 0 and deriv[i]==0:
                continue
            elif sum_col[i]<sum_col_mean/1 and (deriv[i-1]<= 0 and deriv[i]>0.5):
                pilla_w = i-i0                
                pillas.append(pilla_w)
                list.append(i)
                #print ('axis',axis,candidate2Index, i0,'~', i,'w:', pilla_w)
                i0 = i
                candidate2Index+=1
                if axis==1: self.gridArr[i,:] = 255
                else : self.gridArr[:,i] = 255
                
        
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
            plt.title('axis:'+str(axis) +', '+ str(len(list))+', '+ str(letterH))
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
        
        y0 = 0
        y1 = len(sumRow)
        x0 = 0
        x1 = len(sumCol)
        threshold = np.mean(sumRow)/3
        for y in range(len(sumRow)):
            if sumRow[y]> threshold : 
                y0 = y
                break;
        for y in range(len(sumRow)):
            index = len(sumRow)-1-y
            if sumRow[index]> threshold : 
                y1 = index
                break;        
        threshold = np.mean(sumCol)/3
        for x in range(len(sumCol)):
            if sumCol[x]> threshold : 
                x0 = x
                break;
        for x in range(len(sumCol)):
            index = len(sumCol)-1-x
            if sumCol[index]> threshold : 
                x1 = index
                break;
        dst = src[y0:y1+1,x0:x1+1]
        print ('sum axis=1',y0,'~',y1,sumRow)
        print ('sum axis=0',x0,'~',x1,sumCol)
        print ('cut', y0,y1,x0,x1)
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
                    plt.title('cell'+str(y)+'/'+str(x)+str(gridCell.shape))
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

    def ShowGridBorder(self):
        rows = len(self.font_offsets_y)
        cols = len(self.font_offsets_x)
        print('rows:',rows,'cols',cols)
        if self.isFigure:
            plt.figure(self.index_figure)            
            self.index_figure+=1
            lpf = [0,0.1,0.2,0.3,0.4,0.3,0.2,0.1,0]
            #lpf = [0,0.1,0.2,0.3,0.2,0.1,0]
            coordinate = [[0,0],[0,1], [0,cols-3],[0,cols-2], 
                          [rows-2,0],[rows-2,1], [rows-2,cols-3], [rows-2,cols-2]]            
            subCol = len(coordinate)
            subRow = 2            
            for i in range(len(coordinate)):
                print ('i',i,coordinate)
                y = coordinate[i][0]
                x = coordinate[i][1]
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
                plt.subplot(subRow,subCol,1+i)
                plt.title('cell'+str(y)+'/'+str(x)+str(gridCell.shape))
                plt.imshow(imgCell, cmap = plt.get_cmap('gray'))

                plt.subplot(subRow,subCol,subCol + 1+i)                
                plt.title('cell'+str(y)+'/'+str(x)+str(cellChar.shape))
                plt.imshow(imgChar, cmap = plt.get_cmap('gray'))                                
                    
            plt.show()

