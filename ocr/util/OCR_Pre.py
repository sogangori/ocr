from PIL import Image
import PIL.ImageOps 
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy.linalg as li


class OCR_Pre():
    
    grayImg = 0
    grayArr = 0
    localMinimums = 0
    rotateImage = []
    cut_src = []
    rot_angles = []
    font_offsets_y = []
    plot_column = 2
    local_min_offset_ratio = 0.75
    letterSizeCandidate = []
    isShowLocalMin = not True
    isShowLetterHeight = not True
    isFigure = False

    def __init__(self):
        print ("OCR_Pre __init__")

    def Reset(self):
        self.rot_angles.clear()
        self.cut_src.clear()
        self.rotateImage.clear()

    def Read(self, path):
        self.Reset()
        img = Image.open(path)
        grayImg = img.convert('L')
        self.grayImg = PIL.ImageOps.invert(grayImg)
        self.ResizeIfBig()
        return self.grayImg

    def ResizeIfBig(self):
        image_max_h = 512
        if self.grayImg.height > image_max_h :
            self.grayImg = self.grayImg.resize([image_max_h,image_max_h])
        grayArr = np.asarray(self.grayImg)
        self.grayArr = np.round(grayArr/128)

    def GetRotation(self, src):    
        #angle = li.eig(src)
        image = Image.fromarray(src)
        maxAngle = 0
        maxMean = 0
        ratate_step = 0.5
        for i in range(-1,180*(int)(1/ratate_step)):
            angle = i*ratate_step
            image_rot = image.rotate(angle)
            src = np.asarray(image_rot)
            mean_1 = np.mean(np.std(src, axis=1))
            mean_2 = np.mean(np.std(src, axis=0))
            elsilon = 0.1
            mean = 1/(mean_1+elsilon)  * 1/(mean_2+elsilon)
            if mean > maxMean :
                #print ('rotate angle',angle,'mean',mean)
                maxMean  = mean
                maxAngle = angle

        print ('max Angle',maxAngle,'maxMean',maxMean)
        image_rot = image.rotate(maxAngle-90)
        image_rot = np.asarray(image_rot)
        return image_rot

    def GetRotationAngle(self, src):    
        image = Image.fromarray(src)
        maxAngle = 0
        maxMean = 0
        ratate_step = 0.5
        angle_min = -30
        angle_max = 30
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
                #print ('rotate angle',angle,'mean',mean)
                maxMean  = mean
                maxAngle = angle

        print ('max Angle',maxAngle,'maxMean',maxMean)    
        return maxAngle

    def GetRotations(self):    
        src = self.grayArr
        localMinimumIndexs = self.localMinimums
        
        x0 = 0
        x1 = 0
        width = src.shape[1]
        candidate_offset = []
        for i in range(len(localMinimumIndexs)):
            x1 = (int)(localMinimumIndexs[i]/100 * width)+1
            candidate_offset.append([x0,x1])
            x0 = x1
        candidate_offset.append([x0,width])

        image = Image.fromarray(src)
        for i in range(len(candidate_offset)):
            offset = candidate_offset[i]
            print ('offset ',offset )
            candidate = src[:,offset [0]:offset [1]]            
            angle = self.GetRotationAngle(candidate)        
            self.rot_angles.append(angle)
            image_rot = image.rotate(angle, resample=Image.BICUBIC, expand=True)            
            src_rot = np.array(image_rot)
            image_rot_cut = src_rot[:,offset [0]:offset [1]]
            self.rotateImage.append(image_rot_cut)
                
        plot_row  = len(self.rotateImage)
        if self.isFigure:
            for i in range(len(self.rotateImage)):        
                img_rotate = self.rotateImage[i]    
                plt.subplot(plot_row,self.plot_column,self.plot_column*i+1)    
                plt.title('rotate'+str(img_rotate.shape))
                plt.imshow(img_rotate, cmap = plt.get_cmap('gray'))
        return self.rot_angles

    def CutPadding(self, src):
        sum_row = np.sum(src, axis=0)
        sum_col = np.sum(src, axis=1)
    
        x0=0
        x1=0
        y0=0
        y1=0
        threshold = 1
        for i in range(len(sum_row)):
            if sum_row[i]>threshold: 
                x0 = i
                break
        for i in range(len(sum_row)):
            index = len(sum_row)-1-i
            if sum_row[index]>threshold: 
                x1 = index
                break
        for i in range(len(sum_col)):
            if sum_col[i]>threshold: 
                y0 = i
                break
        for i in range(len(sum_col)):
            index = len(sum_col)-1-i
            if sum_col[index]>threshold: 
                y1 = index
                break
        print ('cut', x0,x1,y0,y1)
        return src[y0:y1+1, x0:x1+1]

    def CutPaddings(self):
        
        srcList = self.rotateImage
        for i in range(len(srcList)):
            dst = self.CutPadding(srcList[i])        
            self.cut_src.append(dst)

        plot_row  = len(self.cut_src)
        if self.isFigure:
            for i in range(len(self.cut_src)):        
                img_rotate = self.cut_src[i]    
                plt.subplot(plot_row,self.plot_column,self.plot_column*i+2)
                plt.title('cut'+str(img_rotate.shape))
                plt.imshow(img_rotate, cmap = plt.get_cmap('gray'))

        return self.cut_src

    def RemoveLine(self, src):
        sobelH = ndimage.sobel(src, 0)
        sobelV = ndimage.sobel(src, 1)
        if self.isFigure:
            plt.figure(2)
            plt.title('sobelH')
            plt.imshow(sobelH)            
            plt.figure(3)
            plt.title('sobelVertival')
            plt.imshow(sobelV)        

    def RemoveLines(self):
        srcList = self.cut_src
        for i in range(len(srcList)):
            self.RemoveLine(srcList[i])

    def GetLetterSize(self, src):    
        lpf = [0,0.1,0.2,0.3,0.4,0.3,0.2,0.1,0]
        sum_col = np.sum(src, axis=1)
        #sum_col = np.convolve(sum_col,lpf,'same')
        sum_col = np.convolve(sum_col,lpf,'valid')
        #sum_col = np.convolve(sum_col,lpf,'valid')
        under = np.zeros_like(sum_col)
        under[:-1] = sum_col[1:]
        deriv = under-sum_col    
        sum_col_mean = np.mean(sum_col)
        print ('sum_col mean',np.mean(sum_col))
        i0 = 0
        pillas = []
        pillas_offset = []        
                        
        for i in range(len(deriv)-1):
            #if sum_col[i]<sum_col_mean/2 and ((deriv[i]< 0 and deriv[i+1]>0) or (deriv[i]== 0 and deriv[i+1]>0)):
            if deriv[i]== 0 and deriv[i+1]==0:
                continue
            elif sum_col[i]<sum_col_mean/1 and (deriv[i]<= 0 and deriv[i+1]>0):
                pilla_w = i-i0                
                pillas.append(pilla_w)
                pillas_offset.append(i)
                print ('row',i0,'~', i,'w:', pilla_w)
                i0 = i
    
        print ('pillar_count ',len(pillas))
        pilla_arr = np.array(pillas)
        pilla_std = np.std(pilla_arr)
        pilla_mean = np.mean(pilla_arr)
        mean_simmilar_sum=0
        mean_simmilar_sum_count=0
        print ('pilla_arr',pilla_arr.shape,pilla_mean, pilla_std)
        candidate2Index=0
        font_offset = []
        y0 = 0
        for i in range(len(pillas)):
            v = pilla_arr[i]
            y1 = pillas_offset[i]
            if i>0: y0 = pillas_offset[i-1]
            if v > pilla_mean- pilla_std and v < pilla_mean+ pilla_std :
                mean_simmilar_sum +=v
                mean_simmilar_sum_count+=1
                font_offset.append(pillas_offset[i])                                
                print ('font row',i,candidate2Index, 'y:',y0,'~',y1, v)
                candidate2Index+=1
            #else: //TODO
        self.font_offsets_y.append(font_offset)
        letterH = (int)(mean_simmilar_sum/mean_simmilar_sum_count)
        print ('letterSizeCandidate',letterH,mean_simmilar_sum,mean_simmilar_sum_count)
        
        self.letterSizeCandidate.append(letterH)
        if self.isFigure and self.isShowLetterHeight:
            plt.figure(1)
            plt.plot(sum_col)    
            plt.grid(which='both', axis='both')
            plt.show()        

    def SlideWindow(self, src,font_offsets_y, patchSize):
        #일단 이걸로 찾어보고 얘기하자
        h = src.shape[0]
        w = src.shape[1]
        print ('font_offsets_y',font_offsets_y)
        for i in range(len(font_offsets_y)):
            y = font_offsets_y[i]
            for x in range((int)(w/30)):            
                patch = src[y:y+patchSize, x:x+patchSize]
                patch_sum = np.sum(patch)
                if patch_sum>patchSize:
                    plt.imshow(patch, cmap = plt.get_cmap('gray'))
                    plt.ylabel(y)
                    plt.draw()
                    plt.pause(0.001) 
                    #plt.show()        

    def GetLetterSizes(self):
        srcList = self.cut_src
        for i in range(len(srcList)):
            src = srcList[i]
            self.GetLetterSize(src)            
    
    def SlideCandidateRow(self):
        srcList = self.cut_src
        for i in range(len(srcList)):
            src = srcList[i]
            self.font_offsets_y.clear()
            self.GetLetterSize(src)
            self.SlideWindow(src,self.font_offsets_y[i], self.letterSizeCandidate[i])

    def Normalize(self, src):
        dst = (src - np.min(src)) / (np.max(src)-np.min(src))
        return dst

    def PlotRowSum(self, image):
    
        src = np.asarray(image)
        sum_row = np.sum(src, axis=0)
        R = 4
        sum_row_2d = np.reshape(sum_row,[-1,R])
        sum_row_r = np.mean(sum_row_2d, axis=1)
        rows = Normalize(sum_row_r)*2-1
        plt.plot(rows)    
        plt.show()
        return 0


    def GetLocalMinimum(self, src):    
    
        under = np.zeros_like(src)
        under[:-1] = src[1:]
        deriv = under-src
        deriv_abs = deriv
        localMinimumIndexs = []    
        local_min_offset = (int)(len(deriv)*self.local_min_offset_ratio)
        for i in range(local_min_offset-1):
            if deriv[i]< 0 and deriv[i+1]>0:
                #print ('local', i,deriv[i],deriv[i+1])    
                localMinimumIndexs.append(i)
    
        return localMinimumIndexs

    def GetCutIndexs(self):
        image = self.grayImg
        src = self.grayArr
        image_100 = image.resize([100,image.height])
        src_100 = np.asarray(image_100)
        print ('src resize', src.shape ,'->',src_100.shape)
        sum_row = np.sum(src_100, axis=0)
            
        rows_normal = self.Normalize(sum_row)*2-1
        x = np.arange(len(sum_row ))
    
        loss_min = 10000
        best_cos_arr =0
        iteration = 100
        step = 0.5    
        for i in range(1,iteration):        
            cos_arr = -np.cos(x/(i*step))
            loss = np.mean(np.square(rows_normal - cos_arr))        
            if loss < loss_min: 
                loss_min = loss
                best_cos_arr = cos_arr
                #print ('loss_min_scale',i,'loss',loss)        
            cos_arr = np.sin(x/(i*step))
            loss = np.mean(np.square(rows_normal - cos_arr))
            if loss < loss_min: 
                loss_min = loss            
                best_cos_arr = cos_arr
        
        if self.isShowLocalMin:
            plt.plot(x,rows_normal)    
            plt.plot(x,best_cos_arr)    
            plt.grid(which='both', axis='both')
            plt.show()

        self.localMinimums  = self.GetLocalMinimum(best_cos_arr+1)   
        return self.localMinimums
        
    def DivideColumn(self, image, localMinimumIndexs):       
        src = np.asarray(image)
        print ('localMinimumIndexs ',localMinimumIndexs)
        x0 = 0
        x1 = 0
        columns = []
        for i in range(len(localMinimumIndexs)):
            x1 = (int)(localMinimumIndexs[i]/ 100* image.width) 
            column = src[:,x0:x1]
            x0 = x1
            columns.append(column)
            print ('x0~x1',x0,x1, image.width, column.shape)        
    
        column = src[:,x0:]
        columns.append(column)    
        return columns
