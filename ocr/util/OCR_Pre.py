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
    plot_column = 2
    local_min_offset_ratio = 0.75
    isShowLocalMin = not True
    isShowPatch = not True
    isFigure = False

    def __init__(self):
        print ("OCR_Pre __init__")

    def ReadImage(self, path):
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
            mean = 1/(mean_1+elsilon)  * 1/(mean_2+elsilon)
            if mean > maxMean :
                #print ('rotate angle',angle,'mean',mean)
                maxMean  = mean
                maxAngle = angle

        print ('max Angle',maxAngle,'maxMean',maxMean)    
        return maxAngle-0

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
            image_rot = image.rotate(angle)
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
        return self.rotateImage

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

    def GetLetterSize(self, src):    
        sum_col = np.sum(src, axis=1)
        under = np.zeros_like(sum_col)
        under[:-1] = sum_col[1:]
        deriv = under-sum_col    
        sum_col_mean = np.mean(sum_col)
        print ('sum_col mean',np.mean(sum_col))
        i0 = 0
        pillas = []
        font_offsets_candidate_y = []
        font_offsets_y = []
        for i in range(len(deriv)-1):
            if sum_col[i]<sum_col_mean/3 and ((deriv[i]< 0 and deriv[i+1]>0) or (deriv[i]== 0 and deriv[i+1]>0)):
                pilla_w = i-i0
                pillas.append(pilla_w)
                font_offsets_y.append(i)
                print ('i', i, pilla_w)
                i0 = i
    
        print ('pillar_count ',len(pillas))
        pilla_arr = np.array(pillas)
        pilla_std = np.std(pilla_arr)
        pilla_mean = np.mean(pilla_arr)
        mean_simmilar_sum=0
        mean_simmilar_sum_count=0
        print ('pilla_arr',pilla_arr.shape,pilla_mean, pilla_std)
        for i in range(len(pillas)):
            v = pilla_arr[i]
            if v > pilla_mean- pilla_std and v < pilla_mean+ pilla_std :
                mean_simmilar_sum +=v
                mean_simmilar_sum_count+=1
            #else: //TODO
    
        letterSizeCandidate = mean_simmilar_sum/mean_simmilar_sum_count
        print ('letterSizeCandidate',letterSizeCandidate,mean_simmilar_sum_count)
        if self.isFigure and self.isShowPatch:
            plt.figure(1)
            plt.plot(sum_col)    
            plt.show()
        return font_offsets_y,letterSizeCandidate

    def SlideWindow(self, src,font_offsets_y, patchSize):
        #일단 이걸로 찾어보고 얘기하자
        h = src.shape[0]
        w = src.shape[1]
        print ('font_offsets_y',font_offsets_y)
        for i in range(len(font_offsets_y)):
            y = font_offsets_y[i]
            for x in range((int)(w/10)):            
                patch = src[y:y+patchSize, x:x+patchSize]
                patch_sum = np.sum(patch)
                if patch_sum>patchSize:
                    plt.imshow(patch, cmap = plt.get_cmap('gray'))
                    plt.ylabel(y)
                    plt.draw()
                    plt.pause(0.001) 
                    #plt.show()
        return 0

    def GetLetterSizes(self):
        srcList = self.cut_src
        for i in range(len(srcList)):
            src = srcList[i]
            font_offsets_y, candidateSize = self.GetLetterSize(src)
            self.SlideWindow(src,font_offsets_y, candidateSize )

        return 0

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
            plt.show()

        self.localMinimums  = self.GetLocalMinimum(best_cos_arr+1)   
        return len(self.localMinimums )
        
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


#ocr = OCR_Pre()

#folder = 'c:/Users/pc/Desktop/논문/OCR/image/'
#path = folder+'sample0.jpg'
##path = folder+'sample2.jpg'
##path = folder+'sample1.jpeg'

#ocr.ReadImage(path)
#ocr.ResizeIfBig()
#cutPoint = ocr.GetCutIndexs() #돌리고 자르자 피벗
#print ('cutPoint',cutPoint)
#ocr.GetRotations()
#ocr.CutPaddings()

#if isFigure: 
#    ocr.GetLetterSizes()
#plt.show()