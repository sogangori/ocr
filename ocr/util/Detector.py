from PIL import Image
import PIL.ImageOps 
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy.linalg as li
import os

def Read(path):
        
    img = Image.open(path)
    #grayImg = img.convert('L')
    print ('size',img )    
    return img,np.asarray(img)

def CutPadding(src):
    mean_w = np.mean(src, axis=0)
    mean_h = np.mean(src, axis=1)
    h = src.shape[0]
    w = src.shape[1]
    print (h,w, len(mean_h),len(mean_w))
    x0=0
    x1=0
    y0=0
    y1=0
    threshold_col = np.mean(mean_w)/9
    threshold_row = np.mean(mean_h)/9
    for i in range(w):
        if mean_w[i]>threshold_col: 
            x0 = i
            break
    for i in range(w):
        index = w-1-i
        if mean_w[index]>threshold_col: 
            x1 = index
            break
    for i in range(h):
        if mean_h[i]>threshold_row: 
            y0 = i
            break
    for i in range(h):
        index = h-1-i
        if mean_h[index]>threshold_row: 
            y1 = index
            break
    dst = src[y0:y1+1, x0:x1+1]
    print ('cut x:%d~%d, y:%d~%d'  % (x0,x1,y0,y1))
    print ('src shape',src.shape)
    print ('dst shape',dst.shape)
    return dst

def Detect_Row(src):    
    lpf = [0,0.1,0.2,0.3,0.4,0.3,0.2,0.1,0]
    #lpf = [0,0.1,0.2,0.3,0.2,0.1,0]
    sum_col = np.sum(src, axis=1)        
    sum_col = np.convolve(sum_col,lpf,'same')
    under = np.zeros_like(sum_col)
    under[:-1] = sum_col[1:]
    deriv = under-sum_col    
    sum_col_mean = np.mean(sum_col)
    sum_col_std = np.std(sum_col)
    print ('sum_col mean',sum_col_mean,'std',sum_col_std )
    
    offsets = []                                
    plt.figure(1)  
    plt.subplot(2,1,1)          
    plt.plot(sum_col)
    plt.title('mean:'+str(np.mean(sum_col))+'std:'+str(sum_col_std))
    plt.subplot(2,1,2)          
    plt.plot(deriv)
    deriv_abs_mean = np.mean(np.abs(deriv))
    plt.title('abs mean:'+str(deriv_abs_mean))
    i0 = 0
    for i in range(1, len(deriv)):            
        if deriv[i-1]== 0 and deriv[i]==0:
            continue
        elif sum_col[i]<sum_col_mean*0.6 and (deriv[i-1]<= 0 and deriv[i]>0):
            if deriv[i]<deriv_abs_mean:
                offsets.append(i)                
                i0 = i            
    
    print ('len(offsets)',len(offsets))
    offsets = Remove_no_contents(src, offsets)
    print ('len(offsets)',len(offsets))    
    offsets = Remove_low_high_height(offsets)
    print ('len(offsets)',len(offsets))    
        
    return offsets  

def Remove_low_high_height(offsets):
    count = len(offsets)
    
    list_height = []
    for i in range(count-1):
        height = offsets[i+1]-offsets[i]
        list_height.append(height)
    
    arr_height = np.array(list_height)    
    mean = np.mean(arr_height)
    std  = np.std(arr_height)

    lists = []
    print (count,mean, std)
    for i in range(len(arr_height)):
        height = arr_height[i]
        if height>mean-std and height<mean+std:
            lists.append(offsets[i])

    return lists
def DrawGrid(src, offsets_y):
    dst = src.copy()
    print ('DrawGrid', src.shape, len(offsets_y))
    for i in range(0, len(offsets_y)):        
        y = offsets_y[i]
        dst[y,::2] = 255   

    return dst

def Remove_no_contents(src, offsets):
    
    count = len(offsets)
    list_height = []
    for i in range(count-1):
        cell = src[offsets[i]:offsets[i+1]]
        
        list_height.append(np.mean(cell))

    arr_height = np.array(list_height)    
    mean = np.mean(arr_height)
    std  = np.std(arr_height)

    lists = []
    for i in range(count-1):
        cell = src[offsets[i]:offsets[i+1]]
        if np.mean(cell) > mean-std:
            lists.append(offsets[i])

    return lists

def Binarize(src):
    shp = src.shape
    dst = np.zeros_like(src)
    delete_count = 0
    for y in range(shp[0]):
        for x in range(shp[1]):                
            if src[y,x]>128: dst[y,x] = 255                    
    return dst

def Get_cell_no_line(src, offsets):
    count = len(offsets)
    list_height = []
    for i in range(count-1):
        height = offsets[i+1]-offsets[i]
        list_height.append(height)
    
    arr_height = np.array(list_height)    
    mean = np.mean(arr_height)
    std  = np.std(arr_height)

    list_offset = []
    
    for i in range(count-1):        
        cell = src[offsets[i]:offsets[i+1]]
        height = arr_height[i]
        if height>mean-std and height<mean+std: 
            list_offset.append(cell)

    return list_offset 

def Normalize( src):
    dst = (src - np.min(src)) / (np.max(src)-np.min(src))
    return dst

def GetLocalMinimum(src):    
    
    under = np.zeros_like(src)
    under[:-1] = src[1:]
    deriv = under-src
    deriv_abs = deriv
    localMinimumIndexs = []    
    
    local_min_offset = len(deriv)
    for i in range(local_min_offset-1):
        if deriv[i]< 0 and deriv[i+1]>0:
            #print ('local', i,deriv[i],deriv[i+1])    
            localMinimumIndexs.append(i)
    
    return localMinimumIndexs

def Remove_row_padding(lists):
    list_dst = []
    for i in range(len(lists)):        
        src = lists[i]        
        list_dst.append(src)
        
        mean = np.mean(src, axis=1)
        mean = Normalize(mean)*2-1
        loss_min = 10000
        j_min = 0
        offset_min = 0
        for j in np.arange(0,10,0.1):
            for offset in range(10):
                x = np.arange(len(mean))+offset
                cos_arr = -np.cos(x/j)
                loss = np.mean(np.square(mean - cos_arr))        
                if loss < loss_min: 
                    loss_min = loss
                    j_min =j                
                    offset_min = offset
                    #print ('loss_min_scale',i,'loss',loss)        
        if i==2:
            plt.figure(5)              
            plt.plot(mean)
            x = np.arange(len(mean))+offset_min
            cos_arr = -np.cos(x/j_min)
            plt.plot(cos_arr)
            localMinIndex = GetLocalMinimum(cos_arr)
            print (i,localMinIndex)

    return list_dst
        
def GetCandidateRows(src2d,offsets,folder):
    cells = Get_cell_no_line(src2d, offsets)
    print ('len(offsets)',len(offsets),'->', len(cells))
    cells = Remove_row_padding(cells)
    if not os.path.exists(folder):os.makedirs(folder)
    for i in range(len(cells)):        
        src = cells[i]
        imgRow = Image.fromarray(src) 
        fileName = folder+'/'+str(i)+".png"
        imgRow.save(fileName)      
        
    return 0

def RemoveLines(src, axis):
        
    lpf = [-0.8,-0.5,0,0.8,1.0,0.8,0,-0.5,-0.8]
    #lpf = [0,0.1,0.2,0.3,0.2,0.1,0]
    
    sums = np.mean(src, axis=axis)
    sums /= np.max(sums)
    dst = src.copy()
    sums = np.convolve(sums,lpf,'same')
    under = np.zeros_like(sums)
    under[:-1] = sums[1:]
    deriv = under-sums    
    line_thick = 6
    for i in range(src.shape[axis-1]):
        if sums[i]>1:
            y0 = i
            y1 = i
            for y in range(line_thick):
                index = i-y
                if deriv[index]<0 and deriv[index+1]>0:
                    y0 = index
                    break
            for y in range(line_thick):
                index = i+y
                if deriv[index]<0 and deriv[index+1]>0:
                    y1 = index+1
                    break
            print ('remove line',y0,y1)
            if axis==1: dst [y0:y1] = 0
            else: dst [:,y0:y1] = 0 

    #plt.figure(5)      
    #plt.title('mean'+str(np.mean(sums)))    
    #plt.subplot(1,2,1)
    #plt.plot(sums)
    #plt.subplot(1,2,2)
    #plt.plot(deriv)
    
    return dst 