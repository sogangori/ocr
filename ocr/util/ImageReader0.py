from PIL import Image
import PIL.ImageOps 
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy.linalg as li

def GetRotation(src):    
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

def GetRotations(srcList):    
    rotated = []
    for i in range(len(srcList)):
        rotated.append( GetRotation(srcList[i]))
    return rotated

def CutPadding(src):
    sum_row = np.sum(src, axis=0)
    sum_col = np.sum(src, axis=1)
    
    x0=0
    x1=0
    y0=0
    y1=0
    threshold = 100
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

def CutPaddings(srcList):
    cut_src = []
    for i in range(len(srcList)):
        dst = CutPadding(srcList[i])        
        cut_src.append(dst)
    return cut_src

def Normalize(src):
    dst = (src - np.min(src)) / (np.max(src)-np.min(src))
    return dst

def PlotRowSum(image):
    
    src = np.asarray(image)
    sum_row = np.sum(src, axis=0)
    R = 4
    sum_row_2d = np.reshape(sum_row,[-1,R])
    sum_row_r = np.mean(sum_row_2d, axis=1)
    rows = Normalize(sum_row_r)*2-1
    plt.plot(rows)    
    plt.show()
    return 0


def GetLocalMinimum(src):    
    
    under = np.zeros_like(src)
    under[:-1] = src[1:]
    deriv = under-src
    deriv_abs = deriv
    localMinimumIndexs = []
    local_min_offset = (int)(len(deriv)*0.9)
    for i in range(local_min_offset-1):
        if deriv[i]<0 and deriv[i+1]>0:
            #print ('local', i,deriv[i],deriv[i+1])    
            localMinimumIndexs.append(i)
    
    return localMinimumIndexs

def GetCutIndexs(image):
    
    src = np.asarray(image)
    image_100 = image.resize([100,image.height])
    src_100 = np.asarray(image_100)
    print ('src resize', src.shape ,'->',src_100.shape)
    sum_row = np.sum(src_100, axis=0)
            
    rows_normal = Normalize(sum_row)*2-1
    x = np.arange(len(sum_row ))
    
    loss_min_scale = 0 
    loss_min = 10000
    iteration = 1000
    step = 0.5    
    for i in range(1,iteration):        
        cos_arr = -np.cos(x/(i*step))
        loss = np.mean(np.square(rows_normal - cos_arr))
        if loss < loss_min: 
            loss_min = loss
            loss_min_scale = i
            #print ('loss_min_scale',i,'loss',loss)

    print ('loss_min_scale',loss_min_scale)
    cos_arr = - np.cos(x/(loss_min_scale*step))
    plt.plot(x,rows_normal)    
    plt.plot(x,cos_arr)    
    plt.show()
    localMinimumIndexs = GetLocalMinimum(cos_arr+1)    
    return localMinimumIndexs


def DivideColumn(image, localMinimumIndexs):       
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


#path = 'c:/Users/pc/Desktop/논문/OCR/image/sample0.jpg'
path = 'c:/Users/pc/Desktop/논문/OCR/image/sample2.jpg'
img = Image.open(path)
grayImg = img.convert('L')
grayImg = PIL.ImageOps.invert(grayImg)
image_max_h = 512
if grayImg.height > image_max_h :
    grayImg = grayImg.resize([image_max_h,image_max_h])
grayArr = np.asarray(grayImg)

#img.save('greyscale.png')

#edge_horizont = ndimage.sobel(grayArr, 0)
#edge_vertical = ndimage.sobel(grayArr, 1)
#magnitude = np.hypot(edge_horizont, edge_vertical)
#plt.imshow(grayArr, cmap = plt.get_cmap('gray'))
#plt.show()

#돌리고 자르자 피벗
localMinimums = GetCutIndexs(grayImg)
colums = DivideColumn(grayImg,localMinimums)
plot_column = 3
plot_row  = len(colums)
for i in range(len(colums)):        
    column = colums[i]
    plt.subplot(plot_row,plot_column,plot_column*i+1)
    plt.title('cut')
    plt.imshow(column, cmap = plt.get_cmap('gray'))    

img_rotates = GetRotations(colums)

for i in range(len(img_rotates)):        
    img_rotate = img_rotates[i]    
    plt.subplot(plot_row,plot_column,plot_column*i+2)    
    plt.title('rotate'+str(img_rotate.shape))
    plt.imshow(img_rotate, cmap = plt.get_cmap('gray'))

img_cut_pad = CutPaddings(img_rotates)
for i in range(len(img_cut_pad)):        
    img_rotate = img_cut_pad[i]    
    plt.subplot(plot_row,plot_column,plot_column*i+3)
    plt.title('cut'+str(img_rotate.shape))
    plt.imshow(img_rotate, cmap = plt.get_cmap('gray'))

plt.show()