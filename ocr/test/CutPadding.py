import numpy as np

def CutPadding(src):
    sum_row = np.sum(a, axis=0)
    sum_col = np.sum(a, axis=1)
    print (a.shape)
    print (sum_row)
    print (sum_col)

    x0=0
    x1=0
    y0=0
    y1=0
    for i in range(len(sum_row)):
        if sum_row[i]>0: 
            x0 = i
            break
    for i in range(len(sum_row)):
        index = len(sum_row)-1-i
        if sum_row[index]>0: 
            x1 = index
            break
    for i in range(len(sum_col)):
        if sum_col[i]>0: 
            y0 = i
            break
    for i in range(len(sum_col)):
        index = len(sum_col)-1-i
        if sum_col[index]>0: 
            y1 = index
            break
    print ('cut', x0,x1,y0,y1)
    return src[y0:y1+1, x0:x1+1]

a = [[0,0,0,0,0 ],
     [0,0,0,0,0 ],
     [0,1,2,3,0 ],
     [0,4,5,6,0 ],
     [0,0,0,0,0 ]]
a = np.array(a)

b = CutPadding(a)
print ('b', b)

