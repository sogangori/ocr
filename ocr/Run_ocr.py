import matplotlib.pyplot as plt
from util.OCR_Pre import OCR_Pre

ocr = OCR_Pre()
ocr.isFigure = True

folder = 'c:/Users/pc/Desktop/paper/OCR/image/'
#path = folder+'sample0.jpg'
#path = folder+'sample2.jpg'
path = folder+'sample1.jpeg'

ocr.ReadImage(path)
ocr.ResizeIfBig()
cutPoint = ocr.GetCutIndexs() 
print ('cutPoint',cutPoint)
ocr.GetRotations()
ocr.CutPaddings()

ocr.GetLetterSizes()
plt.show()