import matplotlib.pyplot as plt
from util.OCR_Pre import OCR_Pre
import Data

ocr = OCR_Pre()
ocr.isFigure = True

ocr.ReadImage(Data.path0)
ocr.ResizeIfBig()
cutPoint = ocr.GetCutIndexs() 
print ('cutPoint',cutPoint)
ocr.GetRotations()
ocr.CutPaddings()

ocr.GetLetterSizes()
plt.show()