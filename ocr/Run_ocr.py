import matplotlib.pyplot as plt
from util.OCR_Pre import OCR_Pre
import Data

ocr = OCR_Pre()
ocr.isFigure = True
ocr.isShowLocalMin = True
ocr.isShowLetterHeight = True

ocr.ReadImage(Data.path0)
ocr.ResizeIfBig()
ocr.GetCutIndexs() 
ocr.GetRotations()
ocr.CutPaddings()
ocr.GetLetterSizes()
ocr.SlideCandidateRow()
plt.show()
