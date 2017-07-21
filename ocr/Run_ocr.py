import matplotlib.pyplot as plt
from util.OCR_Pre import OCR_Pre
import Data

ocr = OCR_Pre()
ocr.isFigure = True
ocr.isShowLocalMin = True
ocr.isShowLetterHeight = not True

ocr.Read(Data.path0)
ocr.GetCutIndexs() 
ocr.GetRotations()
ocr.CutPaddings()
ocr.GetLetterSizes()
#ocr.SlideCandidateRow()
plt.show()
