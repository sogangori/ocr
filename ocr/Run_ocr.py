import matplotlib.pyplot as plt
from util.OCR_Pre import OCR_Pre
import Data

ocr = OCR_Pre()
ocr.isFigure = True
ocr.isShowLocalMin = True

ocr.Read(Data.path0)
ocr.GetCutIndexs() 
ocr.GetRotations()
ocr.SaveRotations(Data.folderTest+'cut')
#ocr.CutPaddings()
#ocr.GetLetterSizes()
#ocr.GetCandidateRows(Data.folderTest+'cut_0')
#ocr.SlideCandidateRow()
plt.show()
