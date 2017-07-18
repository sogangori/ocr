import unittest
from util.OCR_Pre import OCR_Pre
import Data

class Test_test1(unittest.TestCase):
   
   def test_B(self):
       self.assertTrue(True)
   
   def test_C(self):
       self.assertEquals(5,5)

   def test_Cut(self):
       ocr = OCR_Pre()
       gt = Data.cut_GT
       ocr.Read(Data.path0)       
       cut = ocr.GetCutIndexs()       
       self.assertEquals(cut,gt[0])
       
       ocr.Read(Data.path1)
       cut = ocr.GetCutIndexs()       
       self.assertEquals(cut,gt[1])
       
       ocr.Read(Data.path2)
       cut = ocr.GetCutIndexs()       
       self.assertEquals(cut,gt[2])

   def test_Rotate(self):
       ocr = OCR_Pre()
       gt = Data.angle_GT
       
       ocr.Read(Data.path0)       
       ocr.GetCutIndexs()
       angles = ocr.GetRotations()       
       self.assertAlmostEquals(angles,gt[0])
              
       ocr.Read(Data.path1)       
       ocr.GetCutIndexs()
       angles = ocr.GetRotations()       
       self.assertAlmostEquals(angles,gt[1])

       ocr.Read(Data.path2)       
       ocr.GetCutIndexs()
       angles = ocr.GetRotations()       
       self.assertAlmostEquals(angles,gt[2])

   def test_row_count1(self):
       ocr = OCR_Pre()
       gt = Data.font_row_GT
       
       ocr.Read(Data.path0)       
       ocr.GetCutIndexs()
       ocr.GetRotations()
       ocr.CutPaddings()       
       ocr.GetLetterSizes()       
       self.assertEquals(len(ocr.font_offsets_y),len(gt[0]))
       self.assertEquals(len(ocr.font_offsets_y[0]),gt[0][0])#22!=24
       self.assertEquals(len(ocr.font_offsets_y[1]),gt[0][1])#22!=24
       self.assertEquals(ocr.letterSizeCandidate,[15,15])

   def test_row_count2(self):
       ocr = OCR_Pre()
       gt = Data.font_row_GT
       
       ocr.Read(Data.path2)       
       ocr.GetCutIndexs()
       ocr.GetRotations()
       ocr.CutPaddings()       
       ocr.GetLetterSizes()       
       self.assertEquals(len(ocr.font_offsets_y),len(gt[2]))
       self.assertEquals(len(ocr.font_offsets_y[0]),gt[2])
       self.assertEquals(ocr.letterSizeCandidate,[15])

if __name__ == '__main__':
   unittest.main()