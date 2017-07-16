import unittest
from util.OCR_Pre import OCR_Pre

class Test_test1(unittest.TestCase):
   folder = 'c:/Users/pc/Desktop/paper/OCR/image/'       
   path0 = folder+'sample0.jpg'
   path1 = folder+'sample2.jpg'
   path2 = folder+'sample1.jpeg'       

   def test_B(self):
       self.assertTrue(True)
   
   def test_C(self):
       self.assertEquals(5,5)

   def test_Cut(self):
       ocr = OCR_Pre()
       cut_GT = [[44],[46],[]]       
       ocr.ReadImage(self.path0)       
       cut = ocr.GetCutIndexs()       
       self.assertEquals(cut,cut_GT[0])
       
       ocr.ReadImage(self.path1)
       cut = ocr.GetCutIndexs()       
       self.assertEquals(cut,cut_GT[1])
       
       ocr.ReadImage(self.path2)
       cut = ocr.GetCutIndexs()       
       self.assertEquals(cut,cut_GT[2])

   def test_Rotate(self):
       ocr = OCR_Pre()
       angle_GT = [[1.5, 1.0],[-1.0, -1.0],[6.5] ]
       
       ocr.ReadImage(self.path0)       
       ocr.GetCutIndexs()
       angles = ocr.GetRotations()       
       self.assertEquals(angles,angle_GT[0])
              
       ocr.ReadImage(self.path1)       
       ocr.GetCutIndexs()
       angles = ocr.GetRotations()       
       self.assertEquals(angles,angle_GT[1])

       ocr.ReadImage(self.path2)       
       ocr.GetCutIndexs()
       angles = ocr.GetRotations()       
       self.assertEquals(angles,angle_GT[2])

if __name__ == '__main__':
   unittest.main()