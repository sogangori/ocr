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

   def test_D(self):
       #self.assertEqual(3,3)
       ocr = OCR_Pre()
       ocr.isFigure = False
       cut0 = 1
       cut1 = 1
       cut2 = 0
       ocr.ReadImage(self.path0)
       
       cut = ocr.GetCutIndexs()
       self.assertEquals(cut,cut0)
       
       ocr.ReadImage(self.path1)
       cut = ocr.GetCutIndexs()
       self.assertEquals(cut,cut1)
       
       ocr.ReadImage(self.path2)
       cut = ocr.GetCutIndexs()
       self.assertEquals(cut,cut2)   

if __name__ == '__main__':
   unittest.main()