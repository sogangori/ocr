import unittest
from TrainCharacterExtractor import TrainCharacterExtractor
import Data

class Test_TrainingData(unittest.TestCase):
   
   def test_rotate_angle(self):
       cls = TrainCharacterExtractor()
       gt = Data.angle_GT
       
       cls.Read(Data.path0)       
       angle = cls.GetRotationAngle()       
       self.assertEquals(angle,gt[0])       

       cls.Read(Data.path1)       
       angle = cls.GetRotationAngle()       
       self.assertEquals(angle,gt[1])       

   def test_grid(self):
       cls = TrainCharacterExtractor()
       gt = Data.grid_GT
       
       cls.Read(Data.path0)       
       cls.GetRotationAngle()  
       cls.GetGrid()          
       self.assertEquals(len(cls.font_offsets_x),gt[0][1])
       self.assertEquals(len(cls.font_offsets_y),gt[0][0])

       cls.Read(Data.path1)       
       cls.GetRotationAngle()  
       cls.GetGrid()          
       self.assertEquals(len(cls.font_offsets_x),gt[1][1])
       self.assertEquals(len(cls.font_offsets_y),gt[1][0])
    
   def test_cell(self):
       cls = TrainCharacterExtractor()
       gt = Data.cell_GT
       
       cls.Read(Data.path0)       
       cls.GetRotationAngle()  
       cls.GetGrid()         
       
       self.assertEquals(cls.cell_hw,gt[0])

       cls.Read(Data.path1)       
       cls.GetRotationAngle()  
       cls.GetGrid()         
       
       self.assertEquals(cls.cell_hw,gt[1])

if __name__ == '__main__':
   unittest.main()