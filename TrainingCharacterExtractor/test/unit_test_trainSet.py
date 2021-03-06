﻿import unittest
import glob
from TrainCharacterExtractor import TrainCharacterExtractor
import Data

class Test_TrainingData(unittest.TestCase):
   
   def test_rotate_angle(self):
       cls = TrainCharacterExtractor()
       gt = Data.angle_GT
       
       cls.Read(Data.path0)       
       angle = cls.GetRotationAngle()              
       self.assertAlmostEquals(angle,gt[0])              

       cls.Read(Data.path1)       
       angle = cls.GetRotationAngle()       
       self.assertAlmostEquals(angle,gt[1])       

   def test_grid(self):
       cls = TrainCharacterExtractor()
       gt = Data.grid_GT
       
       cls.Read(Data.path0)       
       cls.GetRotationAngle()  
       cls.GetGrid()          
       self.assertEquals(len(cls.font_offsets_x)-1,gt[0][1])#51,50
       self.assertEquals(len(cls.font_offsets_y)-1,gt[0][0])#31,32

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

   def test_character_count(self, path):
       cls = TrainCharacterExtractor()
              
       list_path = glob.glob(Data.folder_trainSet)                               
       cls.Read(list_path[1])     
       cls.GetRotationAngle()  
       cls.GetGrid()         
       char_count = cls.GetCharacters()    
       self.assertEquals(char_count,2350)

   def test_character_counts(self):
       
       list_path = glob.glob(Data.folder_trainSet) 
                                
       self.test_character_count(list_path[0])

if __name__ == '__main__':
   unittest.main()