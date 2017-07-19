import matplotlib.pyplot as plt
from TrainCharacterExtractor import TrainCharacterExtractor
import Data
 
folder_character = '../data/character/'

tce = TrainCharacterExtractor()
tce.isFigure = True

tce.Read(Data.path1)
tce.GetRotationAngle()
tce.GetGrid()
tce.ShowGridBorder()
tce.SaveGridCharacter(folder_character)
plt.show()