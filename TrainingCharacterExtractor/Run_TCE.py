import matplotlib.pyplot as plt
from TrainCharacterExtractor import TrainCharacterExtractor
import Data
import glob

tce = TrainCharacterExtractor()
tce.isFigure = True

list_path = glob.glob(Data.folder_trainSet)                 
count = len(list_path)        
print ('ReadFolder() count', len(list_path))
tce.Read(list_path[1])
tce.GetRotationAngle()
tce.GetGrid()
#tce.ShowGridBorder()

tce.GetCharacters()
#tce.SaveCharacters(Data.folder_character)
plt.show()