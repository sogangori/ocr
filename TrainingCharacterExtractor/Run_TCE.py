import matplotlib.pyplot as plt
from TrainCharacterExtractor import TrainCharacterExtractor
import Data
import glob

data_index = 0
tce = TrainCharacterExtractor()
tce.isFigure = True
list_path = glob.glob(Data.folder_trainSet)                 
count = len(list_path)        
print ('ReadFolder() count', len(list_path))
tce.Read(list_path[data_index])
tce.GetRotationAngle()
tce.GetGrid()
tce.GetCharacters()
tce.SaveCharacters(Data.folder_character+str(data_index))
plt.show()