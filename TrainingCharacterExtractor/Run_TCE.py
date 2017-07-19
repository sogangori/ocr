import matplotlib.pyplot as plt
from TrainCharacterExtractor import TrainCharacterExtractor
import Data

tce = TrainCharacterExtractor()
tce.isFigure = True

tce.Read(Data.path1)
tce.GetRotationAngle()
tce.GetGrid()
tce.ShowGridBorder()

tce.SaveGridCharacter(Data.folder_character)
plt.show()