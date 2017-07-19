import matplotlib.pyplot as plt
from TrainCharacterExtractor import TrainCharacterExtractor
import Data
 
tce = TrainCharacterExtractor()
tce.isFigure = True

tce.Read(Data.path0)
tce.GetRotationAngle()
tce.GetGrid()
tce.ShowGridBorder()
plt.show()