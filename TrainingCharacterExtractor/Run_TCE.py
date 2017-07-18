import matplotlib.pyplot as plt
from TrainCharacterExtractor import TrainCharacterExtractor
import Data
 
tce = TrainCharacterExtractor()
tce.isFigure = True

tce.Read(Data.path1)
tce.GetRotationAngle()
tce.GetGrid()
tce.GetGridCell()
plt.show()