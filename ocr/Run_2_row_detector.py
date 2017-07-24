import matplotlib.pyplot as plt
from util.OCR_Pre import OCR_Pre
import Data
import util.Detector as Detector
from PIL import Image

index_figure=0
src_img,src_arr = Detector.Read(Data.path_cut_0)
src_cut = Detector.CutPadding(src_arr)

src_cut_grid = Detector.RemoveLines(src_cut, axis=1)
font_offsets = Detector.Detect_Row(src_cut_grid)
src_cut_grid_show = Detector.DrawGrid(src_cut_grid, font_offsets)
Detector.GetCandidateRows(src_cut_grid, font_offsets, Data.path_rows)

plt.figure(0)            
plt.imshow(Image.fromarray(src_cut_grid_show) , cmap = plt.get_cmap('gray'))        
plt.title('count' + str(len(font_offsets)))
plt.grid(which='both', axis='both')

plt.show()
