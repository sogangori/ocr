from PIL import Image
import PIL.ImageOps 
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

class TrainCharacterExtractor():
    def __init__(self):
        print ("TrainCharacterExtractor __init__")

    def Read(self, path):
        img = Image.open(path)
        grayImg = img.convert('L')
        self.grayImg = PIL.ImageOps.invert(grayImg)
