from util.data_reader import DataReader
import numpy as np

trainData = DataReader();

def Test_ReadFolder():
    trainIn,trainOut,testIn,testOut = trainData.GetData()
    print ('trainIn',trainIn.shape) 
    print ('testIn',testIn.shape) 
    print (trainOut)
    print ('trainIn')
    compareIndex = 0
    print (trainIn[compareIndex])
    print ('testIn')
    print (testIn[compareIndex])
    print ('diff', np.mean(np.abs(trainIn[compareIndex] - testIn[compareIndex])))

Test_ReadFolder()