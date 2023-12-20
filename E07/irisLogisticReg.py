import math
import random
import sys
import numpy as np

def readInData(fileName):
    inFile = open(fileName, 'r')
    for rep in range(6):inFile.readline()
    m = 0
    xList = []
    yList = []
    #xList will include features and y value AND leading 1 for multiplying theta0
    for line in inFile:
        lineList = line.strip().split()[1:] #leave off sample number
        lineList = [1]+[float(val) for val in lineList] #include multiplier for theta0
        xList.append(lineList)
        m += 1
    inFile.close()
    #randomize order
    random.shuffle(xList)
    #split off y's and translate setosas (0's) into 1's and the other two (1's and 2's) into 0's
    xLate=[0,1,0]
    for line in xList:
        yList.append(xLate[int(line.pop())])
    return xList, yList, m

def hOfX(x,thetas):
    z = 0
    for i in range(0,len(x)):
        z += x[i]*thetas[i]
    h = 1/(1+math.exp(-z))
    return h

def costDeriv(xList,yList,thetas,diffsTheta,m):
    for i in range(m):
        temp = hOfX(xList[i], thetas) - yList[i]
        for t in range(len(thetas)):
            diffsTheta[t] += temp*xList[i][t]
    return diffsTheta

def gradientDescent(xList,yList,m,maxIters,alpha):
    thetas = [0,0,0,0,0]
    iters = 1
    while iters <= maxIters: 
        diffsTheta = [0,0,0,0,0]
        diffsTheta=costDeriv(xList,yList,thetas,diffsTheta,m)
        for t in range(len(thetas)):
            thetas[t] = thetas[t] - (alpha * (1/m) * diffsTheta[t])
        if iters % 1000 == 0:
            print(iters)
        iters += 1
    return thetas

def testResults(thetaList,xList,yList):
    errorCount=0
    #Your code here both showing individual tests and calculating apparent error rate
    xArray = np.array(xList)
    yArray = np.array(yList)
    tArray = np.array(thetaList)
    print('calculated y: \t Predicted Classification: \t Actual Classification: ')
    for case in range(len(xList)):
        yHat = hOfX(xList[case], thetaList)
        predClass = 0
        if yHat >= 0.5:
            predClass = 1
        print(f'{yHat}\t {predClass}\t {yArray[case]}')
        if predClass != yArray[case]:
            errorCount += 1
    print("Errors",errorCount/150)
        
def main():
    xList, yList, m = readInData('normScaledFishersIris.txt')
    thetaList = gradientDescent(xList,yList,m,10000,.01)
    print("Predictive model: ", thetaList[0], "+", thetaList[1], "*(x1)", "+", thetaList[2], "*(x2)", "+", thetaList[3], "*(x3)", "+", thetaList[4], "*(x4)")
    testResults(thetaList,xList,yList)

main()
