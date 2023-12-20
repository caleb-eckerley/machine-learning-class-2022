import numpy as np
import math
import random


class Node:
  def __init__(self, left=random.choice([-1, 1]), right=random.choice([-1, 1]), bias=random.choice([-1, 1])):
    self.lWeight = left
    self.rWeight = right
    self.bias = bias
    self.lastOut = 0
    
  def getOutput(self, lInput, rInput):
    n = (lInput * self.lWeight) + (rInput * self.rWeight) + self.bias
    output = 1/(1+math.exp(-n))
    self.lastOut = output
    return output
  
  def setLWeight(self, weight):
    self.lWeight = weight
  
  def setRWeight(self, weight):
    self.rWeight = weight
    
  def setBias(self, bias):
    self.bias = bias
    
  def getlWeight(self):
    return self.lWeight
  
  def getrWeight(self):
    return self.rWeight
  
  def getBias(self):
    return self.bias
  
  def getLastOutput(self):
    return self.lastOut


def getNewNetwork():
  hiddenLayer = [Node(), Node()]
  outputLayer = [Node(), Node()]
  network = [hiddenLayer, outputLayer]
  return network


def forwardProp(network, lInput, rInput):
  hiddenLOut = network[0][0].getOutput(lInput, rInput)
  hiddenROut = network[0][1].getOutput(lInput, rInput)
  
  tNodeOut = network[1][0].getOutput(hiddenLOut, hiddenROut)
  fNodeOut = network[1][1].getOutput(hiddenLOut, hiddenROut)
  return tNodeOut, fNodeOut


def calcOutputErr(node, trueVal):
  outputErr = node.getLastOutput() * (1 - node.getLastOutput()) * (trueVal - node.getLastOutput())
  return outputErr


def calcHiddenErr(node, tErr, fErr, lNodeWeight, rNodeWeight):
  hiddenErr = node.getLastOutput() * (1 - node.getLastOutput()) * ((tErr * lNodeWeight) + (fErr * rNodeWeight))
  return hiddenErr


def backProp(network, trueT, trueF, lIn, rIn, learnRate):
  outputTErr = calcOutputErr(network[1][0], trueT)
  outputFErr = calcOutputErr(network[1][1], trueF)
  
  hiddenLErr = calcHiddenErr(network[0][0], outputTErr, outputFErr, network[1][0].getlWeight(), network[1][1].getlWeight())
  hiddenRErr = calcHiddenErr(network[0][1], outputTErr, outputFErr, network[1][0].getrWeight(), network[1][1].getrWeight())
  
  network[1][0].setBias(network[1][0].getBias() + (outputTErr * learnRate))
  network[1][0].setLWeight(network[1][0].getlWeight() + (outputTErr * network[0][0].getLastOutput() * learnRate))
  network[1][0].setRWeight(network[1][0].getrWeight() + (outputTErr * network[0][1].getLastOutput() * learnRate))
  
  network[1][1].setBias(network[1][1].getBias() + (outputFErr * learnRate))
  network[1][1].setLWeight(network[1][1].getlWeight() + (outputFErr * network[0][0].getLastOutput() * learnRate))
  network[1][1].setRWeight(network[1][1].getrWeight() + (outputFErr * network[0][1].getLastOutput() * learnRate))
  
  network[0][0].setBias(network[0][0].getBias() + (hiddenLErr * learnRate))
  network[0][0].setLWeight(network[0][0].getlWeight() + (hiddenLErr * lIn * learnRate))
  network[0][0].setRWeight(network[0][0].getrWeight() + (hiddenLErr * rIn * learnRate))
  
  network[0][1].setBias(network[0][1].getBias() + (hiddenRErr * learnRate))
  network[0][1].setLWeight(network[0][1].getlWeight() + (hiddenRErr * lIn * learnRate))
  network[0][1].setRWeight(network[0][1].getrWeight() + (hiddenRErr * rIn * learnRate))
  
  return


def runEpoch(network, cases, realResults, learnRate):
  for case in range(len(cases)):
    runCaseForwardBackward(network, cases[case], realResults[case], learnRate)
  return
  
  
def runCaseForwardBackward(network, case, realResult, learnRate):
  forwardProp(network, case[0], case[1])
  backProp(network, realResult[0], realResult[1], case[0], case[1], learnRate)
  return network


def train(network, cases, realResults, numberOfLoops, learnRate=1.0):
  for numRuns in range(numberOfLoops + 1):
    runEpoch(network, cases, realResults, learnRate)
    if numRuns % 10 == 0 and numRuns != 0:
      print(f"Epoch {numRuns}")
  return


def runXOR(network, cases, realResults):
  for caseIndex in range(len(cases)):
    result = False
    tHat, fHat = forwardProp(network, cases[caseIndex][0], cases[caseIndex][1])
    if tHat > fHat:
      result = True
    print(f"Input {cases[caseIndex]} {result}")
    print(f"Expected \t True {realResults[caseIndex][0]} Output: {round(network[1][0].getLastOutput(), 3)}")
    print(f"Expected \t False {realResults[caseIndex][1]} Output: {round(network[1][1].getLastOutput(), 3)}")
  return

  
def main():
  network = getNewNetwork()
  cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
  trueFalseCaseResults = [[0, 1], [1, 0], [1, 0], [0, 1]]
  # lRate has a default value of 1 unless otherwise specified below
  train(network, cases, trueFalseCaseResults, 10000)
  print("AFTER TRAINING")
  runXOR(network, cases, trueFalseCaseResults)
  print("Done")
  
  
main()
