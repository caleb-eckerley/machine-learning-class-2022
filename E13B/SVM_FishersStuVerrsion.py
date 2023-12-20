# From https://datascience.stackexchange.com/questions/26640/how-to-check-for-overfitting-with-svm-and-iris-data
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

iris = load_iris()
X = iris.data[:, :4]
y = iris.target


def createModel(kernel, C_value):
  return svm.SVC(kernel=kernel, C=C_value, gamma='auto', probability=True)


def createPolyModel(degree, C_value):
  return svm.SVC(kernel='poly', C=C_value, gamma='auto', probability=True, degree=degree)


def getScores(svm_model):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  svm_model.fit(X_train, y_train)
  
  predictions = svm_model.predict(X_train)
  train_result = accuracy_score(predictions, y_train)
  
  predictions = svm_model.predict(X_test)
  test_result = accuracy_score(predictions, y_test)
  return train_result, test_result
  

def getIterationScores(model, numIterations):
  TrScore_arr = []
  TeScore_arr = []
  for iteration in range(numIterations):
    train_results, test_results = getScores(model)
    TrScore_arr.append(train_results)
    TeScore_arr.append(test_results)
  return TrScore_arr, TeScore_arr
  
  
def appendData(trainScoreArray, trainValue, testScoreArray, testValue, cArray, cValue, kernelArray, kernelValue, degreeArray, degree='null'):
  trainScoreArray.append(
    np.average(trainValue)
  )
  testScoreArray.append(
    np.average(testValue)
  )
  cArray.append(cValue)
  kernelArray.append(kernelValue)
  degreeArray.append(degree)
  return trainScoreArray, testScoreArray, cArray, kernelArray, degreeArray


def getResult():
  kernels = ['rbf', 'linear', 'poly']
  degrees = [2, 3, 4]
  C_values = [0.5, 1, 1.5, 2]

  C_arr = []
  kernel_arr = []
  degree_arr = []
  Train_Score_Av_arr = []
  Test_Score_Av_arr = []
  for kernel in kernels:
    for C_value in C_values:
      
      if kernel == 'poly':
        for degree in degrees:
          model = createPolyModel(degree, C_value)
          trainIter, testIter = getIterationScores(model, 30)
          
          Train_Score_Av_arr.append(np.average(trainIter))
          Test_Score_Av_arr.append(np.average(testIter))
          C_arr.append(C_value)
          kernel_arr.append(kernel)
          degree_arr.append(degree)
      else:
        model = createModel(kernel, C_value)
        trainIter, testIter = getIterationScores(model, 30)

        Train_Score_Av_arr.append(np.average(trainIter))
        Test_Score_Av_arr.append(np.average(testIter))
        C_arr.append(C_value)
        kernel_arr.append(kernel)
        degree_arr.append("")
  result_dict = dict(zip(
    ['C', 'Kernel', 'Degree', 'Apparent Score Average', 'Test Score Average'],
    [C_arr, kernel_arr, degree_arr, Train_Score_Av_arr, Test_Score_Av_arr]
  ))
  return pd.DataFrame(result_dict)


df = getResult()
df.to_csv("./SVM_Output")
print(df)
