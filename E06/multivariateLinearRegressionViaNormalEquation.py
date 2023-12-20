#Starter Code for multivariate regression with train and test
#Art White, February, 2022
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split

def hypothesis(thetas,Xrow):
    dependentY=0
    for idx in range(len(thetas)):
        dependentY+=thetas[idx]*Xrow[idx]
    return round(dependentY,0)

def getListFeatures():
    # Modeled after https://stackoverflow.com/questions/464864/how-to-get-all-possible-combinations-of-a-list-s-elements
    pFeatures = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated']
    cFeatures = []
    for idx in range(3, 7):
        combination = itertools.combinations(pFeatures, idx)
        for c in combination:
            cFeatures.append(c)
    return cFeatures

def main(xFeatureList):
    # Read data from file using pandas and create a dataframe
    housingDF = pd.read_csv('housing.csv')
    northBendHousingDF = housingDF  # [housingDF['city'] == 'North Bend']
    
    # Subdivide the data into features (Xs) and dependent variable (y) dataframes
    XsDF = northBendHousingDF[xFeatureList]
    YDF = northBendHousingDF['price']
    
    #Convert dataframes to numpy ndarray(matrix) types
    Xs=XsDF.to_numpy()
    Y=YDF.to_numpy()
    
    #Add the 1's column to the Xs matrix (1 * the intercept values, right?)
    XsRows,XsCols=Xs.shape
    X0 = np.ones((XsRows,1))
    Xs = np.hstack((X0,Xs))
    
    #Calc the Thetas via the normal equation
    thetas=(np.linalg.pinv(Xs.T @ Xs)) @ Xs.T @ Y
    #print(thetas)
    
    #Now, generate differences from the predicted
    predictedM=(Xs @ thetas.T)
    diffs=abs(predictedM-Y)
    sumOfDiffs=diffs.sum()
    sumOfPrices=Y.sum()
    #print("average price difference for training values:",str(round(sumOfDiffs/sumOfPrices*100,4))+"%")
    data.append([round(sumOfDiffs / sumOfPrices * 100, 4), xFeatureList])

def splitTest(xFeatureList):
    housingDF = pd.read_csv('housing.csv')
    northBendHousingDF = housingDF
    XsDF = northBendHousingDF[xFeatureList]
    YDF = northBendHousingDF['price']

    X_train, X_test, Y_train, Y_test = train_test_split(XsDF, YDF, test_size=0.7, random_state=3121)

    Xs = X_train.to_numpy()
    Y = Y_train.to_numpy()
    
    Xtest = X_test.to_numpy()
    Ytest = Y_test.to_numpy()

    # Add the 1's column to the Xs matrix (1 * the intercept values, right?)
    XsRows, XsCols = Xs.shape
    X0 = np.ones((XsRows, 1))
    Xs = np.hstack((X0, Xs))
    
    Xr, Xc = Xtest.shape
    Xt0 = np.ones((Xr, 1))
    Xtest = np.hstack((Xt0, Xtest))

    # Calc the Thetas via the normal equation
    thetas = (np.linalg.pinv(Xs.T @ Xs)) @ Xs.T @ Y
    # print(thetas)

    # Now, generate differences from the predicted
    predictedM = (Xtest @ thetas.T)
    diffs = abs(predictedM - Ytest)
    sumOfDiffs = diffs.sum()
    sumOfPrices = Ytest.sum()
    # print("average price difference for training values:",str(round(sumOfDiffs/sumOfPrices*100,4))+"%")
    data.append([round(sumOfDiffs / sumOfPrices * 100, 4), xFeatureList])

data = []
'''
xList = getListFeatures()
for x in xList:
    main(np.asarray(x))
resultdf = pd.DataFrame(data, columns = ["avg_price_dif","features"])
resultdf.sort_values(by=['avg_price_dif']).to_csv('./all_feature_combos.csv')
'''

h1 = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'view', 'yr_built']
h2 = ['bedrooms', 'sqft_living', 'floors', 'view', 'condition', 'yr_built']
h3 = ['bedrooms', 'sqft_living', 'floors', 'waterfront', 'view', 'yr_built']
h = [h1, h2, h3]

for hyp in h:
    splitTest(hyp)
print(f"Multivariate Model Test Error and Feature List:\n {data}")
data = []
main(['sqft_living'])
print(f"Univariate Model and Feature:\n {data}")