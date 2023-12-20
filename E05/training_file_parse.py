import pandas as pd

def readIn(fileName, delimiter):
  keys = ['datasetName', 'numberNonData', 'numberFeature', 'numberPredict', 'headerRow', 'dataDocumentation']
  file = open(fileName, "r")
  metaData = readMetaData(file, keys)
  fileData = readData(file, delimiter)
  file.close()
  df = pd.DataFrame(data=fileData, columns=metaData.get('parsedHeader'))
  print(f'Original Dataset:\n{df}\n')
  featureColumns = getFeatureColumns(metaData)
  df[featureColumns] = df[featureColumns].apply(pd.to_numeric)
  return metaData, df


def getFeatureColumns(metaData):
  listOfColumnIdx = metaData.get('parsedHeader')[
                    int(metaData.get('numberNonData')):
                    int(metaData.get('numberNonData')) + int(metaData.get('numberFeature'))
                    ]
  return listOfColumnIdx


def readMetaData(file, keys):
  fileDataDictionary = {}
  for key in keys:
    if key == 'headerRow':
      line = file.readline()
      fileDataDictionary['parsedHeader'] = readHeaders(line)
      fileDataDictionary[key] = line
    else:
      fileDataDictionary[key] = file.readline()
  return fileDataDictionary


def readHeaders(line):
  return line.strip('\n').split(',')


def readData(file, delimiter):
  stagingList = []
  for line in file:
    stagingList.append(line.strip('\n').split(delimiter))
  return stagingList


def normalize(metaData, data, nDec):
  for feature in getFeatureColumns(metaData):
    fRange = data[feature].max() - data[feature].min()
    fMean = data[feature].mean()
    for i in range(len(list(data[feature]))):
      x = data[feature][i]
      x = (x - fMean)/fRange
      data.at[i, feature] = round(x, nDec)
  return data


def writeOutput(data, metaData, output, delimiter):
  keys = ['datasetName', 'numberNonData', 'numberFeature', 'numberPredict', 'headerRow', 'dataDocumentation']
  f = open(output, 'w')
  for key in keys:
    f.write(metaData.get(key))
  f.close()
  data.to_csv(output, header=None, index=None, sep=delimiter, mode='a')


def main(testFile, outputFile='./dummyOut.txt', delimiter=' ', numberOfDecimalPlaces=4):
  metaData, df = readIn(testFile, delimiter)
  data = normalize(metaData, df, numberOfDecimalPlaces)
  print(f'Modified Dataset:\n {data}')
  writeOutput(data, metaData, outputFile, delimiter)
  

main(testFile='./test/FishersIrisStandardHeader.txt',
     delimiter=',',
     outputFile='FishersOutput.txt',
     numberOfDecimalPlaces=6
     )

main(testFile='./test/dummyTest.txt',
     delimiter=' ',
     outputFile='dummyOut.txt',
     numberOfDecimalPlaces=6
     )
