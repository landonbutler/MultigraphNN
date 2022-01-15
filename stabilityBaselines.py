import os, zipfile, pickle
import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import time
from datetime import datetime

from utils.utils import RMSE
from utils.trainingLoop import trainingLoop

from architectures.Linear import Linear
from architectures.GraphNeuralNetwork import GraphNeuralNetwork
from architectures.MultiGraphNeuralNetwork import MultiGraphNeuralNetwork


def baseline_models(models, datasets):
    lossFunction = nn.SmoothL1Loss()
    evalFunction = RMSE
    lr = 0.005
    nEpochs = 80
    batchSize = 50
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    nonlinearity = torch.nn.Sigmoid()
    scores = {}
    readout = [16,1]
    fs = [1,8]
    ks = [3]
    saveFilename = 'results/stabilityBaselines/baselineModels.csv'

    savedData = {'Models': [],
                'Iterations': [], 
                'Best': [],
                'Last': [],
                'Time Taken': []}
    savedModels = {}
    
    for modelName in models.keys():
      for i, dataset in enumerate(datasets):
        print(f'Model: {modelName}, Dataset: {i}')
        t0 = time.time()
        S_user, S_genre, xTrain, yTrain, xValid, yValid, xTest, yTest, idxTrainMovie = dataset

        GSOs = [S_user] if models[modelName]['GSOs'] == 1 else [S_user, S_genre]
        if modelName == 'Linear':
          arch = Linear(GSOs, ks = ks, fs = [1,1], f_edge = 1, idxTrainMovie = idxTrainMovie, device = device).to(device)
        elif modelName == 'SimpleGNN':
          arch = GraphNeuralNetwork(GSOs, ks = ks, fs = fs, f_edge = 1, readout = readout, nonlinearity = nonlinearity, idxTrainMovie = idxTrainMovie, penaltyMultiplier=0, device = device).to(device)
        elif modelName == 'SimpleILGNN':
          arch = GraphNeuralNetwork(GSOs, ks = ks, fs = fs, f_edge = 1, readout = readout, nonlinearity = nonlinearity, idxTrainMovie = idxTrainMovie, penaltyMultiplier=0.5, device = device).to(device)
        elif modelName == 'MultiChannelGNN':
          arch = GraphNeuralNetwork(GSOs, ks = ks, fs = fs, f_edge = 2, readout = readout, nonlinearity = nonlinearity, idxTrainMovie = idxTrainMovie, penaltyMultiplier=0, device = device).to(device)
        elif modelName == 'MultiChannelILGNN':
          arch = GraphNeuralNetwork(GSOs, ks = ks, fs = fs, f_edge = 2, readout = readout, nonlinearity = nonlinearity, idxTrainMovie = idxTrainMovie, penaltyMultiplier=0.5, device = device).to(device)
        elif modelName == 'MultigraphNN':
          arch = MultiGraphNeuralNetwork(GSOs, depths = [2], fs = fs, readout = readout, nonlinearity = nonlinearity, idxTrainMovie = idxTrainMovie, penaltyMultiplier=0).to(device)
        elif modelName == 'MultigraphILNN':
          arch = MultiGraphNeuralNetwork(GSOs, depths = [2], fs = fs, readout = readout, nonlinearity = nonlinearity, idxTrainMovie = idxTrainMovie, penaltyMultiplier=0.5).to(device)
        else:
          break

        testBest, testLast, bestModel, lastModel = trainingLoop(modelName, arch, lossFunction, xTrain, yTrain, xValid, yValid, xTest, yTest, device, nEpochs = nEpochs, learningRate = lr, verbose = False, batchSize = batchSize, evalFunction=evalFunction)
        
        print(f'\t Best: {testBest}  Last: {testLast}, Time: {time.time() - t0} sec')
        savedData['Models'].append(modelName)
        savedData['Iterations'].append(i)
        savedData['Best'].append(testBest.item())
        savedData['Last'].append(testLast.item())
        savedData['Time Taken'].append(time.time() - t0)
        savedModels[(modelName, i)] = (bestModel, lastModel)
        torch.cuda.empty_cache()
    
    fn = 'results/stabilityBaselines/' + str(datetime.datetime.now()) + '.csv'
    pd.DataFrame(savedData).to_csv(fn)

    fn = 'models/stabilityBaselines/' + str(datetime.datetime.now()) + '.csv'
    with open(fn, 'wb') as handle:
      pickle.dump(savedModels, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    # Specify which models
    models = {}
    models['Linear'] = {'GSOs': 1}
    models['SimpleGNN'] = {'GSOs': 1}
    models['SimpleILGNN'] = {'GSOs': 1}
    models['MultiChannelGNN'] = {'GSOs': 2}
    models['MultiChannelILGNN'] = {'GSOs': 2}
    models['MultigraphNN'] = {'GSOs': 2}
    #models['MultigraphILNN'] = {'GSOs': 2}

    # Specify which datasets
    zipfile.ZipFile('data/datasets.zip').extractall('data') # create zipfile object
    with open('data/datasets.pkl', 'rb') as handle:
      datasets = pickle.load(handle)
    os.remove('data/datasets.pkl') # delete pkl file

    # Run baseline models script
    baseline_models(models, [datasets[0]])