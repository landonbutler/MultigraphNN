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

def epsilon_models(models, datasets, Es):
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

    savedData = {'Models': [],
                'Iterations': [],
                'Epsilon': [],
                'Best': [],
                'Last': [],
                'Time Taken': []
                }
    savedModels = {}
    startTime = datetime.now()
    
    for modelName in models.keys():
      for i, dataset in enumerate(datasets):
        for e in range(len(Es)):
          print(f'Model: {modelName}, Dataset: {i}, E {e}')
          t0 = time.time()
          S_user, S_genre, xTrain, yTrain, xValid, yValid, xTest, yTest, idxTrainMovie = dataset
          E = Es[e][i] if models[modelName]['GSOs'] == 1 else Es[e][i] / 2
          S_user = S_user + E @ S_user + S_user @ E
          S_genre = S_genre + E @ S_genre + S_genre @ E
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
            arch = MultiGraphNeuralNetwork(GSOs, depths = [2], fs = fs, readout = readout, nonlinearity = nonlinearity, idxTrainMovie = idxTrainMovie, penaltyMultiplier=0, device = device).to(device)
          elif modelName == 'MultigraphILNN':
            arch = MultiGraphNeuralNetwork(GSOs, depths = [2], fs = fs, readout = readout, nonlinearity = nonlinearity, idxTrainMovie = idxTrainMovie, penaltyMultiplier=0.5, device = device).to(device)
          else:
            break

          testBest, testLast, bestModel, lastModel = trainingLoop(modelName, arch, lossFunction, xTrain, yTrain, xValid, yValid, xTest, yTest, device, nEpochs = nEpochs, learningRate = lr, verbose = False, batchSize = batchSize, evalFunction=evalFunction)
        
          print(f'\t Best: {testBest}  Last: {testLast}, Time: {time.time() - t0} sec')
          savedData['Models'].append(modelName)
          savedData['Iterations'].append(i)
          savedData['Best'].append(testBest.item())
          savedData['Last'].append(testLast.item())
          savedData['Epsilon'].append(e)
          savedData['Time Taken'].append(time.time() - t0)
          savedModels[(modelName, i, e)] = (bestModel, lastModel)
          torch.cuda.empty_cache()
    
          fn = 'results/stabilityEpsilons/' + str(startTime) + '.csv'
          pd.DataFrame(savedData).to_csv(fn)

          fn = 'models/stabilityEpsilons/' + str(startTime) + '.pkl'
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

    # Specify which Es
    n = datasets[0][0].shape[0]
    epsilons = np.logspace(-3, 0, num=10)
    np.random.seed(0)
    Es = []
    for eps in epsilons:
        Es.append([np.random.uniform((1-eps) * eps, eps, size = n) for _ in range(len(datasets))])

    # Run epsilon models script
    baseline_models(models, datasets, Es)