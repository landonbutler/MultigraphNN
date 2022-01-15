import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy


# returns  evalTestBest,  evalTestLast, bestModel, lastModel
def trainingLoop(arch_type, architecture, lossFunction, xTrain, yTrain, xValid, yValid, xTest, yTest, device, sparse = False, nEpochs = 200, learningRate = 0.05, batchSize = 20, classes = None, verbose = True, evalFunction = None): 
    xTrain = xTrain.to(device)
    yTrain = yTrain.to(device)
    xValid = xValid.to(device)
    yValid = yValid.to(device)
    xTest = xTest.to(device)
    yTest = yTest.to(device)
    
    if evalFunction is None:
      evalFunction = lossFunction
    loss = lossFunction
    validationInterval = 7 if verbose else 30 # after how many steps to print training loss

    nTrain = xTrain.shape[0]
    optimizer = optim.Adam(architecture.parameters(), lr=learningRate)

    if nTrain < batchSize:
        nBatches = 1
        batchSize = [nTrain]
    elif nTrain % batchSize != 0:
        nBatches = np.ceil(nTrain/batchSize).astype(np.int64)
        batchSize = [batchSize] * nBatches
        while sum(batchSize) != nTrain:
            batchSize[-1] -= 1
    else:
        nBatches = np.int(nTrain/batchSize)
        batchSize = [batchSize] * nBatches
    batchIndex = np.cumsum(batchSize).tolist()
    batchIndex = [0] + batchIndex

    epoch = 0 # epoch counter

    # Store the training...
    lossTrain = dict()
    # ...and test variables
    lossTestBest = dict()
    lossTestLast = dict()

    bestModel = dict()

    lossTrain = []
    lossTrain_whole = []
    lossValid = [] # initialize list to store the validation losses and keep track 
                   # of the best model
    evalValid = []
    xAxisValid = []
    t0 = time.time()
    while epoch < nEpochs:

        randomPermutation = np.random.permutation(nTrain)
        idxEpoch = [int(i) for i in randomPermutation]

        batch = 0 

        while batch < nBatches:
            # Determine batch indices
            thisBatchIndices = idxEpoch[batchIndex[batch] : batchIndex[batch+1]]
            # Get the samples in this batch
            xTrainBatch = xTrain[thisBatchIndices,:]
            yTrainBatch = yTrain[thisBatchIndices,:]

            # Reset gradients
            architecture.zero_grad()

            # Obtain the output of the architectures
            
            yHatTrainBatch = architecture(xTrainBatch)
            lossValueTrain = loss(yHatTrainBatch, yTrainBatch)

            # Compute gradients
            if architecture.penaltyMultiplier > 0:
              lossValueTrain = lossValueTrain + architecture.penaltyMultiplier * architecture.ILconstant()
            lossValueTrain.backward()

            # Optimize
            optimizer.step()

            lossTrain += [lossValueTrain.item()]
            if (epoch * nBatches + batch) % validationInterval == 0:
                if verbose:
                  with torch.no_grad():
                      yHatTrain = architecture(xTrain)
                      if architecture.penaltyMultiplier > 0:
                        lossTrain_whole+=[loss(yHatTrain, yTrain).item() + architecture.penaltyMultiplier * architecture.ILconstant(skipCalc = True)]
                      else:
                        lossTrain_whole+=[loss(yHatTrain, yTrain).item()]

                
                xAxisValid.append(epoch * nBatches + batch)
                # Obtain the output of the GNN for the validation set
                # without computing gradients
                with torch.no_grad():
                    yHatValid = architecture.forward(xValid)
                    lossValueValid = loss(yHatValid, yValid)
                    if architecture.penaltyMultiplier > 0:
                      lossValueValid = lossValueValid + architecture.penaltyMultiplier * architecture.ILconstant(skipCalc = True)
                    evalValueValid = evalFunction(yHatValid, yValid)
                    evalValueTest = evalFunction(architecture.forward(xTest), yTest)
                    

                # Compute validation loss and save it
                lossValueValid = lossValueValid.item()
                lossValid += [lossValueValid]
                evalValid += [evalValueValid.item()]

                # Print training and validation loss  
                if verbose and (epoch * nBatches + batch) % validationInterval == 0:
                    print("")
                    print("    (E: %2d, B: %3d)" % (epoch+1, batch+1), end = ' ')
                    print("")

                    print("\t Loss: %6.4f [T]" % (lossValueTrain.item()) + " %6.4f [V]" % (
                                    lossValueValid) + " %6.4f [Eval]" % (evalValueValid.item()) + " %6.4f [Test]" % (evalValueTest.item()))
            
                # Save the best model so far 
                if len(lossValid) > 1:
                    if lossValueValid <= min(lossValid):
                        bestModel =  copy.deepcopy(architecture)
                else:
                    bestModel =  copy.deepcopy(architecture)
            batch +=1
        #print(time.time() - t0)
        t0 = time.time()
        epoch+=1

    print("")


    ################################
    ########## EVALUATION ##########
    ################################
    

    # Testing last model
    with torch.no_grad():
        yHatValid = architecture.forward(xValid)
        lossValueValid = loss(yHatValid, yValid)

        evalValueValid = evalFunction(yHatValid, yValid)
        yHatTest = architecture.forward(xTest)
        lossTestLast = loss(yHatTest, yTest)
        yHatTrain = architecture(xTrain)
        if architecture.penaltyMultiplier > 0:
          lossValueValid = lossValueValid + architecture.penaltyMultiplier * architecture.ILconstant(skipCalc = True)
          lossTestLast = lossTestLast + architecture.penaltyMultiplier * architecture.ILconstant(skipCalc = True)
          lossTrain_whole+=[loss(yHatTrain, yTrain).item() + architecture.penaltyMultiplier * architecture.ILconstant(skipCalc = True)]
        else:
          lossTrain_whole+=[loss(yHatTrain, yTrain).item()]
        evalTestLast = evalFunction(yHatTest, yTest)

    lossValid += [lossValueValid.item()]
    evalValid += [evalValueValid.item()]
    costTestLast = lossTestLast.item()
    xAxisValid += [(epoch * nBatches + batch)]

    # Testing best model (according to validation)
    with torch.no_grad():
        yHatValid = bestModel.forward(xValid)
        lossValidBest = loss(yHatValid, yValid)
        yHatTest = bestModel.forward(xTest)
        lossTestBest = loss(yHatTest, yTest)
        evalTestBest = evalFunction(yHatTest, yTest)
    costTestBest = lossTestBest.item()

    if verbose:
        print("Final evaluation results")
        # Print test results
        print("Test loss: %6.4f [Best]" % (
                            costTestBest) + " %6.4f [Last]" % (
                            costTestLast))
        
        print("Test RMSE: %6.4f [Best]" % (
                            evalTestBest) + " %6.4f [Last]" % (
                            evalTestLast))
        print(f"Number of Parameters: {sum(p.numel() for p in architecture.parameters())}")
        
        ################################
        ############# PLOT #############
        ################################
        
        plt.plot(xAxisValid, lossTrain_whole, label = "Train")
        plt.plot(xAxisValid, lossValid, label = 'Valid')
        plt.legend()
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Step')
        plt.title(f'{arch_type}')
        plt.show()

        plt.plot(xAxisValid, evalValid, label = "Eval Valid")
        plt.legend()
        plt.ylabel('Loss')
        plt.title('Validation Evaluation Loss')
        plt.xlabel('Step')
        plt.title(f'{arch_type}')
        plt.show()

        sns.set_style("darkgrid")
        data = pd.DataFrame({'step loss':lossTrain})
        #print(data.head)
        sns.lineplot(data=data)
        plt.ylabel('Training Loss Batch')
        plt.xlabel('Step')
        plt.title(f'{arch_type}')
        plt.show()
    return evalTestBest,  evalTestLast, bestModel, architecture