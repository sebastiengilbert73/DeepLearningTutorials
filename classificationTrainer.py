import argparse
import torch
import pandas
import numpy
import ast
import sys
import matplotlib.pyplot as plt
import LinearSeparation.linearClassifier as linearClassifier # Our linear neural network model
import cloudsVisualizer # To display the neural network decision boundary

parser = argparse.ArgumentParser()
parser.add_argument('datasetFilename', help='The filename of the dataset, in csv format')
parser.add_argument('--featuresList', help="The list of features of the input. Default: ['x', 'y']", default="['x', 'y']")
parser.add_argument('--numberOfClasses', help='The number of classes. Default: 2', type=int, default=2)
parser.add_argument('--validationProportion', help='The proportion of examples to use for validation [0, 1]. Default: 0.2', type=float, default=0.2)
parser.add_argument('--numberOfEpochs', help='The number of epochs. Default: 10000', type=int, default=10000)
parser.add_argument('--learningRate', help='The learning rate. Default: 0.1', type=float, default=0.1)
parser.add_argument('--momentum', help='The learning momentum. Default: 0.9', type=float, default=0.9)
parser.add_argument('--outputFile', help='The output text file for training data. Default: ./trainingOutput.csv', default='./trainingOutput.csv')
parser.add_argument('--saveDecisionBoundaryImages', help='Save the images showing the neural network decision boundaries',
                    dest='saveDecisionBoundaryImages', action='store_true')
parser.set_defaults(saveDecisionBoundaryImages=False)
args = parser.parse_args()

def main():
    print ("classificationTrainer.py > main()")

    features = ast.literal_eval(args.featuresList)
    numberOfFeatures = len(features)
    print ("features: {}".format(features))
    # Create a (linear) neural network
    neuralNetwork = linearClassifier.LinearClassifierNeuralNet(numberOfFeatures, args.numberOfClasses)
    # Load the data
    trainingInputsTensor, trainingTargetOutputsTensor, validationInputsTensor, validationTargetOutputsTensor =\
        SplitExamplesInTensors(args.datasetFilename, args.validationProportion, features)

    # Wrap the training and validation tensors in Variables
    trainingInputsTensor = torch.autograd.Variable(trainingInputsTensor)
    trainingTargetOutputsTensor = torch.autograd.Variable(trainingTargetOutputsTensor)
    validationInputsTensor = torch.autograd.Variable(validationInputsTensor)
    validationTargetOutputsTensor = torch.autograd.Variable(validationTargetOutputsTensor)

    # Create a loss function: Negative Log Likelihood
    lossFunction = torch.nn.NLLLoss()

    # Create an optimizer
    optimizer = torch.optim.SGD(neuralNetwork.parameters(), lr=args.learningRate, momentum=args.momentum)

    # Run the training
    lowestValidationLoss = sys.float_info.max
    championNeuralNetworkStateDict = {}
    trainingOutputFile = open(args.outputFile, 'w')
    trainingOutputFile.write("Epoch,trainLoss,validationLoss,validationAccuracy\n")
    frameNdx = 1
    for epoch in range(args.numberOfEpochs):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        trainingActualOutputsTensor = neuralNetwork(trainingInputsTensor)

        # Loss
        loss = lossFunction(trainingActualOutputsTensor, trainingTargetOutputsTensor)

        # Backward pass
        loss.backward()

        # Parameters update
        optimizer.step()

        trainLoss = loss.data[0]

        if (epoch + 1) % 10 == 0:
            # Validation loss
            validationActualOutputsTensor = neuralNetwork(validationInputsTensor)
            validationLoss = lossFunction(validationActualOutputsTensor, validationTargetOutputsTensor).data[0]
            validationAccuracy = Accuracy(validationActualOutputsTensor, validationTargetOutputsTensor)
            print ("Epoch {}: trainLoss = {}\tvalidationLoss = {}\tvalidationAccuracy = {}".format(
                epoch, trainLoss, validationLoss, validationAccuracy))
            trainingOutputFile.write("{},{},{},{}\n".format(epoch, trainLoss, validationLoss, validationAccuracy))

            if validationLoss < lowestValidationLoss:
                lowestValidationLoss = validationLoss
                championNeuralNetworkStateDict = neuralNetwork.state_dict()

            if args.saveDecisionBoundaryImages == True:
                f, ax = plt.subplots()
                cloudsVisualizer.ColorClasses(neuralNetwork, ax)
                cloudsVisualizer.CloudPlot(args.datasetFilename, ax)
                plt.savefig('boundary' + str(frameNdx) + '.png')
                plt.close()
                frameNdx += 1

    # Save the champion neural network state
    torch.save(championNeuralNetworkStateDict, 'champion_' + ("%.5f" % lowestValidationLoss ) + '.pth')
    trainingOutputFile.close()


def SplitExamplesInTensors(datasetFilename, validationProportion, featuresList):
    dataFrame = pandas.read_csv(datasetFilename)
    #print ("SplitExamplesInTensors(): dataFrame.loc[0, 'x']: {}".format(dataFrame.loc[0, 'x']))
    numberOfExamples = dataFrame.shape[0]
    if len(featuresList) != dataFrame.shape[1] - 1:
        raise Exception("SplitExamplesInTensors(): len(featuresList) ({}) != dataFrame.shape[1] - 1 ({})".format(
            len(featuresList), dataFrame.shape[1] - 1)  )
    numberOfValidationExamples = int(validationProportion * numberOfExamples)
    #print ("SplitExamplesInTensors(): numberOfValidationExamples = {}".format(numberOfValidationExamples))
    validationSampleIndices = numpy.random.choice(numberOfExamples, size=numberOfValidationExamples,
                                                  replace=False)
    #print ("validationSampleIndices = {}".format(validationSampleIndices))
    trainingExamplesArr = numpy.zeros((numberOfExamples - numberOfValidationExamples, len(featuresList)) )
    trainingTargetOutputsArr = numpy.zeros( (numberOfExamples - numberOfValidationExamples), dtype=int ) # The NLLLoss function requires the target outputs to be encoded as int's (LongTensor)
    validationExamplesArr = numpy.zeros( (numberOfValidationExamples, len(featuresList)) )
    validationTargetOutputsArr = numpy.zeros( (numberOfValidationExamples), dtype=numpy.int )
    currentTrainingNdx = 0
    currentValidationNdx = 0
    for exampleNdx in range(numberOfExamples):
        isValidationExample = (exampleNdx in validationSampleIndices)
        if isValidationExample:
            for columnNdx in range(len(featuresList)):
                validationExamplesArr[currentValidationNdx, columnNdx] = dataFrame.loc[exampleNdx,
                                                                                       featuresList[columnNdx]]
            classNdx = dataFrame.loc[exampleNdx, 'class']
            validationTargetOutputsArr[currentValidationNdx] = classNdx
            currentValidationNdx = currentValidationNdx + 1
        else:
            for columnNdx in range(len(featuresList)):
                trainingExamplesArr[currentTrainingNdx, columnNdx] = dataFrame.loc[exampleNdx,
                                                                                   featuresList[columnNdx]]
            classNdx = dataFrame.loc[exampleNdx, 'class']
            trainingTargetOutputsArr[currentTrainingNdx] = classNdx
            currentTrainingNdx = currentTrainingNdx + 1

    return torch.from_numpy(trainingExamplesArr).float(), torch.from_numpy(trainingTargetOutputsArr), \
           torch.from_numpy(validationExamplesArr).float(), torch.from_numpy(validationTargetOutputsArr)


def Accuracy(actualOutputsTensor, targetOutputsTensor):
    if actualOutputsTensor.data.shape[0] != targetOutputsTensor.data.shape[0]:
        raise Exception ("Accuracy(): The number of actual outputs ({}) doesn't match the number of target outputs ({})".format(
            actualOutputsTensor.data.shape[0], targetOutputsTensor.data.shape[0] ))
    _, actualChosenIndexTensor = torch.max(actualOutputsTensor, 1)
    numberOfCorrectPredictions = 0.0
    for exampleNdx in range(actualChosenIndexTensor.data.shape[0]):
        #print ("{}, {}".format(actualChosenIndexTensor[exampleNdx].data[0], targetOutputsTensor[exampleNdx].data[0]))
        if actualChosenIndexTensor[exampleNdx].data[0] == targetOutputsTensor[exampleNdx].data[0]:
            numberOfCorrectPredictions = numberOfCorrectPredictions + 1.0
    return numberOfCorrectPredictions/actualChosenIndexTensor.data.shape[0]

if __name__ == '__main__':
    main()
