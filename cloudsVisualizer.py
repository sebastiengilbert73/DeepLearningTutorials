import argparse
import matplotlib.pyplot as plt
import pandas
import torch
import numpy
import LinearSeparation.linearClassifier

"""parser = argparse.ArgumentParser()
parser.add_argument('--datasetFilepath', help='The filepath of the dataset. Default:None', default=None)

args = parser.parse_args()
"""

def CloudPlot(datasetFilepath, ax):
    dataFrame = pandas.read_csv(datasetFilepath)
    headersList = list(dataFrame)
    print ("headersList = {}".format(headersList))
    if len(headersList) != 3:
        raise Exception("CloudPlot(): The number of columns ({}) is not 3".format(len(headersList)))
    if headersList[2] != 'class':
        raise Exception("CloudPlot(): The 3rd column is not 'class'")

    x0_class0_list = []
    x1_class0_list = []
    x0_class1_list = []
    x1_class1_list = []
    numberOfExamples = dataFrame.shape[0]
    for exampleNdx in range(numberOfExamples):
        classNdx = dataFrame.loc[exampleNdx, 'class']
        if classNdx == 0:
            x0_class0_list.append(dataFrame.loc[exampleNdx, headersList[0]])
            x1_class0_list.append(dataFrame.loc[exampleNdx, headersList[1]])
        elif classNdx == 1:
            x0_class1_list.append(dataFrame.loc[exampleNdx, headersList[0]])
            x1_class1_list.append(dataFrame.loc[exampleNdx, headersList[1]])
        else:
            raise Exception ("CloudPlot(): The class for example {} is not 0 or 1 ({})".format(exampleNdx, classNdx))

    #f, ax = plt.subplots()
    ax.scatter(x0_class0_list, x1_class0_list)
    ax.scatter(x0_class1_list, x1_class1_list)
    #plt.show()


def ColorClasses(neuralNetwork, ax, x0Range=(-1, 1), x1Range=(-1, 1), color0=(0.6, 0.6, 0.9),
                 color1=(0.9, 0.7, 0.5)):

    x0s = numpy.linspace(x0Range[0], x0Range[1], 101)
    x1s = numpy.linspace(x1Range[0], x1Range[1], 101)
    class0x0s = []
    class0x1s = []
    class1x0s = []
    class1x1s = []
    color0s = []
    color1s = []
    for x1 in x1s:
        for x0 in x0s:
            inputTensor = torch.Tensor([x0, x1])
            outputTensor = neuralNetwork(torch.autograd.Variable(inputTensor))
            if outputTensor.data[0] > outputTensor.data[1]:
                class0x0s.append(x0)
                class0x1s.append(x1)
                color0s.append(color0)
            else:
                class1x0s.append(x0)
                class1x1s.append(x1)
                color1s.append(color1)
    color0Arr = numpy.ndarray(shape=(len(class0x0s), 3))
    for index in range(len(class0x0s)):
        color0Arr[index, 0] = color0[0]
        color0Arr[index, 1] = color0[1]
        color0Arr[index, 2] = color0[2]
    color1Arr = numpy.ndarray(shape=(len(class1x0s), 3))
    for index in range(len(class1x0s)):
        color1Arr[index, 0] = color1[0]
        color1Arr[index, 1] = color1[1]
        color1Arr[index, 2] = color1[2]


    ax.scatter(class0x0s, class0x1s, c=color0Arr)
    ax.scatter(class1x0s, class1x1s, c=color1Arr)

def main():
    print ("cloudsVisualizer.py main()")
    neuralNetwork = linearClassifier.LinearClassifierNeuralNet(2, 2)
    neuralNetwork.load_state_dict(torch.load('/home/sebastien/projects/DeepLearningTutorials/LinearSeparation/champion_0.27631.pth'))
    f, ax = plt.subplots()
    ColorClasses(neuralNetwork, ax)
    """if args.datasetFilepath is not None:

        CloudPlot(args.datasetFilepath, ax)

    plt.show()
    """

if __name__ == '__main__':
    main()