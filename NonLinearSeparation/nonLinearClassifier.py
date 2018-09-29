import torch
import numpy

class TwoLayersClassificationNeuralNet(torch.nn.Module):
    def __init__(self, numberOfInputs, numberOfHiddenUnits, numberOfClasses):
        super(TwoLayersClassificationNeuralNet, self).__init__()
        self.structure = torch.nn.Sequential(
            torch.nn.Linear(numberOfInputs, numberOfHiddenUnits),
            torch.nn.ReLU(),
            #torch.nn.Sigmoid(),
            torch.nn.Linear(numberOfHiddenUnits, numberOfClasses)
        )


    def forward(self, inputs):
        h = self.structure(inputs)
        return torch.nn.functional.log_softmax(h)


if __name__ == '__main__':
    classifier = TwoLayersClassificationNeuralNet(2, 5, 2)
    inputTensor = torch.Tensor([1.5, -0.7])
    outputTensor = classifier(torch.autograd.Variable(inputTensor))
    print ("outputTensor: {}".format(outputTensor))

    # Get the number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, classifier.parameters())
    numberOfParameters = sum([numpy.prod(p.size()) for p in model_parameters])
    print ("Number of trainable parameters: {}\n".format(numberOfParameters))

    print ("Trainable parameters:")
    for name, param in classifier.named_parameters():
        if param.requires_grad:
            print (name, param.data)