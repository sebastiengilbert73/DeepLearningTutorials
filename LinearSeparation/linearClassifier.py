import torch

class LinearClassifierNeuralNet(torch.nn.Module):
    def __init__(self, numberOfInputs, numberOfClasses):
        super(LinearClassifierNeuralNet, self).__init__()
        self.linear = torch.nn.Linear(numberOfInputs, numberOfClasses)

    def forward(self, inputTensor):
        outputLinear = self.linear(inputTensor)
        return torch.nn.functional.log_softmax(outputLinear)

if __name__ == '__main__':
    classifier = LinearClassifierNeuralNet(2, 2)
    inputTensor = torch.Tensor([1.5, -0.7])

    outputTensor = classifier( torch.autograd.Variable(inputTensor) )
    print ("outputTensor: {}".format(outputTensor))