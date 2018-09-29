import argparse
import random
import math
import numpy
import matplotlib.pyplot as plt
import ast

parser = argparse.ArgumentParser()
parser.add_argument('outputFilename', help='The output filename')
parser.add_argument('--center', help="The clouds center. Default: '(0, 0)'", default='(0, 0)')
parser.add_argument('--radius', help='The inner cloud radius. Default: 0.5', type=float, default=0.5)
parser.add_argument('--noiseSigma', help='The noise standard deviation. Default: 0', type=float, default=0)
parser.add_argument('--numberOfPoints', help='The number of generated points. Default: 200', type=int, default=200)
args = parser.parse_args()

center = ast.literal_eval(args.center)
if len(center) != 2:
    raise Exception("The string {} cannot be converted to a 2D point".format(args.center))

def main():
    print ("generateDoughnutClouds.py")
    x0_class0 = list()
    x1_class0 = list()
    x0_class1 = list()
    x1_class1 = list()
    with open(args.outputFilename, 'w') as outputFile:
        outputFile.write('x0,x1,class\n')
        for pointNdx in range(args.numberOfPoints):
            uncorrupted_x0 = -1.0 + 2.0 * random.random()
            uncorrupted_x1 = -1.0 + 2.0 * random.random()

            radius = math.sqrt(pow(uncorrupted_x0 - center[0], 2.0) + pow(uncorrupted_x1 - center[1], 2.0))
            isOne = radius > args.radius
            corrupted_x0 = uncorrupted_x0 + args.noiseSigma * numpy.random.normal()
            corrupted_x1 = uncorrupted_x1 + args.noiseSigma * numpy.random.normal()
            outputFile.write('{},{},{}\n'.format(corrupted_x0, corrupted_x1, (1 if isOne else 0)))
            if isOne:
                x0_class1.append(corrupted_x0)
                x1_class1.append(corrupted_x1)
            else:
                x0_class0.append(corrupted_x0)
                x1_class0.append(corrupted_x1)
    plt.scatter(x0_class0, x1_class0)
    plt.scatter(x0_class1, x1_class1)
    plt.show()

if __name__ == '__main__':
    main()