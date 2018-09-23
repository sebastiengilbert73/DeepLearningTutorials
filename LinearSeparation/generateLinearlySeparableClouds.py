import argparse
import random
import math
import numpy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('rho', help='The radius rho in (rho, theta)', type=float)
parser.add_argument('theta', help='The angle theta in (rho, theta)', type=float)
parser.add_argument('outputFilename', help='The output filename')
parser.add_argument('--noiseSigma', help='The noise standard deviation. Default: 0', type=float, default=0)
parser.add_argument('--numberOfPoints', help='The number of generated points. Default: 200', type=int, default=200)
args = parser.parse_args()

def main():
    print ('generateLinearlySeparableCloud.py main()')
    xs_0 = list()
    ys_0 = list()
    xs_1 = list()
    ys_1 = list()
    with open(args.outputFilename, 'w') as outputFile:
        outputFile.write('x0,x1,class\n')
        for pointNdx in range(args.numberOfPoints):
            uncorrupted_x = -1.0 + 2.0 * random.random()
            uncorrupted_y = -1.0 + 2.0 * random.random()

            isOne = (uncorrupted_x * math.cos(args.theta) + uncorrupted_y * math.sin(args.theta)) > args.rho
            corrupted_x = uncorrupted_x + args.noiseSigma * numpy.random.normal()
            corrupted_y = uncorrupted_y + args.noiseSigma * numpy.random.normal()
            print ('{}, {}, {}'.format(corrupted_x, corrupted_y, (1 if isOne else 0)))
            outputFile.write('{},{},{}\n'.format(corrupted_x, corrupted_y, (1 if isOne else 0)))
            if isOne:
                xs_1.append(corrupted_x)
                ys_1.append(corrupted_y)
            else:
                xs_0.append(corrupted_x)
                ys_0.append(corrupted_y)
    plt.scatter(xs_0, ys_0)
    plt.scatter(xs_1, ys_1)
    plt.show()


if __name__ == '__main__':
    main()
