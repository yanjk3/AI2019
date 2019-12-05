# Author: Junkai-Yan
# Finished in 2019/12/05
# using EM algorithm and GGM model to cluster, data set is CFD data set

import numpy as np
import matplotlib.pyplot as plt

def loadData(filename):
    """
    generate data
    :param filename: name of the file
    :return: dataset: coordinate of each node
             datadic: name of each node
    """
    dataset = []
    datadic = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        contury = curLine[0]
        intLine = list(map(int, curLine[1:]))
        datadic.append(contury)
        dataset.append(intLine)
    return dataset, datadic


def prob(x, mu, sigma):
    """
    calculate the probability of x using Gaussian model
    :param x: data x
    :param mu: mean value of the class
    :param sigma: covariance matrix
    :return: x's probability
    """
    n = np.shape(x)[1]
    determinant = np.linalg.det(sigma)
    exp = float(-0.5 * (x - mu) * np.linalg.inv(sigma) * (x - mu).T)
    div = pow(2 * np.pi, n / 2) * (determinant**0.5)
    return pow(np.e, exp) / div


def EM(dataMat, maxIter=25):
    """
    EM algorithm, calculate likelihood matrix for data set
    :param dataMat: data set
    :param maxIter: max turns for iteration
    :return: matrix gamma, the likelihood matrix
    """
    m, n = np.shape(dataMat)
    # generate alpha(mixing coefficients, rate of each class) for 3 classes
    alpha = [1 / 3, 1 / 3, 1 / 3]

    # generate mean value for 3 classes
    mu = [dataMat[0, :], dataMat[1, :], dataMat[9, :]]

    # generate covariance matrix(unit matrix) for 3 classes
    sigma = [np.mat((np.eye(7, dtype=float))) for x in range(3)]

    # generate log likelihood matrix, dimension is m*classes
    gamma = np.mat(np.zeros((m, 3)))

    for i in range(maxIter):
        # calculate the log likelihood
        for j in range(m):
            sumAlphaMulP = 0
            # calculate posterior probability
            for k in range(3):
                gamma[j, k] = alpha[k] * prob(dataMat[j, :], mu[k], sigma[k])
                sumAlphaMulP += gamma[j, k]
            for k in range(3):
                gamma[j, k] /= sumAlphaMulP
        sumGamma = np.sum(gamma, axis=0)

        # update value
        for k in range(3):
            mu[k] = np.mat(np.zeros((1, n)))
            sigma[k] = np.mat(np.zeros((n, n)))

            # update mu(mean)
            for j in range(m):
                mu[k] += gamma[j, k] * dataMat[j, :]
            mu[k] /= sumGamma[0, k]

            # update sigma(covariance matrix)
            for j in range(m):
                sigma[k] += gamma[j, k] * (dataMat[j, :] - mu[k]).T *(dataMat[j, :] - mu[k])
            sigma[k] /= sumGamma[0, k]

            # update mixing coefficients
            alpha[k] = sumGamma[0, k] / m

        # update sigma(covariance matrix) by adding unit matrix to avoid it becoming a singular matrix
        for sigma_ in sigma:
            sigma_ += np.mat((np.eye(7, dtype=float)))

    return gamma


def gaussianCluster(dataMat):
    """
    GMM main function, calculate likelihood by calling function EM,
    and generate label for each data according to likelihood matrix
    :param dataMat: data set
    :return: the result of classification
    """
    # m: total number of samples, n: total number of attribute of each sample
    m, n = np.shape(dataMat)

    # cluster result
    clusterAssign = np.mat(np.zeros((m, 1)))

    # get gamma matrix (likelihood matrix)
    gamma = EM(dataMat)
    for i in range(m):
        # determine class according to gamma[i, :]'s max value's position
        clusterAssign[i] = np.argmax(gamma[i, :])
    return clusterAssign


def print_result(datadic, clusterAssment):
    """
    print the classification result
    :param datadic: list, element is the name of country
    :param clusterAssment: cluster result
    :return: none
    """
    f=[]
    s=[]
    t=[]
    for i in range(len(clusterAssment)):
        if int(clusterAssment[i]) == 0:
            f.append(datadic[i])
        if int(clusterAssment[i]) == 1:
            s.append(datadic[i])
        if int(clusterAssment[i]) == 2:
            t.append(datadic[i])
    print('class 1:', f)
    print('class 2:', s)
    print('class 3:', t)

def plot_result(dataset, datadic, clusterAssment):
    """
    reduce the dimension of data set and plot the result
    :param dataset: data set
    :param datadic: list, element is the name of country
    :param clusterAssment: cluster result
    :return: none
    """
    m, n = np.shape(dataset)
    # generate lower dimension data
    new_dataset = np.mat(np.zeros((m, 2)))
    for i in range(m):
        new_dataset[i, 0] = dataset[i, :4].mean()
        new_dataset[i, 1] = dataset[i, 4:].mean()

    # plot the result
    showCluster(new_dataset, datadic, clusterAssment)


def showCluster(dataMat, datadic, clusterAssment):
    """
    plot the result
    :param dataMat: data set
    :param datadic: list, element is the name of country
    :param clusterAssment: cluster result
    :return: none
    """
    numSamples, dim = dataMat.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(numSamples):
        markIndex = int(clusterAssment[i])
        x, y = dataMat[i, 0], dataMat[i, 1]
        plt.plot(x, y, mark[markIndex])
        plt.annotate(datadic[i], xy=(x, y), weight='light')
    plt.xlabel("Average WorldCup Rank")
    plt.ylabel("Average AsianCup Rank")
    plt.show()


if __name__=="__main__":
    # generate data set
    dataset, datadic = loadData('CFD.txt')
    dataset = np.mat(dataset)

    # get classification for data set
    clusterAssment = gaussianCluster(dataset)

    # print result
    print_result(datadic, clusterAssment)

    # reduce the dimension of data and plot the result
    plot_result(dataset, datadic, clusterAssment)