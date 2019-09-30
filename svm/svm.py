'''
sequential minimal optimization:
1. create a vector of alpha with initial value 0
2. while n < maxLoop:
        for every sample vector in the data:
            if the vector can be optimized:
                randomly choose another sample vector
                optimize these two vectors
                if neither of them cannot be optimized:
                    break
            if all the vector cannbot be optimized:
                n += 1
'''

import numpy as np

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(data, label, C, tol, maxIter):
    b = 0
    m, n = data.shape
    label = label.T
    alphas = np.zeros((m,1))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = False
        for i in range(m):
            fxi = float(np.multiply(alphas, label).T * (data * data[i,:].T)) + b
            Ei = fxi - float(label[i])
            if label[i] * Ei < -tol and alphas[i] < C or label[i] * Ei > tol and alphas[i] > 0:
                j = np.random.choice(range(i, m), 1)                # randomly choose a sampe
                fxj = float(np.multiply(alphas, label).T * (data * data[j,:].T)) + b
                Ej = fxj - float(label[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if label[i] != label[j]:                        
                    L = max(0, alphas[j] - alphas[i])               # make sure alpha in [0, C]
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    continue
                eta = 2.0 * data[i, :] * data[j, :].T - data[i, :] * data[i, :].T - data[j, :] * data[j,:].T
                if eta >= 0:
                    continue
                alphas[j] -= label[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if np.abs(alphas[j] - alphaJold) < 0.00001:
                    continue
                alphas[i] += label[j] * label[i] * (alphaJold - alphas[j])
                b1 = b - Ei - label[i] * (alphas[i] - alphaIold) * data[i, :] * data[i, :].T - \
                    label[j] * (alphas[j] - alphaJold) *  data[i, :] * data[j, :].T
                b2 = b - Ej - label[i] * (alphas[i] - alphaIold) * data[i, :] * data[j, :].T - \
                    label[j] * (alphas[j] - alphaJold) * data[j, :] * data[j, :].T
                if 0 < alphas[i] and C > alphas[i]:
                    b = b1
                elif 0 < alphas[j]  and  C > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphaPairsChanged += 1
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
    return b, alphas

