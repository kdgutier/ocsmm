import numpy as np
import itertools


def kernel(X,Z, gamma):
    dists_2 = np.sum(np.square(X)[:,np.newaxis,:],axis=2)-2*X.dot(Z.T)+np.sum(np.square(Z)[:,np.newaxis,:],axis=2).T
    k_XZ = np.exp(-gamma*dists_2)
    return k_XZ

def gramMeasure(S, gamma):
    n = len(S)
    Ktrain = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            k=kernel(S[i],S[j], gamma)
            Ktrain[i,j] = np.average(k)
            Ktrain[j,i] = Ktrain[i,j]
            if i == j:
                break
    normKtrain = measureNormSquare(S, gamma)
    normalizer = np.sqrt(normKtrain*normKtrain.T)
    Ktrain = np.multiply(Ktrain, np.reciprocal(normalizer))
    return Ktrain

def cross_gramMeasure(S1, S2, gamma):
    n1, n2 = len(S1), len(S2)
    Kcross = np.zeros(shape=(n1,n2))
    for i in range(n1):
        for j in range(n2):
            k=kernel(S1[i],S2[j], gamma)
            Kcross[i,j] = np.average(k)
    normK1 = measureNormSquare(S1, gamma)
    normK2 = measureNormSquare(S2, gamma)
    normalizer = np.sqrt(normK1*normK2.T)
    Kcross = np.multiply(Kcross, np.reciprocal(normalizer))
    return Kcross

def measureNormSquare(S, gamma):
    n = len(S)
    K = np.zeros(shape=(n,1))
    for i in range(n):
        K[i,0] = np.average(kernel(S[i],S[i], gamma))
    return K
