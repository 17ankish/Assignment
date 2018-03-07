import numpy as np
import matplotlib.pyplot as plt
np.random.seed(20)

def euc(a,b):
    distance = np.sqrt(np.square(a[0] - b[0])+np.square(a[1] - b[1]))
    return distance



x = np.random.uniform(0, 1, (2,100))
w = np.random.uniform(0, 1, (2,20))

#plt.plot(w[0], w[1], 'ro')

distances = np.zeros((1,20))

#plt.figure()
#plt.plot(x[0],x[1],'ro')
m=10
epsilon = 0.1
sigma0 = 0.1
tau = 2
for training in range(m):
    sigma = sigma0 * np.exp(-(training/tau))
    
    x = np.random.permutation(x.T)
    
    x_train = x.T
    x_train = x_train[:,1]
    print(x_train)
    x_train = x_train.reshape(x_train.shape[0], -1)
    print(x_train.shape)
    distances = euc(x_train, w)

#    for j in range(10):
    #        temp =  w[:,j]
    #        temp = temp.reshape(temp.shape[0], -1)
    #        #print(temp.shape, x_train.shape)
    #        distances[:,j] = euc(x_train, temp)
    #        #print(euc(x_train,temp))
            
    i_index = np.argmin(distances)
    #print(i_index)
    winner = w[:, i_index]
    winner = winner.reshape(winner.shape[0], -1)
    gaussian_temp = euc(winner, w)
    gaussian_fun = np.exp(-(gaussian_temp/(2*sigma*sigma + epsilon)))
    print((x_train-w).shape)
    #print((np.multiply(gaussian_fun, (x_train-w))).shape)
    w = w + np.multiply(gaussian_fun, (x_train-w))
    if (training%2 == 0):
        plt.figure()
        plt.plot(w[0],w[1], 'ro')

    
    
    

#plt.plot(w[0], w[1], 'ro')
#plt.plot(w[0], w[1], 'ro')
#print(x.shape)
        


