import numpy as np
import matplotlib.pyplot as plt

# In[]:
def euc(a,b):
    distance = np.sqrt(np.square(a[0] - b[0])+np.square(a[1] - b[1]))
    return distance

# In[]:
x = np.random.randn(2,200)
w = np.random.randn(2,20)

distances = np.zeros((1,20))

plt.plot(x[0], x[1], 'ro')
# In[]:
m = 5000
tau = 1
eps = 0.001
plt.ion()

for i in range(m):
    x_train = np.random.permutation(x.T)
    x_train = x_train.T
    x_train = x_train[:, 1]
    x_train = x_train.reshape(x_train.shape[0], -1)
    
    distances = euc(x_train, w)
    distances = distances.reshape(-1, distances.shape[0])
    
    i_index = np.argmin(distances)
    
    winner = w[:, i_index]
    
    distances = euc(winner, w)
    distances = distances.reshape(-1, distances.shape[0])
    
    sigma = np.exp(-(i/tau))
   
    w = w + np.exp(-(distances**2)/(2*sigma**2 + eps)) * (x_train-w)
    

    if i%1000 == 0:
        plt.plot(w[0], w[1], 'ro') 
        plt.pause(0.0001)


   
   