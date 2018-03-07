import numpy as np
import matplotlib.pyplot as plt

# In[]:
def data_reading(filename):
    a = np.zeros((4999, 5))
    
    with open(filename) as f:
        i = -1
        for line in f:
            i = i+1
            j = 0
            for word in line.split():
                a[i][j] = word
                j = j+1
    return a

# In[]
filename = 'data_cut.txt'
a = data_reading(filename)

# In[]:

def normalization(data):
    data_mean = (np.sum(data, axis=0))
    data = data-data_mean
    data_norm = data/(np.sum(data, axis=0))
    return data_norm

#data_norm = normalization(a[10:15])
#print(a[10:15])
#print(data_norm)

# In[]:
def parameters_initialization(n_x, n_h1, n_h2, n_y):
    w1 = np.random.randn(n_h1, n_x) * 0.01
    w2 = np.random.randn(n_h2, n_h1) * 0.01
    w3 = np.random.randn(n_y, n_h2) * 0.01
    return w1, w2, w3

#print(w1.shape, w2.shape, w3.shape)


# In[19]:

def sigmoid(data):
    sig = np.exp(-data)
    return 1/(1+sig)


# In[20]:

def forward_pass(w1, w2, w3, x, y):
    z1 = np.dot(w1, x)
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1)
    a2 = sigmoid(z2)
    z3 = np.dot(w3, a2)
    a3 = sigmoid(z3)
    cost = (np.sum(np.power((y-a3),2)))/500
    return a1, a2, a3, cost

#print(a1.shape, a2.shape, a3.shape, cost)

# In[]:
    
def backward_pass_with_LF(w1, w2, w3, a1, a2, a3, x, y):
    error = y - a3
    delta3 = a3 * (1-a3) * error
    delta2 = a2 * (1-a2) * np.dot(w3.T, delta3)
    delta1 = a1 * (1-a1) * np.dot(w2.T, delta2)
    error_mid_layer2 = np.dot(w3.T, delta3)
    error_mid_layer1 = np.dot(w2.T, delta2)
    return error, delta1, delta2, delta3, error_mid_layer1, error_mid_layer2

#delta1, delta2, error = backward_pass(w1, w2, z1, a1, z2, a2, x, y)
#print(delta1.shape, delta2.shape)

# In[]:
    
def update_with_LF(x, a1, a2, a3, error, delta1, delta2, delta3, \
                   w1, w2, w3, learning_rate, error_mid_layer1, error_mid_layer2):
    delta_w3 = np.dot(delta3, a2.T)
    delta_w2 = np.dot(delta2, a1.T)
    delta_w1 = np.dot(delta1, x.T)
    
    error_sum3 = np.power(np.sum(error), 2)
    delta_w2_sum3 = np.power(np.sum(delta_w3), 2)
    learning_rate_with_LF13 = learning_rate * (error_sum3 / delta_w2_sum3)
    
    error_sum2 = np.power(np.sum(error_mid_layer2), 2)
    delta_w2_sum2 = np.power(np.sum(delta_w2), 2)
    learning_rate_with_LF12 = learning_rate * (error_sum2 / delta_w2_sum2)
    
    error_sum1 = np.power(np.sum(error_mid_layer1), 2)
    delta_w2_sum1 = np.power(np.sum(delta_w1), 2)
    learning_rate_with_LF11 = learning_rate * (error_sum1 / delta_w2_sum1)
    
    w3 = w3 + learning_rate_with_LF13 * delta_w3
    w2 = w2 + learning_rate_with_LF12 * delta_w2
    w1 = w1 + learning_rate_with_LF11 * delta_w1
    return w1, w2, w3

# In[]:
    

def NNmodel(last_iteration):
    error_list = np.zeros((1,50))
    error_list = error_list.reshape(-1, error_list.shape[1])
    
    a = data_reading(filename)
    a = normalization(a)
    
    learning_rate = 0.01
    n_x, n_h1, n_h2, n_y, m = 3, 50, 10, 2, 4999
    w1, w2, w3 = parameters_initialization(n_x, n_h1, n_h2, n_y)
    for i in range(50):
        
        a = np.random.permutation(a)
        offset = (i*500)%m
        if offset >= m:
            offset = 0
        
        x = (a[offset:offset+500, :3]).T
        y = (a[offset:offset+500, -2:]).T
#        x = (a[:, :3]).T
#        y = (a[:, -2:]).T
        
        a1, a2, a3, cost = forward_pass(w1, w2, w3, x, y)
        error, delta1, delta2, delta3, error_mid_layer1, error_mid_layer2 = \
        backward_pass_with_LF(w1, w2, w3, a1, a2, a3, x, y)

        w1, w2, w3 = update_with_LF(x, a1, a2, a3, error, delta1, delta2, delta3, \
                   w1, w2, w3, learning_rate, error_mid_layer1, error_mid_layer2)
        error_list[:,i] = cost
        if cost <= .00001:
            last_iteration = i
            break
        
    return w1, w2, w3, error_list, last_iteration

# In[]:
    
it = np.arange(1,50)
it = it.reshape(-1, it.shape[0])

last_iteration = 0

w1, w2, w3, error_list, last_iteration = NNmodel(last_iteration)
plt.figure()
plt.plot(it, error_list, 'ro')

print(last_iteration)

# In[]


