
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
#training exp
n_hc = 100
n_x = 3
n_y = 2
learning_rate = 0.01
sigma = .5

# In[]:

a = np.zeros((4999, 5))

with open('data_cut.txt') as f:
    i = -1
    for line in f:
        i = i+1
        j = 0
        for word in line.split():
            a[i][j] = word
            j = j+1

# In[]:
            
# normalization
a_mean = (np.sum(a, axis=0))
a = a-a_mean
a = a/(np.sum(a, axis=0))


# In[]:

w = np.random.randn(n_y, n_hc) * 0.01

a = np.random.permutation(a)
c = a[:100, :3]
a = np.random.permutation(a)

phi_distance = np.zeros((n_hc, 1))
#print(x.shape, y.shape, u.shape, w.shape, c.shape, phi_distance.shape)
phi_temp = np.zeros((n_x, n_hc))

# In[]:

def euclidean_distance(data1, data2):
    distance = np.sqrt((data1[0] - data2[0])**2 + (data1[1] - data2[1])**2 + \
                       (data1[2] - data2[2])**2)
    return distance

# In[]:
 
m = 4999

#error_list = np.zeros((1, 99))
for k in range(10):
    error_list = []
    a = np.random.permutation(a)
    a_new = a.T
    
    for i in range(m):
        temp = a_new[:, i]
        temp = temp.reshape(temp.shape[0], -1)
        x_train = temp[:3, :]
        y_train = temp[-2:, :]
    #    print(x_train.shape, y_train.shape)
        
        #print(x_train, x_train.shape[0])
        #print(phi_distance.shape)
        
        #print(data1.shape)
        for j in range(n_hc):
            c_temp = c[j, :]
            
            c_temp = c_temp.reshape(c_temp.shape[0], -1)
    #        print(c_temp.shape)
            #print("data2.shape", str(data2.shape))
            #print("minus", str((data1 - data2).shape))
            #distance = np.linalg.norm(data1.T, data2.T)
            distance = euclidean_distance(x_train, c_temp)
    #        print(x_train, c_temp)
    #        print(distance)
            phi_distance[j, :] = distance
            temp = np.array([x_train - c_temp])
            temp = temp.reshape(temp.shape[0], -1)
    #        print(temp)
            phi_temp[:, j] = temp
            
         
        #print("phi_temp.shape", str(phi_temp.shape))
        #print("phi_distance", str(phi_distance.shape))
        phi_matrix = np.exp(-(np.multiply(phi_distance, phi_distance))/ (2*sigma*sigma*100))
        #print(phi_matrix.shape)
        
        Z = np.dot(w, phi_matrix)
    #    print(Z.shape)
        
        error = y_train - Z
        cost = np.sum(np.square(error))/2
        
        
        w = w + learning_rate * np.dot(error, phi_matrix.T)
        
        update = np.dot(error.T, (np.dot(w, phi_matrix))) * phi_temp
        update = update * learning_rate / (sigma**2)
        
        c  = c + update.T
    #    print(update.shape)
        #print("ZZZZZZZZZZZZZZZZZZZZZZZZZZ", str(c.shape))
        #print(w.shape)
        
        if i % 50 == 0:
            error_list.append(cost)
    
    if k%2 == 0:
        error_list = np.asarray(error_list)   
    #    print(error_list.shape)
        error_list = error_list.reshape(-1, error_list.shape[0])
        it = np.arange(1,101)
        it = it.reshape(-1, it.shape[0])
        plt.figure()
        plt.plot(it, error_list, 'ro')

'''
###### testing  
print("testing")
print(x_test.shape, y_test.shape)
phi_temp = np.zeros((n_x, n_hc)) 
phi_distance = phi_matrix =  np.zeros((1, n_hc))      
error_test = np.zeros((1, m_testing))        
Z_test = np.zeros((1, m_testing))
   
for i in range(m_testing):
    x_temp = x_test[:, i]
    y_temp = y_test[:, i]
    u_temp = u_test[:, i]
    
    #print(phi_distance.shape)
    data1 = np.array([[x_temp[0]], [u_temp[0]]])
    
    
    for j in range(n_hc):
        data2 = c[:, j]
        data2 = data2.reshape(data2.shape[0], -1)
        #print("data2.shape", str(data2.shape))
        #print("minus", str((data1 - data2).shape))
        #distance = np.linalg.norm(data1.T, data2.T)
        distance = euclidean_distance(data1, data2)
        #print(distance)
        phi_distance[:, j] = distance
        phi_temp[:, :j] = data1 - data2
        
     
    #print("phi_temp.shape", str(phi_temp.shape))
    #print("phi_distance", str(phi_distance.shape))
    phi_matrix = np.exp(-(np.multiply(phi_distance, phi_distance))/ (2*sigma*sigma*100))
    #print(phi_matrix.shape)
    
    Z_test[:, i] = np.dot(phi_matrix, w.T)
    
    error_test[:, i] = Z_test[:, i] - y_temp    

    if i%50 == 0:
        print(error_test[:, i])

print(np.sum(np.abs(error))/ 1000)

#print(Z_test[:, :10], y_test[:, :10])

#plt.plot(x_test, Z_test)
#
i = np.linspace(1,500,1)
i = i.reshape(-1, i.shape[0])
plt.stem(i, error_test, linewidth=100)    
#
#print("max and min of error")
#print(np.max(error), np.min(error))
#print("max and min of error_test")
#print(np.max(error_test), np.min(error_test))
#
'''
