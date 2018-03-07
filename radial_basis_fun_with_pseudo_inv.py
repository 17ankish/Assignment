
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
#training exp
n_hc = 100
n_x = 3
n_y = 2
learning_rate = 0.01
sigma = .5
m = 9

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
phi_matrix = np.zeros((n_hc, m))

# In[]:

def euclidean_distance(data1, data2):
    distance = np.sqrt((data1[0] - data2[0])**2 + (data1[1] - data2[1])**2 + \
                       (data1[2] - data2[2])**2)
    return distance

# In[]:
 
m = 9

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
             
    #print("phi_temp.shape", str(phi_temp.shape))
    #print("phi_distance", str(phi_distance.shape))
    
    phi_value = np.exp(-(np.multiply(phi_distance, phi_distance))/ (2*sigma*sigma*100))

    phi_matrix[:, i] = phi_value.reshape(100)
 
print(phi_matrix.shape)
i = np.dot(phi_matrix.T, phi_matrix)
print(i.shape)
inv = np.linalg.inv(i)
print(inv)
phi_temp = np.dot(inv, phi_matrix.T)
w = np.dot(y_train, phi_temp)
    
print(phi_matrix.shape)
    if i == m-1:
        phi_nth_matrix = phi_matrix[:, i]
        print(phi_nth_matrix.shape)
        
# In[]: recursive least square
        
################ y_new is the current result after that we will update our weight
p_n_prev = np.dot(phi_nth_matrix.T, phi_nth_matrix)
k_n = p_n_prev * np.dot(phi_nth_matrix.T, np.linalg.inv(1 + np.dot(phi_nth_matrix, np.dot(p_n_prev, phi_nth_matrix.T))))
p_n = p_n_prev - np.dot(k_n, np.dot(phi_nth_matrix, p_n_prev))
w = w + np.dot((y_new - np.dot(w, phi_nth_matrix)), k_n)
p_n_prev = p_n

# In[]:
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
