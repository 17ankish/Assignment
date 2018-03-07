import numpy as np

a = np.array([[1,2],[3,4]])
a_temp =np.array((a[:, 1]))

#print(([a_temp]).shape)
#
#a = np.zeros((4999, 5))
#
#with open('data_cut.txt') as f:
#    i = -1
#    for line in f:
#        i = i+1
#        j = 0
#        for word in line.split():
#            a[i][j] = word
#            j = j+1
#            
#
#print(a[:5])




#import pandas as pd
#import csv
#import numpy as np
#
#filename = 'data_cut.txt'
#
##with open(filename) as f:
##    lines = f.readlines()
##
##
##lines = [line.rstrip('\n') for line in open(filename)]
##
##temp = lines[:1]
##
##temp = np.asarray(temp)
##print(temp.shape)
###for word in temp.split():
###    print(word)
#
#lines = []
#with open(filename,'r') as f:
#    for line in f:
#        lines.append(line)
##        for word in line.split():
##           print(word)  
#lines = np.asarray(lines)
#temp = lines[:1]
#np.fromstring(temp, dtype=int, sep="\t")
#
#print(temp)
#
##print(lines[:1])
#
#
#
