import csv

#rawdata = 'name,age\nDan,33\nBob,19\nSheri,42'
#myreader = csv.reader(rawdata.splitlines())
#for row in myreader:
#    print(row[0], row[1])

'''
data = [
    ['Dan', 42],
    ['Cordelia', 33],
    ['Sammy', 52]
]

with open('output_csv.txt', 'w') as f:
    mywriter = csv.writer(f)
    
    mywriter.writerow(['name', 'age'])
    for row in data:
        mywriter.writerow(row)
'''
'''
data = [
    {'name': 'Pat', 'age': 78},
    {'name': 'Nancy', 'age': 23},
]
data_header = ['name', 'age']

with open('output_dictwriter.txt', 'w') as f:
    mywriter = csv.DictWriter(f, fieldnames = data_header)
    mywriter.writeheader()
    
    for row in data:
        mywriter.writerow(row)
'''
'''
data = ['id,name,age', '7,James Bond,42', '11,Dani West,19']

for row in data:
    col = row.split(',')
    print(col[2])
    print(type(col))
'''


import numpy as np

'''
data = "1\t234\t4456\t789"
a = np.fromstring(data, dtype=int, sep="\t")
print(a)
'''
'''
# import programm
data = "12\t34\t56\n32\t53\t65"
a = np.fromstring(data, dtype=None, sep="\t")
print(a)
'''

data = "12\t34\t56\n32\t53\t65"
for row in data.split(sep="\n"):
    for col in row.split(sep="\t"):
        print(col[1])





