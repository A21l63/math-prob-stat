import csv
import numpy as np
import matplotlib.pyplot as plt
import math

def countJakkar(x):
            min_inc = list(x[0])
            max_inc = list(x[0])
            for interval in x:
                min_inc[0] = max(min_inc[0], interval[0])
                min_inc[1] = min(min_inc[1], interval[1])
                max_inc[0] = min(max_inc[0], interval[0])
                max_inc[1] = max(max_inc[1], interval[1])
            JK = (min_inc[1] - min_inc[0]) / (max_inc[1] - max_inc[0])
            return JK


r_file = open("table1.csv", "r")
file_reader = csv.reader(r_file, delimiter = ";")
table1 = np.array(list(file_reader))
U1 = np.array(table1[:,0])
U1 = np.delete(U1, 0)
U1 = np.array(U1, dtype = float)

r_file = open("table2.csv", "r")
file_reader = csv.reader(r_file, delimiter = ";")
table2 = np.array(list(file_reader))
U2 = np.array(table2[:,0])
U2 = np.delete(U2, 0)
U2 = np.array(U2, dtype = float)

eps = 1e-4
N = np.linspace(1, 200, num = 200)

plt.scatter(N,U2,s=1)
plt.title('ФП2')
plt.show()
plt.scatter(N,U1,s=1)
plt.title('ФП1')
plt.show()

goodsensor = np.zeros(shape = (200, 2))
badsensor = np.zeros(shape = (200, 2))
for i in range(len(U2)):
    goodsensor[i][0] = U2[i] - eps
    goodsensor[i][1] = U2[i] + eps
    badsensor[i][0] = U1[i] - eps
    badsensor[i][1] = U1[i] + eps

#linear programming

tau = list()
w = list()

etalonTau = list()
etalonW = list()

#got solution using Octave
with open("ch1.txt") as file:
  tau = [float(t) for t in file.readline().split()]
  for line in file.readlines():
    w.append(float(line))

with open("ch2.txt") as file:
  etalonTau = [float(t) for t in file.readline().split()]
  for line in file.readlines():
    etalonW.append(float(line))


intervals = [[d - eps, d + eps] for d in U1]
eIntervals = [[d - eps, d + eps] for d in U2]

for i in range(len(intervals)):
  if i == 0:
    plt.vlines(i + 1, intervals[i][0], intervals[i][1], 'C0', lw=1, label = "$I_1$")
  else:
    plt.vlines(i + 1, intervals[i][0], intervals[i][1], 'C0', lw=1)

plt.plot([1, len(intervals)], [tau[1] + tau[0], len(intervals) * tau[1] + tau[0]], color='green')
plt.title('ФП1')
plt.show()

for i in range(len(eIntervals)):
  if i == 0:
    plt.vlines(i + 1, eIntervals[i][0], eIntervals[i][1], 'C0', lw=1, label = "$I_1$")
  else:
    plt.vlines(i + 1, eIntervals[i][0], eIntervals[i][1], 'C0', lw=1)

plt.plot([1, len(eIntervals)], [etalonTau[1] + etalonTau[0], len(eIntervals) * etalonTau[1] + etalonTau[0]], color='green')
plt.title('ФП2')
plt.show()

plt.hist(w, bins = 10)
plt.title('ФП1')
plt.show()

plt.hist(etalonW, bins = 10)
plt.title('ФП2')
plt.show()

A1 = tau[0]
B1 = tau[1]

A2 = etalonTau[0]
B2 = etalonTau[1]

for i in range(len(U2)):
    goodsensor[i][0] = goodsensor[i][0] - B2*i
    goodsensor[i][1] = goodsensor[i][1]- B2*i
    badsensor[i][0] = badsensor[i][0]- B1*i
    badsensor[i][1] = badsensor[i][1]- B1*i


for i in range(len(eIntervals)):
    plt.vlines(i + 1, goodsensor[i][0], goodsensor[i][1], 'C0', lw=1)
plt.title('ФП2')
plt.show()


for i in range(len(eIntervals)):
    plt.vlines(i + 1, badsensor[i][0], badsensor[i][1], 'C0', lw=1)
plt.title('ФП1')
plt.show()

#histogram
min_value = goodsensor[0][0]
max_value = goodsensor[-1][1]
step = 0.0001
hist = [(t[1] + t[0]) / 2 for t in goodsensor]
plt.hist(hist)
plt.title('ФП2')
plt.show()

min_value = badsensor[0][0]
max_value = badsensor[-1][1]
step = 0.0001
hist = [(t[1] + t[0]) / 2 for t in badsensor]
plt.hist(hist)
plt.title('ФП1')
plt.show()


JK = np.array([], dtype = float)

# I ran proramm few times and found interval for optimal R
start = 1.085
stop = 1.095
step = 1e-6
R = start
Rarr = np.array([], dtype = float)
while R<=stop:

    x = np.concatenate((goodsensor, np.multiply(R, badsensor)))
    JK = np.append(JK, countJakkar(x))
    Rarr = np.append(Rarr, R)
    R = R + step

print('max jakkar is', np.amax(JK))
print('index of max = ', np.argmax(JK))
print('R21 max = ', Rarr[np.argmax(JK)])

plt.close()
plt.plot(Rarr,JK, c = 'k', label = 'JK')
plt.plot(Rarr[np.argmax(JK)], np.amax(JK), 'o', label = "R21 max")
plt.legend()
plt.show()

#united histogram
x = np.concatenate((goodsensor, np.multiply(Rarr[np.argmax(JK)], badsensor)))
min_value = x[0][0]
max_value = x[-1][1]
step = 0.0001
hist = [(t[1] + t[0]) / 2 for t in x]
plt.hist(hist)
plt.title('United histogram')
plt.show()