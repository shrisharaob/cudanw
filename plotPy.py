from numpy import *
import time
import pylab as plt
x = loadtxt("vm.csv")
cm = loadtxt("conMat.csv")
nsteps, nNeurons =  shape(x)
nNeurons = nNeurons / 2
print "#N", nNeurons
v = x[:, 0::2]
print cm
plt.ion()
for i in arange(nNeurons):
    for j in arange(nNeurons):
        if(i != j and cm[i][j] > 0):
            plt.close()
            plt.figure()
            plt.plot(v[:, i], 'r')
            plt.plot(v[:, j], 'k')
            plt.title('%s-->%s'%(i, j))
            plt.ylim(-90, 120)
            plt.draw()
            state = False
            while(~state):
                state = plt.waitforbuttonpress(-1)
                print state
                break
                





