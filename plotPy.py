from numpy import *
import pylab as plt
x = loadtxt("vm.csv")
cm = loadtxt("conMat.csv")
st = loadtxt("spkTimes.csv", delimiter = ';')
#spkNeuronIds = unique(st[:,1])
nsteps, nNeurons =  shape(x)
nNeurons = nNeurons
print "#N", nNeurons
v = x
print cm
plt.ion()
for i in arange(nNeurons):
    for j in arange(nNeurons):
        if(i != j and cm[i][j] > 0):
            #            if(any(spkNeuronIds == i)):
            p1, = plt.plot(v[:, i], 'r')
            p2, = plt.plot(v[:, j], 'k')
            plt.legend([p1, p2], ["%s" %(i), "%s" %(j)])
            plt.title('%s-->%s'%(i, j))
            plt.ylim(-90, 120)
            plt.draw()
            plt.waitforbuttonpress(-1)
            plt.clf()
            





