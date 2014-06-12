from numpy import *
from pylab import *
x = loadtxt("vm.csv")
cm = loadtxt("conMat.csv")
nsteps, nNeurons =  shape(x)
nNeurons = nNeurons / 2
v = x[:, 0::2]
print cm
for i in arange(nNeurons):
    for j in arange(nNeurons):
        if(i != j and cm[i][j] == 1):
            plot(v[:, i], 'r')
            plot(v[:, j], 'k')
            title('%s-->%s'%(i, j))
            show()
            waitforbuttonpress()


