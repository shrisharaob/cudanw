from numpy import *
from pylab import *
x = loadtxt("vm.csv")
print shape(x)
plot(x[:, 0])
plot(x[:, 2])
plot(x[:, 4])
plot(x[:, 6])
show()
