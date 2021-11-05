import matplotlib.pyplot as plt
import numpy as np


def getQ(p, pi):
    M = 1.2
    q = p*M*(1-np.log(p/pi))
    return q


M = 1.2
font = {'size': 15}
fig = plt.figure()
ax = fig.add_subplot(111)
for pi in [100, 200, 300]:
    pp = np.linspace(0.1, pi*np.e, 100)
    q = getQ(pp, pi)
    ax.plot(pp, q, label='$p_i=%d$' % pi)
pp = np.linspace(0, 350, 100)
ax.plot(pp, M*pp)
plt.legend()
plt.tight_layout()
plt.xlim([0, 850])
plt.ylim([0, 450])
plt.text(x=50, y=380, s=r'$q=Mp(1-\ln\frac{p}{p_i})$', **font)
plt.text(x=50, y=330, s='$p_i$ controls the size of the yield surface', **font)
plt.savefig('YieldSurface.svg')
