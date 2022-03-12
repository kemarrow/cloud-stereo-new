"""
Plots histogram of the change in height frame to frame

@author: Erin
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.optimize import curve_fit
from scipy.stats import chisquare

date =  '2021-10-24_12'

with open('cloud_height_'+date+'.pkl','rb') as f:
    height = pickle.load(f)

with open('cloud_speed_'+date+'.pkl','rb') as f:
    speed = pickle.load(f)

with open('cloud_updraft_'+date+'.pkl','rb') as f:
    updraft = pickle.load(f)




x = height.flatten()
y = speed.flatten()

#x= x[0:640*480*40]
#y= y[0:640*480*40]


def gaussian(x, A, mu, sigma, c):
    return A*(1 / (np.sqrt(2 * np.pi)*sigma) *
            np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))) + c

# def Voigt(x, x0, y0, a, sigma, gamma):
#     #sigma = alpha / np.sqrt(2 * np.log(2))
#     return y0 + a * np.real(wofz((x - x0 + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)

def lorentzian( x, a, x0, gam , c):
    return a * gam**2 / ( gam**2 + ( x - x0 )**2) + c


import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
          #'figure.figsize': (5, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

fig5, ax5 = plt.subplots(1,1)


n, bins, patches = ax5.hist(updraft*5, 1000, alpha=0.75, label='Data')
idx = (~np.isnan(updraft))
bins1 = np.arange(-1000, 1000, 2)
bin_centre = (bins[:-1] + bins[1:]) / 2

p, cov = curve_fit(lorentzian, bin_centre, n, p0=[10000, 0, 88, 1])

ax5.plot(bins1, lorentzian(bins1, *p), label='Lorentzian fit,'+'\n'+'HWHM =' +str(round(abs(p[2]), 2)), linewidth =3)
ax5.set_xlabel('Change in height from frame to frame (m)')
ax5.set_ylabel('Number of pixels')
ax5.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#ax5.set_title('Change in height from frame to frame')
plt.xlim(-500, 500)
plt.legend()
print("parameters: ")
print('Amp :',p[0],'centre :', p[1],'gam :', p[2])
print("Error: ")
print(np.sqrt(np.diag(cov)))
plt.show()

chi = chisquare(n, lorentzian(bins1, *p))


# fig5, ax5 = plt.subplots(1,1)


# n, bins, patches = ax5.hist(y, 1000, alpha=0.75, label='Data')
# idx = (~np.isnan(y))
# bins1 = np.linspace(-1000, 1000, 1001)


# #ax5.plot(bins, lorentzian(bins1, *p), label='Lorentzian fit, HWHM =' +str(round(abs(p[2]), 2)), linewidth =3)
# ax5.set_xlabel('Speed (m/s)')
# ax5.set_ylabel('Number of pixels')
# #ax5.set_title('Change in height from frame to frame')
# #plt.xlim(-500, 500)
# plt.legend()
# # print("arameters: ")
# # print('Amp :',p[0],'centre :', p[1],'gam :', p[2])
# # print("Error: ")
# # print(np.sqrt(np.diag(cov)))
# plt.show()




# plt.show()



