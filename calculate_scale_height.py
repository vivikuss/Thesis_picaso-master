from trans_spectra import case_obj
import numpy as np
import patch_scipy
import matplotlib.pyplot as plt
from picaso import justdoit as jdi
from picaso import justplotit as jpi
import astropy.constants as ap
import astropy.units as apu

### CUSTOMIZE IF NEEDED ###
### K2-18b ###
# Planet's Properties
mpla = 0.0272           # Mjup ##### NASA Archive Benneke 2019
rpla = 0.2328           # Rjup ##### NASA Archive Benneke 2019
ps = 20

# Star's Properties
tstel = 3500            # K               #### NASA Archive
met = 0.123             # dex           #### NASA Archive
g = 4.858               # log g value       #### NASA Archive
rstel = 0.44            # Rsun              #### NASA Archive
mag = 8.899
num_scale = 4

### LHS 1140b ###
# # Planet's Properties
# mpla = 0.0176           # Mjup ##### NASA Archive Benneke 2019
# rpla = 0.1543            # Rjup ##### NASA Archive Benneke 2019
# ps = 20

# # Star's Properties
# tstel = 3500             # K               #### NASA Archive
# met = -0.15            # dex           #### NASA Archive
# g = 5.0               # log g value       #### NASA Archive
# rstel = 0.21              # Rsun              #### NASA Archive
# mag = 9.612                             #### NASA Archive
# gridlines = np.array([5700,5800,5900,6000,6100,6200])  # gridlines for LHS 114b
# num_scale = 3                       # number of contributing scale heights
##################

cases = [case_obj[0],case_obj[1],case_obj[2]]   

### CUSTOMIZE IF NEEDED ###
# 2 CO2 lines, 2 CH4 lines    
lower_limits = [1.5,2.0,3.0,4.0]   # lower limits for targeted absorption lines
upper_limits = [2.0,2.5,3.5,4.5]   # upper limits for targeted absorption lines
# 3 CO2 lines
# lower_limits = [1.75,2.5,4.0]   # lower limits for targeted absorption lines
# upper_limits = [2.25,3.0,4.5]   # upper limits for targeted absorption lines
###    

h = np.zeros((3,len(upper_limits)))                     # scale heights
delta_h = np.zeros((3,(len(upper_limits))))              # peak amplitude

for j, case in enumerate(cases):
    # call spectrum function of object
    opa = jdi.opannection(wave_range=[0.6,5])
    df= case.spectrum(opa, full_output=True, calculation='transmission') 
    wno, rprs  = df['wavenumber'] , df['transit_depth']
    wno, rprs = jdi.mean_regrid(wno, rprs, R=150)
    full_output = df['full_output']
    waves = 1e4/wno         # convert to um
    waves = waves[::-1]     # reverse order

    transit = df['transit_depth']*1e6   # convert from ppm to R2/R*2

    for i,(lows,ups) in enumerate(zip(lower_limits,upper_limits)):
        low_index = np.where((waves > lows - 0.05) & (waves < lows + 0.05))[0].min()  # determine corresponding indices
        up_index = np.where((waves > ups - 0.05) & (waves < ups + 0.05))[0].min()       # determine corresponding indices

        feature1 = max(transit[low_index:up_index])             # find peak of absorption line
        reference = min(transit)                                 # define reference value as overall minimum
        deltadelta = feature1-reference                        # peak amplitude
       
        temp = (np.sqrt((rpla*ap.R_jup)**2 + (rstel*ap.R_sun)**2*(deltadelta)*1e-6) - rpla*ap.R_jup)/num_scale  
        h[j,i] = temp.value                                 # calculate scale height
        delta_h[j,i] = deltadelta                          # peak amplitude

heights = np.array([0.0, 0.0, 0.0])
stds = np.array([0.0, 0.0, 0.0])

for j in range(heights.size):
    for k in range(len(upper_limits)):
        heights[j] += delta_h[j,k]*h[j,k]       # weighted mean scale height

    heights[j] /= sum(delta_h[j,:])
    stds[j] = np.sqrt(np.sum(delta_h[j,:]*(h[j,:] - heights[j])**2)/np.sum(delta_h[j,:]))   # weighted std

heights *= 1.e-3    # convert to km
stds *= 1.e-3        # convert to km

print(heights)
print(stds)

