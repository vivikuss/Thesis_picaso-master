import numpy as np
import patch_scipy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os

### ADJUST THIS PART ###
# Paths
os.environ['picaso_refdata'] = '/Users/new/Desktop/THESIS/THESIS_picaso-master/reference'
os.environ['PYSYN_CDBS'] = '/Users/new/Desktop/THESIS/THESIS_picaso-master' 
plot_labels = [r'99% He, 1% CO$_2$', r'50% He, 50% CO$_2$',r'99% H$_2$, 1% CO$_2$'] # plot labels
######

import pandas as pd
import astropy.constants as ap
import astropy.units as apu

# PICASO
from picaso import justdoit as jdi
from picaso import justplotit as jpi

# PICASO plotting
jpi.output_notebook()

### CUSTOMIZE IF NEEDED ###
### K2-18b ###
# # Planet's Properties
# mpla = 0.0272           # Mjup ##### NASA Archive Benneke 2019
# rpla = 0.2328           # Rjup ##### NASA Archive Benneke 2019
# ps = 20

# # Star's Properties
# tstel = 3500            # K               #### NASA Archive 
# met = 0.123             # dex           #### NASA Archive 
# g = 4.858               # log g value       #### NASA Archive 
# rstel = 0.44            # Rsun              #### NASA Archive 
# mag = 8.899

# output_names = ['spectrum_k218b_case1.txt','spectrum_k218b_case2.txt','spectrum_k218b_case3.txt']   # output file names
# gridlines = np.array([2950,3050,3150,3250,3350,3450])  # gridlines for K2-18b
# num_scale = 4                       # number of contributing scale heights
# plottitle = r'Transmission Spectra, K2-18b, 1.8$\%$ CH$_4$, at 20 bar'
# filename_out = 'all_spectra_plot_K218b.png'

### LHS 1140b ###
# Planet's Properties
mpla = 0.0176           # Mjup ##### Wunderlich 2020/Cherubim 2025
rpla = 0.1543            # Rjup ##### Wunderlich 2020/Cherubim 2025
ps = 20

# Star's Properties
tstel = 3500             # K               #### NASA Archive
met = -0.15            # dex           #### NASA Archive
g = 5.0               # log g value       #### NASA Archive
rstel = 0.21              # Rsun              #### NASA Archive
mag = 9.612                             #### NASA Archive
output_names = ['spectrum_lhs_case1.txt','spectrum_lhs_case2.txt','spectrum_lhs_case3.txt']   # output file names
gridlines = np.array([5700,5800,5900,6000,6100,6200])  # gridlines for LHS 114b
num_scale = 3                       # number of contributing scale heights
plottitle = r'Transmission Spectra, LHS 1140b, at 20 bar'
filename_out = 'all_spectra_plot_LHS1140b.png'

##################

case_obj = [jdi.inputs() for i in range(3)]
filenames = [jdi.model1p_pt(), jdi.model50p_pt(), jdi.modelh2_pt()]        
colors = ['deeppink', 'crimson', 'navy']

cases = [case_obj[0],case_obj[1],case_obj[2]]  

### CUSTOMIZE IF NEEDED ###
# 2 CO2 lines, 2 CH4 lines    
# lower_limits = [1.5,2.0,3.0,4.0]   # lower limits for targeted absorption lines
# upper_limits = [2.0,2.5,3.5,4.5]   # upper limits for targeted absorption lines
# 3 CO2 lines
lower_limits = [1.75,2.5,4.0]   # lower limits for targeted absorption lines
upper_limits = [2.25,3.0,4.5]   # upper limits for targeted absorption lines
###
h = np.zeros((3,len(upper_limits)))                     # scale heights
delta_h = np.zeros((3,(len(upper_limits))))              # peak amplitude

plt.figure()

for j, (case, file, output, col, lab) in enumerate(zip(case_obj, filenames, output_names, colors, plot_labels)):
    opa = jdi.opannection(wave_range=[0.6,5])
    case.phase_angle(0)

    # specify gravity through planet's properties
    case.gravity(mass = mpla, mass_unit=jdi.u.Unit('M_jup'),  
                radius = rpla, radius_unit=jdi.u.Unit('R_jup'))  

    # specifiy star
    case.star(opannection = opa, temp = tstel, metal = met, logg = g, radius = rstel, radius_unit = jdi.u.Unit('R_sun'))  

    # atmosphere
    case.atmosphere(filename = file, sep=r'\s+') 
    case.approx(p_reference = ps)   # estimated surface pressure 

    # call spectrum function on object
    df= case.spectrum(opa, full_output=True,calculation='transmission') 
    wno, rprs  = df['wavenumber'] , df['transit_depth']
    wno, rprs = jdi.mean_regrid(wno, rprs, R=150)
    full_output1 = df['full_output']

    data = np.column_stack((1e4/wno, rprs))     # convert to um and stack
    data = data[data[:, 0].argsort()]               # resort array
    np.savetxt(output, data, fmt="%.6f", comments='')

    plt.plot(1e4/wno,rprs*1e6,color=col,label=lab)     

    waves = 1e4/wno         # convert to um
    waves = waves[::-1]     # reverse order

    transit = df['transit_depth']*1e6   # convert from ppm to Rp^2/R*^2

    for i,(lows,ups) in enumerate(zip(lower_limits,upper_limits)):
        low_index = np.where((waves > lows - 0.05) & (waves < lows + 0.05))[0].min()  # determine corresponding indices
        up_index = np.where((waves > ups - 0.05) & (waves < ups + 0.05))[0].min()       # determine corresponding indices

        feature1 = max(transit[low_index:up_index])             # find peak of absorption line
        reference = min(transit)                                 # define reference value as overall minimum
        deltadelta = feature1-reference                        # peak amplitude
       
        temp = (np.sqrt((rpla*ap.R_jup)**2 + (rstel*ap.R_sun)**2*(deltadelta)*1e-6) - rpla*ap.R_jup)/num_scale  
        h[j,i] = temp.value                                 # calculate scale height
        delta_h[j,i] = deltadelta                          # peak amplitude

plt.yticks(gridlines) 
plt.grid(True)
plt.title(plottitle)
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel('Transit Depth (ppm)')

### CUSTOMIZE IF NEEDED ###
# K2-18b
#plt.ylim([2950,3450])  
# LHS 1140b        
plt.ylim([5700,6200])            
###

plt.legend(loc='best')
plt.savefig(filename_out, dpi=500)
#plt.show()

heights = np.array([0.0, 0.0, 0.0])
stds = np.array([0.0, 0.0, 0.0])

for j in range(heights.size):
    for k in range(len(upper_limits)):
        heights[j] += delta_h[j,k]*h[j,k]       # weighted mean scale height

    heights[j] /= sum(delta_h[j,:])
    stds[j] = np.sqrt(np.sum(delta_h[j,:]*(h[j,:] - heights[j])**2)/np.sum(delta_h[j,:]))   # weighted std

heights *= 1.e-3    # convert to km
stds *= 1.e-3        # convert to km

print('Scale heights: ', heights)
print('Standard deviation: ', stds)
