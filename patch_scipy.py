import scipy.integrate
import numpy as np

# Monkey-patch: add 'trapz' to 'scipy.integrate' if missing
if not hasattr(scipy.integrate, 'trapz'):
    scipy.integrate.trapz = np.trapz

# Monkey-patch: add 'simps' to 'scipy.integrate' if missing
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

