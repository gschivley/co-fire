
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:09:30 2014

@author: Greg
"""


import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.integrate import cumtrapz

# Radiative efficiencies of each gas, calculated from AR5 & AR5 SM
co2_re, ch4_re, n2o_re, sf6_re = 1.756E-15, 1.277E-13 * 1.65, 3.845E-13, 2.010E-11

# AR5 2013 IRF values
a0, a1, a2, a3 = 0.2173, 0.224, 0.2824, 0.2763
tau1, tau2, tau3 = 394.4, 36.54, 4.304

def f0(t):
    return a0
def f1(t):
    return a1*np.exp(-t/tau1)
def f2(t):
    return a2*np.exp(-t/tau2)
def f3(t):
    return a3*np.exp(-t/tau3)
def CO2_AR5(t):
    return f0(t) + f1(t) + f2(t) + f3(t)
    
#Methane response fuction
CH4tau = 12.4
def CH4_AR5(t):
    return np.exp(-t/CH4tau)
    
#N2O response fuction
N2Otau = 121
def N2O_AR5(t):
    return np.exp(-t/CH4tau)

#SF6 response fuction   
SF6tau = 3200
def SF6_AR5(t):
    return np.exp(-t/CH4tau)

#Temperature response function to radiative forcing
def AR5_GTP(t):
    c1, c2, d1, d2 = 0.631, 0.429, 8.4, 409.5
    """ The default response function for radiative forcing from AR5. Source is \
    Boucher (2008). ECR is 3.9K, which is on the high side.
    Convolve with radiative forcing to get temperature.
    """
    return c1/d1*np.exp(-t/d1) + c2/d2*np.exp(-t/d2)

def Alt_GTP(t):
    c1, c2, d1, d2 = 0.43, 0.32, 2.57, 82.24
    """ The response function for radiative forcing. Taken from Olivie and Peters (2013),
    Table 4, using the CMIP5 data. This has a slightly lower climate response value than
    Boucher (2008), which is used in AR5.

    Convolve with radiative forcing to get temperature.
    """
    return c1/d1*np.exp(-t/d1) + c2/d2*np.exp(-t/d2)

def Alt_low_GTP(t):
    c1, c2, d1, d2 = 0.43 / (1 + 0.29), 0.32 / (1 + 0.59), 2.57 * 1.46, 82.24 * 2.92
    #c1, c2, d1, d2 = 0.48 * (1 - 0.3), 0.20 * (1 - 0.52), 7.15 * 1.35, 105.55 * 1.38
    #c1, c2, d1, d2 = 0.48 * (1 - 0.3), 0.20 * (1 - 0.52), 7.15, 105.55
    #c1, c2, d1, d2 = 0.631 * 0.7, 0.429 * 0.7, 8.4, 409.5
    """ The response function for radiative forcing. Taken from Olivie and Peters (2013),
    Table 4, using the CMIP5 data. This has a lower climate response value than AR5.
    The uncertainty in Table 4 assumes lognormal distributions, which is why values less
    than the median are determined by dividing by (1 + uncertainty).

    Convolve with radiative forcing to get temperature.
    """
    return c1/d1*np.exp(-t/d1) + c2/d2*np.exp(-t/d2)

def Alt_high_GTP(t):
    c1, c2, d1, d2 = 0.43 * 1.29, 0.32 * 1.59, 2.57 / (1 + 0.46), 82.24 / (1 + 1.92)
    #c1, c2, d1, d2 = 0.48 * 1.3, 0.20 * 1.52, 7.15 * (1 - 0.35), 105.55 * (1 - 0.38)
    #c1, c2, d1, d2 = 0.48 * 1.2, 0.20 * 1.52, 7.15, 105.55
    #c1, c2, d1, d2 = 0.631, 0.429 * 1.3, 8.4, 409.5    
    """ The response function for radiative forcing. Taken from Olivie and Peters (2013),
    Table 4, using the CMIP5 data. This has a higher climate response value than AR5.
    The uncertainty in Table 4 assumes lognormal distributions, which is why values less
    than the median are determined by dividing by (1 + uncertainty).

    Convolve with radiative forcing to get temperature.
    """
    return c1/d1*np.exp(-t/d1) + c2/d2*np.exp(-t/d2)

def CO2_rf(emission, years, tstep=0.01, kind='linear'): 
    """Transforms an array of CO2 emissions into radiative forcing with user-
    defined time-step.
    
    emission: an array of emissions, should be same size as years
    years: an array of years at which the emissions take place
    tstep: time step to be used in the calculations
    kind: the type of interpolation to use; can be linear or cubic
    """
#emission is a series of emission numbers, years should match up with it
    if min(years) > 0:
        years = years - min(years)
    
    end = max(years) 
    f = interp1d(years, emission, kind=kind)
    time = np.linspace(years[0], end, end/tstep + 1)    
    inter_emissions = f(time)
    atmos = np.resize(fftconvolve(CO2_AR5(time), inter_emissions), time.size)
    rf = atmos * co2_re
    fil = np.zeros_like(time, dtype=bool)
    for i in time:
        if i == int(i):
            fil[i/tstep] = True
        
    return rf[fil]

def CO2_rate(emission, years, tstep=0.01, kind='linear'): 
    """Transforms an array of CO2 emissions into radiative forcing with user-
    defined time-step.
    
    emission: an array of emissions, should be same size as years
    years: an array of years at which the emissions take place
    tstep: time step to be used in the calculations
    kind: the type of interpolation to use; can be linear or cubic
    """
#emission is a series of emission numbers, years should match up with it
    if min(years) > 0:
        years = years - min(years)
    
    end = max(years) 
    f = interp1d(years, emission, kind=kind)
    time = np.linspace(years[0], end, end/tstep + 1)    
    inter_emissions = f(time)
    atmos = np.resize(fftconvolve(CO2_AR5(time), inter_emissions), time.size)
    rf = atmos * co2_re
    dx = np.gradient(time)
    rate = np.gradient(rf, dx)
    fil = np.zeros_like(time, dtype=bool)
    for i in time:
        if i == int(i):
            fil[i/tstep] = True
        
    return rate[fil]

def CO2_crf(emission, years, tstep=0.01, kind='linear'): 
    """Transforms an array of CO2 emissions into radiative forcing with user-
    defined time-step.
    
    emission: an array of emissions, should be same size as years
    years: an array of years at which the emissions take place
    tstep: time step to be used in the calculations
    kind: the type of interpolation to use; can be linear or cubic
    """
#emission is a series of emission numbers, years should match up with it
    if min(years) > 0:
        years = years - min(years)
    
    end = max(years) 
    f = interp1d(years, emission, kind=kind)
    time = np.linspace(years[0], end, end/tstep + 1)    
    inter_emissions = f(time)
    atmos = np.resize(fftconvolve(CO2_AR5(time), inter_emissions), time.size)
    rf = atmos * co2_re
    crf = cumtrapz(rf, dx = tstep, initial = 0)
    fil = np.zeros_like(time, dtype=bool)
    for i in time:
        if i == int(i):
            fil[i/tstep] = True
        
    return crf[fil]


def CO2_temp(emission, years, tstep=0.01, kind='linear', source='AR5'): 
    """Transforms an array of CO2 emissions into temperature with user-defined
    time-step. Default temperature IRF is from AR5, use 'Alt_low' or 'Alt_high'
    for a sensitivity test.
    
    emission: an array of emissions, should be same size as years
    years: an array of years at which the emissions take place
    tstep: time step to be used in the calculations
    kind: the type of interpolation to use; can be linear or cubic
    source: the source of parameters for the temperature IRF. default is AR5,
    'Alt', 'Alt_low', and 'Alt_high' are also options.
    """
    if min(years) > 0:
        years = years - min(years)
    
    end = max(years)       
    f = interp1d(years, emission, kind=kind, bounds_error=False)
    time = np.linspace(min(years), end, end/tstep + 1) 
    inter_emissions = f(time)
    atmos = np.resize(fftconvolve(CO2_AR5(time), inter_emissions), time.size)
    rf = atmos * co2_re
    if source == 'AR5':
        temp = np.resize(fftconvolve(AR5_GTP(time), rf), time.size) * tstep
    elif source == 'Alt':
        temp = np.resize(fftconvolve(Alt_GTP(time), rf), time.size)
    elif source == 'Alt_low':
        temp = np.resize(fftconvolve(Alt_low_GTP(time), rf), time.size)
    elif source == 'Alt_high':
        temp = np.resize(fftconvolve(Alt_high_GTP(time), rf), time.size)

    fil = np.zeros_like(time, dtype=bool)
    for i in time:
        if i == int(i):
            fil[i/tstep] = True
        
    return temp[fil]


def ch42co2(t, alpha=0.51):
    """As methane decays some fraction is converted to CO2. This function is 
    from Boucher (2009). By default it converts 51%. The convolution of this 
    function with the methane emission profile gives the CO2 emission profile.
    
    t: time
    alpha: fraction of methane converted to CO2
    """
    ch4tau = 12.4
    return 1/ch4tau * alpha * np.exp(-t/ch4tau)



def CH4_rf(emission, years, tstep=0.01, kind='linear',
             decay=True):
    """Transforms an array of methane emissions into radiative forcing with user-defined
    time-step.
    
    emission: an array of emissions, should be same size as years
    years: an array of years at which the emissions take place
    tstep: time step to be used in the calculations
    kind: the type of interpolation to use; can be linear or cubic
    """
    if min(years) > 0:
        years = years - min(years)
    
    end = max(years) 
    fch4 = interp1d(years, emission, kind=kind)
    time = np.linspace(years[0], end, end/tstep + 1)    
    ch4_inter_emissions = fch4(time)
    ch4_atmos = np.resize(fftconvolve(CH4_AR5(time), ch4_inter_emissions),
                          time.size)
    co2 = np.resize(fftconvolve(ch42co2(time), ch4_inter_emissions),
                    time.size)
    co2_atmos = np.resize(fftconvolve(CO2_AR5(time), co2),
                          time.size)
    
    if decay == True:
         rf = ch4_atmos * ch4_re + co2_atmos * co2_re
    else:
        rf = ch4_atmos * ch4_re
    fil = np.zeros_like(time, dtype=bool)
    for i in time:
        if i == int(i):
            fil[i/tstep] = True
    
    return rf[fil]
    
def CH4_rf_cc(emission, years, tstep=0.01, kind='linear',
             decay=True):
    """Transforms an array of methane emissions into radiative forcing with user-defined
    time-step, accounting for climate-carbon feedbacks.
    
    emission: an array of emissions, should be same size as years
    years: an array of years at which the emissions take place
    tstep: time step to be used in the calculations
    kind: the type of interpolation to use; can be linear or cubic
    """
    gamma = (44.0/12.0) * 10**12
    
    if min(years) > 0:
        years = years - min(years)
    
    end = max(years) 
    fch4 = interp1d(years, emission, kind=kind)
    time = np.linspace(years[0], end, end/tstep + 1)    
    ch4_inter_emissions = fch4(time)
    ch4_atmos = np.resize(fftconvolve(CH4_AR5(time), ch4_inter_emissions),
                          time.size)
    co2 = np.resize(fftconvolve(ch42co2(time), ch4_inter_emissions),
                    time.size) * tstep
    co2_atmos = np.resize(fftconvolve(CO2_AR5(time), co2),
                          time.size)
    cc_co2 = CH4_cc_tempforrf(emission, years) * gamma
    cc_co2_atmos = np.resize(fftconvolve(CO2_AR5(time), cc_co2),
                          time.size)
    
    if decay == True:
         rf = ch4_atmos * ch4_re + (co2_atmos +cc_co2_atmos) * co2_re
    else:
        rf = ch4_atmos * ch4_re + (cc_co2_atmos) * co2_re
    fil = np.zeros_like(time, dtype=bool)
    for i in time:
        if i == int(i):
            fil[i/tstep] = True
    
    return rf[fil]

def CH4_rate(emission, years, tstep=0.01, kind='linear'):
    """Transforms an array of methane emissions into radiative forcing with user-defined
    time-step, accounting for climate-carbon feedbacks.
    
    emission: an array of emissions, should be same size as years
    years: an array of years at which the emissions take place
    tstep: time step to be used in the calculations
    kind: the type of interpolation to use; can be linear or cubic
    """
    gamma = (44.0/12.0) * 10**12
    
    if min(years) > 0:
        years = years - min(years)
    
    end = max(years) 
    fch4 = interp1d(years, emission, kind=kind)
    time = np.linspace(years[0], end, end/tstep + 1)    
    ch4_inter_emissions = fch4(time)
    ch4_atmos = np.resize(fftconvolve(CH4_AR5(time), ch4_inter_emissions),
                          time.size)
    co2 = np.resize(fftconvolve(ch42co2(time), ch4_inter_emissions),
                    time.size)
    co2_atmos = np.resize(fftconvolve(CO2_AR5(time), co2),
                          time.size)
    cc_co2 = CH4_cc_tempforrf(emission, years) * gamma
    cc_co2_atmos = np.resize(fftconvolve(CO2_AR5(time), cc_co2),
                          time.size)
    
    rf = ch4_atmos * ch4_re + (co2_atmos +cc_co2_atmos) * co2_re
    dx = np.gradient(time)
    rate = np.gradient(rf, dx)
    fil = np.zeros_like(time, dtype=bool)
    for i in time:
        if i == int(i):
            fil[i/tstep] = True
    
    return rate[fil]

def CH4_crf(emission, years, tstep=0.01, kind='linear',
             decay=True):
    """Transforms an array of methane emissions into radiative forcing with user-defined
    time-step.
    
    emission: an array of emissions, should be same size as years
    years: an array of years at which the emissions take place
    tstep: time step to be used in the calculations
    kind: the type of interpolation to use; can be linear or cubic
    """
    if min(years) > 0:
        years = years - min(years)
    
    end = max(years) 
    fch4 = interp1d(years, emission, kind=kind)
    time = np.linspace(years[0], end, end/tstep + 1)    
    ch4_inter_emissions = fch4(time)
    ch4_atmos = np.resize(fftconvolve(CH4_AR5(time), ch4_inter_emissions),
                          time.size) * tstep
    co2 = np.resize(fftconvolve(ch42co2(time), ch4_inter_emissions),
                    time.size) * tstep
    co2_atmos = np.resize(fftconvolve(CO2_AR5(time), co2),
                          time.size) * tstep
    
    if decay == True:
         rf = ch4_atmos * ch4_re + co2_atmos * co2_re
    else:
        rf = ch4_atmos * ch4_re
    crf = cumtrapz(rf, dx = 1, initial = 0)
    fil = np.zeros_like(time, dtype=bool)
    for i in time:
        if i == int(i):
            fil[i/tstep] = True
    
    return crf[fil]

def CH4_crf_cc(emission, years, tstep=0.01, kind='linear',
             decay=True):
    """Transforms an array of methane emissions into radiative forcing with user-defined
    time-step.
    
    emission: an array of emissions, should be same size as years
    years: an array of years at which the emissions take place
    tstep: time step to be used in the calculations
    kind: the type of interpolation to use; can be linear or cubic
    """
    gamma = (44.0/12.0) * 10**12

    if min(years) > 0:
        years = years - min(years)
    
    end = max(years) 
    fch4 = interp1d(years, emission, kind=kind)
    time = np.linspace(years[0], end, end/tstep + 1)    
    ch4_inter_emissions = fch4(time)
    ch4_atmos = np.resize(fftconvolve(CH4_AR5(time), ch4_inter_emissions),
                          time.size) * tstep
    co2 = np.resize(fftconvolve(ch42co2(time), ch4_inter_emissions),
                    time.size) * tstep
    co2_atmos = np.resize(fftconvolve(CO2_AR5(time), co2),
                          time.size) * tstep
    cc_co2 = CH4_cc_tempforrf(emission, years) * gamma
    cc_co2_atmos = np.resize(fftconvolve(CO2_AR5(time), cc_co2),
                          time.size) * tstep
    
    if decay == True:
         rf = ch4_atmos * ch4_re + (co2_atmos +cc_co2_atmos) * co2_re
    else:
        rf = ch4_atmos * ch4_re + (cc_co2_atmos) * co2_re
    crf = cumtrapz(rf, dx = 1, initial = 0)
    fil = np.zeros_like(time, dtype=bool)
    for i in time:
        if i == int(i):
            fil[i/tstep] = True
    
    return crf[fil]

def CH4_temp(emission, years, tstep=0.01, kind='linear', source='AR5',
             decay=True): 
    """Transforms an array of methane emissions into temperature with user-defined
    time-step. Default temperature IRF is from AR5, use 'Alt_low' or 'Alt_high'
    for a sensitivity test.
    
    emission: an array of emissions, should be same size as years
    years: an array of years at which the emissions take place
    tstep: time step to be used in the calculations
    kind: the type of interpolation to use; can be linear or cubic
    source: the source of parameters for the temperature IRF. default is AR5,
    'Alt', 'Alt_low', and 'Alt_high' are also options.
    decay: a boolean variable for if methane decay to CO2 should be included
    """
    if min(years) > 0:
        years = years - min(years)
    
    end = max(years) 
    f = interp1d(years, emission, kind=kind)
    time = np.linspace(years[0], end, end/tstep + 1)    
    ch4_inter_emissions = f(time)
    ch4_atmos = np.resize(fftconvolve(CH4_AR5(time), ch4_inter_emissions),
                          time.size)
    co2 = np.resize(fftconvolve(ch42co2(time), ch4_inter_emissions),
                    time.size)
    co2_atmos = np.resize(fftconvolve(CO2_AR5(time), co2),
                          time.size) * tstep
    if decay == True:
         rf = ch4_atmos * ch4_re + co2_atmos * co2_re
    else:
        rf = ch4_atmos * ch4_re
    if source == 'AR5':
        temp = np.resize(fftconvolve(AR5_GTP(time), rf), time.size) * tstep
    elif source == 'Alt':
        temp = np.resize(fftconvolve(Alt_GTP(time), rf), time.size)
    elif source == 'Alt_low':
        temp = np.resize(fftconvolve(Alt_low_GTP(time), rf), time.size)
    elif source == 'Alt_high':
        temp = np.resize(fftconvolve(Alt_high_GTP(time), rf), time.size)

    fil = np.zeros_like(time, dtype=bool)
    for i in time:
        if i == int(i):
            fil[i/tstep] = True
    
    return temp[fil]

def CH4_cc_tempforrf(emission, years, tstep=0.01, kind='linear', source='AR5',
             decay=True): 
    """Transforms an array of methane emissions into temperature with user-defined
    time-step. Default temperature IRF is from AR5, use 'Alt_low' or 'Alt_high'
    for a sensitivity test.
    
    emission: an array of emissions, should be same size as years
    years: an array of years at which the emissions take place
    tstep: time step to be used in the calculations
    kind: the type of interpolation to use; can be linear or cubic
    source: the source of parameters for the temperature IRF. default is AR5,
    'Alt', 'Alt_low', and 'Alt_high' are also options.
    decay: a boolean variable for if methane decay to CO2 should be included
    """
    if min(years) > 0:
        years = years - min(years)
    
    end = max(years) 
    f = interp1d(years, emission, kind=kind)
    time = np.linspace(years[0], end, end/tstep + 1)    
    ch4_inter_emissions = f(time)
    ch4_atmos = np.resize(fftconvolve(CH4_AR5(time), ch4_inter_emissions),
                          time.size) * tstep
    co2 = np.resize(fftconvolve(ch42co2(time), ch4_inter_emissions),
                    time.size) * tstep
    co2_atmos = np.resize(fftconvolve(CO2_AR5(time), co2),
                          time.size) * tstep
    if decay == True:
         rf = ch4_atmos * ch4_re + co2_atmos * co2_re
    else:
        rf = ch4_atmos * ch4_re
    if source == 'AR5':
        temp = np.resize(fftconvolve(AR5_GTP(time), rf), time.size) * tstep
    elif source == 'Alt':
        temp = np.resize(fftconvolve(Alt_GTP(time), rf), time.size) * tstep
    elif source == 'Alt_low':
        temp = np.resize(fftconvolve(Alt_low_GTP(time), rf), time.size) * tstep
    elif source == 'Alt_high':
        temp = np.resize(fftconvolve(Alt_high_GTP(time), rf), time.size) * tstep

    fil = np.zeros_like(time, dtype=bool)
    for i in time:
        if i == int(i):
            fil[i/tstep] = True
    
    return temp

def CH4_temp_cc(emission, years, tstep=0.01, kind='linear', source='AR5',
             decay=True): 
    """Transforms an array of methane emissions into temperature with user-defined
    time-step. Default temperature IRF is from AR5, use 'Alt_low' or 'Alt_high'
    for a sensitivity test. Accounts for climate-carbon feedbacks.
    
    emission: an array of emissions, should be same size as years
    years: an array of years at which the emissions take place
    tstep: time step to be used in the calculations
    kind: the type of interpolation to use; can be linear or cubic
    source: the source of parameters for the temperature IRF. default is AR5,
    'Alt', 'Alt_low', and 'Alt_high' are also options.
    decay: a boolean variable for if methane decay to CO2 should be included
    """
    gamma = (44.0/12.0) * 10**12
    
    if min(years) > 0:
        years = years - min(years)
    
    end = max(years) 
    f = interp1d(years, emission, kind=kind)
    time = np.linspace(years[0], end, end/tstep + 1)    
    ch4_inter_emissions = f(time)
    ch4_atmos = np.resize(fftconvolve(CH4_AR5(time), ch4_inter_emissions),
                          time.size)
    co2 = np.resize(fftconvolve(ch42co2(time), ch4_inter_emissions),
                    time.size)
    co2_atmos = np.resize(fftconvolve(CO2_AR5(time), co2),
                          time.size)
    cc_co2 = CH4_cc_tempforrf(emission, years) * gamma
    cc_co2_atmos = np.resize(fftconvolve(CO2_AR5(time), cc_co2),
                          time.size)
	
    if decay == True:
         rf = ch4_atmos * ch4_re + (co2_atmos + cc_co2_atmos) * co2_re
    else:
        rf = ch4_atmos * ch4_re + cc_co2_atmos * co2_re
    if source == 'AR5':
        temp = np.resize(fftconvolve(AR5_GTP(time), rf), time.size)
    elif source == 'Alt':
        temp = np.resize(fftconvolve(Alt_GTP(time), rf), time.size) * tstep
    elif source == 'Alt_low':
        temp = np.resize(fftconvolve(Alt_low_GTP(time), rf), time.size) * tstep
    elif source == 'Alt_high':
        temp = np.resize(fftconvolve(Alt_high_GTP(time), rf), time.size) * tstep

    fil = np.zeros_like(time, dtype=bool)
    for i in time:
        if i == int(i):
            fil[i/tstep] = True
    
    return temp[fil]