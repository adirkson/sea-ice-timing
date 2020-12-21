from NCGR import ncgr, sitdates
import numpy as np

# load data
X_all = np.load('Data/fud_X.npy') 
Y_all = np.load('Data/fud_Y.npy') 

# define time variables
event='fud' 
years_all = np.arange(1979, 2017+1)
im = 12 # initialization month

# instantiate the sitdates class
si_time = sitdates.sitdates(event=event)
a = si_time.get_min(im) # minimum date possible
b = si_time.get_max(im) # maximum date possible

# the forecast year to calibrate
t=2016 

# set up relevant time variables for calibrating the forecast
# using data only from previous
t_idx = np.where(years_all==t)[0][0]
tau = years_all[years_all<t] # years used for training
X_tau = X_all[years_all<t,:] # training hindcasts
Y_tau = Y_all[years_all<t] # training observations
X_t = X_all[t_idx] # forecast to be calibrated
Y_t = Y_all[t_idx] # the observed freeze-up date corresponding to the forecast

############## calibration  #######################
# instatiate the ncgr_gridpoint module
ngp = ncgr.ncgr_gridpoint(a, b)
# build calibration model
predictors_tau, predictors_t, coeffs0 = ngp.build_model(X_t, X_tau, Y_tau, tau, t)
# optimize/minmize the CRPS to estimate regression coefficients
coeffs = ngp.optimizer(Y_tau, predictors_tau, coeffs0)
# Compute calibrated forecast distribution parameters for year t
mu_t, sigma_t = ngp.forecast_mode(predictors_t, coeffs)

###### Compute forecast probabilities relative to climatology #################
# instantiate the fcst_vs_clim module
fvc = ncgr.fcst_vs_clim(a,b) 

n_clim=10 # number of years preceding the forecast to use as climatology (note: this is not the training period)
Y_clim = Y_all[t_idx-n_clim:t_idx] # observations for defining the climatology (past 10 years for this example) 
# get result
result = fvc.event_probs(mu_t, sigma_t, Y_clim)
# unpack result
fcst_probs, clim_terc, clim_params = result.probs, result.terciles, result.params