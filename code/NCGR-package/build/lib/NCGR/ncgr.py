#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:55:31 2020

@author: arlan
"""
from dcnorm import dcnorm_gen
#from NCGR.dcnorm import dcnorm_gen

import numpy as np
from scipy.stats import norm, pearsonr
from scipy.stats.mstats import hdquantiles
from scipy.signal import detrend
from scipy.optimize import minimize
from netCDF4 import Dataset
import netCDF4 as nc4
import os, time, sys
from collections import namedtuple



# Function for progress bar
def update_progress(progress):
    # update_progress() : Displays or updates a console progress bar
    ## Accepts a float between 0 and 1. Any int will be converted to a float.
    ## A value under 0 represents a 'halt'.
    ## A value at 1 or bigger represents 100%
    barLength = 40 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent Completed: [{0}] {1}% {2}".format( "#"*block + " "*(barLength-block), np.around(progress*100,2), status)
    sys.stdout.write(text)
    sys.stdout.flush()


eps_sigma = 1e-6 # to constrain \sigma>=1e-6 during minimization
eps_mu = 1. # to constrain a-1<=\mu<=b+1 during minimization

def build_cons(predictors, y, a, b):
    '''
    Builds a dictionary for the constrainst on the DCNORM distribution parameters
    when calling on :py:class:`scipy.optimize.minimize` in the 
    :py:meth:`calibrate_fullfield`.
    
    Returns:
        cons (dict):
            Contains the constraint callables used in :py:class:`scipy.optimize.minimize`.
    '''
    
    N_pred_m = predictors[0].shape[0] # number of predictors for mu
    N_pred_s = predictors[1].shape[0] # number of predictors for sigma
    
    def con_mu1(params, predictors,y):    
        params_mu = params[:N_pred_m]
            
        predictors_mu = predictors[0]
                               
        mu_hat = np.dot(params_mu.T,predictors_mu)
        
        return mu_hat - (a-eps_mu)

    def con_mu2(params, predictors,y):    
        params_mu = params[:N_pred_m]
            
        predictors_mu = predictors[0]
                               
        mu_hat = np.dot(params_mu.T,predictors_mu)
        
        return b + eps_mu - mu_hat
    
    def con_std(params, predictors,y):
        params_s = params[N_pred_m:N_pred_m+N_pred_s]
        
        predictors_s = predictors[1]    

        s_hat = np.dot(params_s.T,predictors_s)     
        
        return s_hat - eps_sigma
    
    cons = ({'type': 'ineq', 'fun': con_mu1, 'args':(predictors,y)},
            {'type': 'ineq', 'fun': con_mu2, 'args':(predictors,y)},
            {'type': 'ineq', 'fun': con_std, 'args':(predictors,y)})

    # cons = ({'type': 'ineq', 'fun': con_std, 'args':(predictors,y)})
    
    return cons

class ncgr_gridpoint():
    '''
    Args:
        a (float or int):
            Minimum possible date for the event in 
            day-of-year units; e.g. 1=Jan 1, 91=April 1, 365=Dec 31). See definition 
            in [1]. IFDs or FUDs provided in input NetCDF files should be
            set to the value provided to ``a`` when the event has occurred at the time
            of forecast initialization.
            
        b (float or int):
            Maximum possible date for the event in 
            day-of-year units; e.g. 1=Jan 1, 91=April 1, 365=Dec 31). See definition 
            in [1]. IFDs or FUDs provided in input NetCDF files should be
            set to the value provided to ``b`` when the event does not occur by the end
            of the season.        


        Notes
        -----
        The following provides a brief description of NCGR; for a full description, see [1]_.
        
        NCGR assumes the observed IFD or FUD, :math:`Y(t)` (a random variable), conditioned
        on the ensemble forecast :math:`x_1(t),...,x_n(t)` follows a DCNORM distribution --
        i.e. :math:`Y(t)|x_1(t),...,x_n(t)\sim N_{dc}(\mu(t),\sigma(t))`. The parameter :math:`\mu`
        is modelled as
         
            .. math::        
               \mu(t) = \alpha_1\mu_{c}(t) + \alpha_2 x_{\langle i \rangle}^{d}(t)
               
        The user can choose one of the following equations for modelling the paremter :math:`\sigma`
    
            .. math::               
               \sigma_{I}(t) &=\beta_1\sigma_{c}, \\ 
               \sigma_{II}(t) &=\beta_1\sigma_{c}+\beta_2 s_x(t), \\ 
               \sigma_{III}(t) &=\beta_1\sigma_{c}+\beta_2 x_{\langle i \rangle}^{tc}(t)
    
        through the ``sigma`` argument, but by default :math:`\sigma=\sigma_{III}`.  
    
    
        The relevant method contained in this class is:
             
        ``calibrate_gridpoint()``
            Performs NCGR on the forecast IFD or FUD at a single gridpoint.
            
            
        %(after_notes)s
        
        References
        ----------
        .. [1] Dirkson, A.​, B. Denis., M.,Sigmond., & Merryfield, W.J. (2020). Development and Calibration of SeasonalProbabilistic Forecasts of Ice-free Dates and Freeze-up Dates. ​Weather and Forecasting​. doi:10.1175/WAF-D-20-0066.1.
    

    '''


    def __init__(self, a, b):
        self.a = a
        self.b = b
        
        
    def build_model(self, X_t, X, Y, tau_t, t, sigma_eqn='s3', pred_pval=0.05):
        r'''
        Args:
            X_t (ndarray), shape (`n,`):
                Forecast to be calibrated, where `n` is the ensemble size.
                
            X (ndarray), shape (`N,n`):
                Forecasts for training NCGR, where `N` is the number of years
                for training and `n` is the ensemble size for a 
                given forecast.
            
            Y (ndarray), shape (`N,`):
                Observations for training NCGR, where `N` is the number of years
                for training.
                
            tau_t (ndarray), shape (`N,`):
                Years corresponding to those used for training period (this should not
                contain the forecast year). The year should be based on the initialization 
                date, not the date of the IFD or FUD event.
                
            t (float or int):
                The year in which the forecast is initialized.            

            sigma_eqn (str, optional): 
                Refers to the regression equation to be used for the :math:`\sigma`
                parameter in the NCGR model. This can be one of 's1', 's2', or 's3'.
                These are defined by the regression equations below as :math:`\sigma_{I}`,
                :math:`\sigma_{II}`, and :math:`\sigma_{III}`, respectively. By default,
                ``sigma='s3'`` (i.e. :math:`\sigma_{III}`). 

            pred_pval (float, optional):
                The p-value for determining statistical significance of the second 
                predictor for the  :math:`\sigma_{II}` or :math:`\sigma_{III}`
                regression equations. By default, ``pred_pval=0.05``. 
           
        Returns:
            predictors_tau (ndarray or ndarray object), shape (2,2,N) or (2,)
                Contains the predictors for both mu and sigma over the training 
                period. It's an object
                if mu and sigma have different numbers of predictors (i.e. when
                there is no second term in the equation for sigma). It's an array
                if mu and sigma have the same number of predictors. Regardless, the 
                following is always true for the first two entries:
            
                predictors_tau[0] (ndarray), shape (2, N):
                    The predictors for mu over the training period

                predictors_tau[1] (ndarray), shape (1, N) or (2, N):
                    The predictor(s) for sigma over the training period
                    
            predictors_t (ndarray or object array), shape (2,2) or shape (2,)
                Contains the predictors for both mu and sigma for the forecast year
                t. It's an object
                if mu and sigma have different numbers of predictors (i.e. when
                there is no second term in the equation for sigma). It's an array
                if mu and sigma have the same number of predictors. Regardless, the 
                following is always true for the first two entries:
                    
                predictors_tau[0] (ndarray), shape (2,):
                    The predictors for mu for the forecast year t

                predictors_tau[1] (ndarray), shape (1,) or (2,):
                    The predictor(s) for sigma for the forecast year t
                    
            coeffs0 (array), shape (3,) or (4,):
                Initial guesses for the coefficients in the NCGR regression equations.
            

        '''        
        a, b = self.a, self.b
        
        t_all = np.sort(np.append(np.array(t),tau_t))
        N_t_all = len(t_all) # number of years in t_all
        tau_ind = np.where(t_all!=t)[0] # indices of t_all array corresponding to tau_t
        t_ind = np.where(t_all==t)[0][0] # index of t_all array corresponding to t
                  
        X_all_curr = np.copy(X)
        X_all_curr = np.insert(X_all_curr, t_ind, X_t, axis=0)
        Y_curr = np.copy(Y)
                
        if np.all(Y_curr==Y_curr[0]): # check if array elements are constant
            pval_y = 999.
        else:
            pval_y = pearsonr(tau_t,Y_curr)[1]
            
        if pval_y<0.05:
            coeffs = np.polyfit(tau_t, Y_curr, deg=1)
            fp = np.poly1d(coeffs)
            mu_clim = fp(t_all)
            mu_clim[mu_clim>b] = b
            mu_clim[mu_clim<a] = a
        else:
            mu_clim = Y_curr.mean()*np.ones(N_t_all)

        std_clim = np.ones(N_t_all)*detrend(Y_curr).std(ddof=1) 
 
                  
        ############# Predictors for mu ###############################
                
        X_d = detrend(X_all_curr.mean(axis=1))     
        X_d[mu_clim+X_d<a] = a - mu_clim[mu_clim+X_d<a]
        X_d[mu_clim+X_d>b] = b - mu_clim[mu_clim+X_d>b]
        predictors_all_mu = np.array([mu_clim,
                                          X_d])


        predictors_tau_mu = predictors_all_mu[:,tau_ind]
        predictors_t_mu = predictors_all_mu[:,t_ind]  


        ############# Predictors for sigma ##############################           
        predictors_all_std = np.array([std_clim])
        
        # trend-corrected hindcasts (note: values on [a,b] by construction of X_d)
        X_tc = np.around(mu_clim + X_d,4) 
        # unbiased ensemble-mean error
        error = np.abs(X_tc[tau_ind] - Y_curr)   
        
        if sigma_eqn=='s1': # no second predictor for sigma
            None 
            
        elif sigma_eqn=='s2': # predictor is the ensemble standard deviation 
            pred = np.std(X_all_curr,ddof=1,axis=1)
            
            if np.all(pred[tau_ind]==pred[tau_ind][0]) or np.all(error==error[0]):
                p_val_x = 999.
            else:
                p_val_x = pearsonr(pred[tau_ind], error)[1]
            
            if p_val_x<pred_pval:
                predictors_all_std = np.concatenate((predictors_all_std, np.array([pred])),axis=0)                                
            else:
                None
        
        elif sigma_eqn=='s3':
            pred = X_tc # predictor is the trend-corrected ensemble mean
            if np.all(pred[tau_ind]==pred[tau_ind][0]) or np.all(error==error[0]):
                p_val_x = 999.
            else:
                p_val_x = pearsonr(pred[tau_ind], error)[1]
                
            if p_val_x<pred_pval:
                predictors_all_std = np.concatenate((predictors_all_std, np.array([pred])),axis=0)            
            else:
                None
       
        predictors_tau_std = predictors_all_std[:,tau_ind]
        predictors_t_std = predictors_all_std[:,t_ind]                                             


        predictors_tau = np.array([predictors_tau_mu,predictors_tau_std]) 
        predictors_t = np.array([predictors_t_mu,predictors_t_std]) 


        N_pred_mu = predictors_tau[0].shape[0]
        N_pred_s = predictors_tau[1].shape[0] 

        # set initial parameter guesses 
        coeffs0 = np.concatenate((np.ones(N_pred_mu),np.ones(N_pred_s)))
        
        if N_pred_s==1:
            None
        elif N_pred_s==2:
            coeffs0[-1] = std_clim[0]/np.mean(predictors_all_std[1,:])
                
        return predictors_tau, predictors_t, coeffs0


    def optimizer(self, Y, predictors_tau, coeffs0, es_tol=0.05, options=None):    
        r'''
        Function for minimizing the CRPS over N forecasts. For this, 
        :py:class:`scipy.optimize.minimize(method=’SLSQP’)` is called to 
        minimize the analytic expression for the CRPS when the distribution under
        consideration for the forecast is the DCNORM distribution.
        
        Args:                 
            Y (ndarray), shape (`N,`):
                Observations for training NCGR, where `N` is the number of years
                for training.
                
            predictors_tau (ndarray or object of ndarray), shape (`2,`) or (`2,4,N`)
                Predictors over the training period. If an object of ndarray, then
                `predictors_tau[0]` has shape (`2,N`) corresponding to predictors for
                :math:`mu`, and `predictors_tau[1]` has shape
                (`N,`) corresponding to the single predictor for :math:`sigma`.
                
            coeffs0 (array), shape (3,) or (4,):
                Initial guesses for the coefficients in the NCGR regression equations. 
            
            es_tol (float or None, optional):
                Early stopping threshold used for minimizing the CRPS. 
                By default ``es_tol=0.05``. Specifically, this argument
                sets the ``tol`` argument in :py:class:`scipy.optimize.minimize(method=’SLSQP’)`.  

            options (dict, optional): 
                A dictionary of options to pass to :py:method:'scipy.optimize.minimize' corresponding to 
                its ``options`` argument (see :py:class:`scipy.optimize.minimize(method=’SLSQP’)`).                 
    
        Returns:
            coeffs (array):
                Optimized regression coefficients.
        

        '''
        
        # instantiate the two classes needed
        crps_funcs_ = crps_funcs(self.a, self.b)
        ############################################################  
        res_beinf = minimize(crps_funcs_.crps_ncgr, coeffs0, args=(predictors_tau,Y),
                             jac=crps_funcs_.crps_ncgr_jac,
                             tol=es_tol,
                             options=options, 
                             constraints=build_cons(predictors_tau,Y, self.a, self.b))
        
        if np.isnan(res_beinf.fun) or res_beinf.fun==np.inf or res_beinf.fun<0.0:
            print("Minimization couldn't converge - this shouldn't ever happen; if it does,"+ 
                  "please contact arlan.dirkson@gmail.com with the following information:")
            print("predictors for mu", predictors_tau[0])
            print("predictors for sigma",predictors_tau[1])
            print("initial guesses for coefficients", coeffs0)
            raise(ValueError)
            
        else:
            None

        return res_beinf.x

    def forecast_mode(self, predictors, coeffs):
        N_pred_mu = len(predictors[0])
        N_pred_s = len(predictors[1])
        
        coeffs_mu, coeffs_std = coeffs[0:N_pred_mu], coeffs[N_pred_mu:N_pred_mu+N_pred_s]
        
        mu_cal = min(max(self.a-eps_mu,np.dot(coeffs_mu.T,predictors[0])),self.b+eps_mu) # constrain to [a-eps_mu,b+eps_mu]
        sigma_cal = max(eps_sigma,np.dot(coeffs_std.T,predictors[1])) # constrain to [eps_sigma,inf]
           
        return mu_cal, sigma_cal


class ncgr_fullfield:    
    r'''    

    * Performs non-homogeneous censored gaussian regression (NCGR) [1]
      on a forecast ice-free date (IFD) or freeze-up date (FUD) field using input
      NetCDF files.
    
    * An output NetCDF file is created that contains several quantities relevant to calibration.

    Args:
        fcst_netcdf (str):
            Path of the NetCDF file containing the ensemble forecast IFDs or FUDs to be calibrated. 
            Requirements for the file are as follows:
                         
            * The structure/shape of the IFD/FUD variable should be: 
              (time, ensemble members, latitude, longitude), although the actual names of those dimensions 
              and corresponding variables (if they are included in the NetCDF files) can be anything.

            * Relevant variable names and dimension names are to be specified by the user with the 
              ``model_dict`` argument provided to this function. 
            
            * The variable for the time coordinate, which should represent the forecast initialization date,
              should follow CF conventions (https://cfconventions.org/).
              In particular, it's necessary for the initialization year be retrievable from the time 
              variable using the :meth:`netCDF4.num2date` function (https://unidata.github.io/netcdf4-python/netCDF4/index.html). 
              
            * Masked locations will be ignored in calibration and will be written as masked locations
              in the output NetCDF file. 

        hc_netcdf (str):
            Path of the NetCDF file
            containing the model ensemble forecast IFDs or FUDs used to train NCGR. This file should
            contain several years of model re-forecasts (hindcasts), and may in fact include data for years after the 
            year of the forecast being calibrated (e.g. if leave-one-out cross-validation is being performed).
            The requirements for the file are the same as for ``fcst_netcdf``, except that the variable corresponding
            to the time coordinate should contain several dates (i.e. the initialization date for each forecast).
        
        obs_netcdf (str):
            Path of the NetCDF file
            containing the observed IFDs or FUDs used to train NCGR. This file should
            contain several years of observed IFDs/FUDs (corresponding to what was verified for
            the model hindcasts in ``hc_netcdf``). Requirements for the file are as follows:
            
            * Relevant variable names and dimension names are to be specified by the user with the 
              ``obs_dict`` argument provided to this function.
              
            * The structure/shape of the IFD/FUD variable should be: 
              (time, latitude, longitude), although the actual names of those dimensions 
              and corresponding variables (if they are included in the NetCDF files) can be anything.
            
            * The variable for the time coordinate, which should represent the initialization dates of the re-forecasts
              in ``hc_netcdf``, should follow CF conventions (https://cfconventions.org/).
              In particular, it's necessary for the initialization year be retrievable from the time 
              variable using the :meth:`netCDF4.num2date` function (https://unidata.github.io/netcdf4-python/netCDF4/index.html). 

            * Masked locations will be ignored in calibration and will be written as masked locations
              in the output NetCDF file. 

        out_netcdf (str):
            The absolute path of the NetCDF file to
            be written that contains relevant calibrated forecast quantities. For more information on this file
            see :py:meth:`ncgr_fullfield.write_output`.
            
        a (float or int):
            Minimum possible date for the event in 
            day-of-year units; e.g. 1=Jan 1, 91=April 1, 365=Dec 31). See definition 
            in [1]. IFDs or FUDs provided in input NetCDF files should be
            set to the value provided to ``a`` when the event has occurred at the time
            of forecast initialization.
            
        b (float or int):
            Maximum possible date for the event in 
            day-of-year units; e.g. 1=Jan 1, 91=April 1, 365=Dec 31). See definition 
            in [1]. IFDs or FUDs provided in input NetCDF files should be
            set to the value provided to ``b`` when the event does not occur by the end
            of the season.

        model_dict (tuple(dict,dict)):
            A tuple of two dictionaries. The first specifies the
            relevant variables in the NetCDF files 
            assigned to ``fcst_netcdf`` and ``hc_netcdf``. The second specifies the
            relevant demension names in those same files
            
            For the first dictionary, its keys (which must follow the naming conventions used below) 
            and values (specified by user) are:
                
                'event_vn': ``str``: Variable name for the ice-free date or freeze-up date field(s)
            
                'time_vn': ``str``: Variable name for the forecast initialization date
                
            For the second dictionary, its keys (which must follow the naming conventions used below) 
            and values (specified by user) are:           

                'time_dn': ``str``: Dimension name for the forecast initialization date
                
                'ens_dn': ``str``: Dimension name for the forecast realization/ensemble member

            Example:
                .. highlight:: python
                .. code-block:: python
                                
                    model_dict = ({'event_vn' : 'ifd',
                                   'time_vn' : 'time'},
                                  {'time_dn' : 'time',
                                   'ens_dn' : 'ensemble'})
                    
        obs_dict (tuple(dict,dict)):
            A tuple of two dictionaries. The first specifies the
            relevant variables in the NetCDF file
            assigned to ``obs_netcdf``. The second specifies the
            relevant demension names in that same file.
            
            For the first dictionary, its keys (which must follow the naming conventions used below) 
            and values (specified by user) are:
                
                'event_vn': ``str``: Variable name for ice-free date or freeze-up date field(s)
            
                'time_vn': ``str``: Variable name for the forecast initialization date
                
            For the second dictionary, its keys (which must follow the naming conventions used below) 
            and values (specified by user) are:           

                'time_dn': ``str``: Dimension name for the forecast initialization date

            Example:
                .. highlight:: python
                .. code-block:: python
                                
                    obs_dict = ({'event_vn' : 'ifd', 
                                   'time_vn' : 'time'},
                                  {'time_dn' : 'time'}) 

        clim_netcdf (str, optional):
            The absolute path of the NetCDF file that contains several years of observed IFDs or FUDs
            used to construct the climatology that forecast IFDs/FUDs will be in
            reference to. If this is included, forecast probabilities for each the 
            early, near-normal, and late events, as well as ensemble-mean anomalies are 
            included in the ``out_netcdf`` file. Requirements for this file are the same as those
            for ``obs_netcdf``.
            
        terc_interp (str or None, optional):
            Interpolation scheme used to compute the terciles for the observed climatology. Default is None.
            Can be one of the following:
                * None: By default y_clim data are fit to the DCNORM distribution,
                and terciles are computed using its :py:meth:`_ppf` method.
                
                * 'HD': Estimate terciles using the Harrell-Davis estimator (see :py:class:scipy.stats.mstats.hdquantiles)
                
                * 'nearest-rank': Nearest rank or rank order method (see 
                https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method)
                
                * Any of the interpolation arguments for :py:class:`numpy.percentile`.
             
        sigma_eqn (str, optional): 
            Refers to the regression equation to be used for the :math:`\sigma`
            parameter in the NCGR model. This can be one of 's1', 's2', or 's3'.
            These are defined by the regression equations below as :math:`\sigma_{I}`,
            :math:`\sigma_{II}`, and :math:`\sigma_{III}`, respectively. By default,
            ``sigma_eqn='s3'`` (i.e. :math:`\sigma_{III}`). 
             
        es_tol (float or None, optional):
            Early stopping threshold used for minimizing the CRPS. 
            By default ``es_tol=0.05``. Specifically, this argument
            sets the ``tol`` argument in :py:class:`scipy.optimize.minimize(method=’SLSQP’)`.  

        pred_pval (float, optional):
            The p-value for determining statistical significance of the second 
            predictor for the  :math:`\sigma_{II}` or :math:`\sigma_{III}`
            regression equations. By default, ``pred_pval=0.05``.
            
        options (dict, optional): 
            A dictionary of options to pass to :py:method:'scipy.optimize.minimize' corresponding to 
            its ``options`` argument (see :py:class:`scipy.optimize.minimize(method=’SLSQP’)`).  

    
    Notes
    -----
    The following provides a brief description of NCGR; for a full description, see [1].
    
    NCGR assumes the observed IFD or FUD, :math:`Y(t)` (a random variable), conditioned
    on the ensemble forecast :math:`x_1(t),...,x_n(t)` follows a DCNORM distribution --
    i.e. :math:`Y(t)|x_1(t),...,x_n(t)\sim N_{dc}(\mu(t),\sigma(t))`. The parameter :math:`\mu`
    is modelled as
     
        .. math::        
           \mu(t) = \alpha_1\mu_{c}(t) + \alpha_2 x_{\langle i \rangle}^{d}(t)
           
    The user can choose one of the following equations for modelling the paremter :math:`\sigma`

        .. math::               
           \sigma_{I}(t) &=\beta_1\sigma_{c}, \\ 
           \sigma_{II}(t) &=\beta_1\sigma_{c}+\beta_2 s_x(t), \\ 
           \sigma_{III}(t) &=\beta_1\sigma_{c}+\beta_2 x_{\langle i \rangle}^{tc}(t)

    through the ``sigma`` argument, but by default :math:`\sigma=\sigma_{III}`.  


    The relevant methods contained in this class are:
         
    ``calibrate_fullfield()``
        Performs NCGR on the forecast IFD or FUD field. 
        
    ``write_output()``
        Creates a NetCDF file containing several relevant fields to the 
        NCGR-calibrated forecast.
    
    References
    ----------
    .. [1] Dirkson, A.​, B. Denis., M.,Sigmond., & Merryfield, W.J. (2020). Development and Calibration of SeasonalProbabilistic Forecasts of Ice-free Dates and Freeze-up Dates. ​Weather and Forecasting​. doi:10.1175/WAF-D-20-0066.1.

    '''

    def __init__(self, fcst_netcdf, hc_netcdf, obs_netcdf, out_netcdf, a, b, model_dict, obs_dict, 
                 clim_netcdf=None, terc_interp=None, sigma_eqn='s3', es_tol=5e-2, 
                 pred_pval=0.05, options=None):
        
        
        # path and forecast file obects needed for writing output
        self.out_netcdf = out_netcdf
        self.fcst_netcdf = fcst_netcdf 
        

        # get hindcasts, observations, and forecast IFDs or FUDs from netcdf files        
        file_hc = Dataset(hc_netcdf)
        file_obs = Dataset(obs_netcdf)
        file_fcst = Dataset(fcst_netcdf)


        self.model_dict = model_dict
        self.obs_dict = obs_dict
        
        self.X = file_hc.variables[self.model_dict[0]['event_vn']][:]
        self.Y = file_obs.variables[self.obs_dict[0]['event_vn']][:]
        self.X_t = file_fcst.variables[self.model_dict[0]['event_vn']][:][0]
        
        # Set fill value to apply in masked locations
        self.fill_value = 9.969209968386869e+36
        self.prob_undefined = np.nan
        
        # get spatial dimensions from netcdf (use forecast for this)        
        self.nrow = self.X_t.shape[-2]
        self.ncol = self.X_t.shape[-1]
        
        # get time variables relevant to the forecast
        fcst_time_var = file_fcst.variables[self.model_dict[0]['time_vn']]
        fcst_time_var = nc4.num2date(fcst_time_var[:], units=fcst_time_var.units, calendar=fcst_time_var.calendar)
        self.t = fcst_time_var[0].year
        
        # get time variables relevant to training 
        hc_time_var = file_hc.variables[self.model_dict[0]['time_vn']]
        hc_time_var = nc4.num2date(hc_time_var[:], units=hc_time_var.units, calendar=hc_time_var.calendar)
        self.tau = np.array([]).astype('int')
        for curr_time in hc_time_var:            
            self.tau = np.append(self.tau, curr_time.year)
        
        # all years
        self.t_all = np.sort(np.append(np.array(self.t),self.tau)) # array of all years included in both `tau` and `t`        

        self.a = a # minimum date possible
        self.b = b # maximum date possible
       
        self.clim_netcdf = clim_netcdf
        if self.clim_netcdf:
            file_clim = Dataset(self.clim_netcdf)
            self.Y_clim = file_clim.variables[self.obs_dict[0]['event_vn']][:] #dimensions (time, grid rows, grid columns)
        
        # optional agrumetns
        self.terc_interp = terc_interp
        self.sigma_eqn = sigma_eqn 
        self.es_tol = es_tol
        self.pred_pval = pred_pval
        
        # Perform NCGR calibration
        result = self.calibrate_fullfield()
        # save output to NetCDF
        self.write_output(result)           
        
        

    def calibrate_fullfield(self):
        '''
        Loops through the spatial grid and performs NCGR
        at each grid point. 
        
        Returns:
            result (ndarray object):
                An object array. If ``clim_netcdf`` is included as an argument to :py:class:`ncgr_fullfield`,
                ``results`` is an object containing six arrays corresponding to the following variables:
                    mu_cal (ndarray), shape (`nrow`, `ncol`):
                        Calibrated :math:`\mu` parameter for the predictive 
                        DCNORM distribution, where `nrow` is the number of rows and
                        `ncol` is the number of columns provided as spatial coordinates
                        in the NetCDF files given to :py:class:`ncgr_fullfield`.
                        
                    sigma_cal (ndarray), shape (`nrow`, `ncol`):
                        Calibrated :math:`\sigma` parameter for the predictive 
                        DCNORM distribution, where `nrow` and `ncol` are defined as
                        in ``mu_cal``.
                        
                    fcst_probs (ndarray), shape (`3`, `nrow`, `ncol`):
                        The three forecast probabilities for early, near-normal, and late
                        sea-ice retreat/advance, where `nrow` and `ncol` are defined as
                        in ``mu_cal``.                

                    fcst_pre (ndarray), shape (`nrow`, `ncol`):
                        The probability for the pre-occurrence of the event. 

                    fcst_non (ndarray), shape (`nrow`, `ncol`):
                        The probability for the non-occurrence of the event. 
        
                    clim_terc (ndarray), shape (`2`, `nrow`, `ncol`):
                        The two terciles (i.e. 1/3 and 2/3 quantiles) for the observed climatology,
                        where `nrow` and `ncol` are defined as
                        in ``mu_cal``.                
        
                    mean (float):
                        Expected value of the forecast DCNORM distribution.
                        
                    mean_anom (float):
                        Anomaly of ``mean`` relative to climatology.
                        
                If climatology is not included as an argument to :py:class:`ncgr_fullfield`, ``results`` contains two
                arrays corresponding to the following variables:
                    mu_cal (ndarray), shape (`nrow`, `ncol`):
                        Calibrated :math:`\mu` parameter for the predictive 
                        DCNORM distribution, where `nrow` is the number of rows and
                        `ncol` is the number of columns provided as spatial coordinates
                        in the NetCDF files given to :py:class:`ncgr_fullfield`.
                        
                    sigma_cal (ndarray), shape (`nrow`, `ncol`):
                        Calibrated :math:`\sigma` parameter for the predictive 
                        DCNORM distribution, where `nrow` and `ncol` are defined as
                        in ``mu_cal``.                     

                    fcst_pre (ndarray), shape (`nrow`, `ncol`):
                        The probability for the pre-occurrence of the event. 

                    fcst_non (ndarray), shape (`nrow`, `ncol`):
                        The probability for the non-occurrence of the event. 


        '''

        mu_cal = np.zeros((self.nrow,self.ncol))
        sigma_cal = np.zeros((self.nrow,self.ncol))
        fcst_probs = np.zeros((3,self.nrow,self.ncol))
        clim_terc = np.zeros((2,self.nrow,self.ncol))
        ens_mean = np.zeros((self.nrow,self.ncol))
        ens_mean_anom = np.zeros((self.nrow,self.ncol))
        clim_params = np.zeros((2, self.nrow, self.ncol))
        fcst_pre = np.zeros((self.nrow,self.ncol))
        fcst_non = np.zeros((self.nrow,self.ncol))
        
        ngp = ncgr_gridpoint(self.a, self.b)
        fvc = fcst_vs_clim(self.a, self.b)
        
        count = 0.
        for row in np.arange(self.nrow):
            for col in np.arange(self.ncol):  
                # check if any of the data at this grid cell are masked - if so set all outputs to masked values
                if np.ma.is_masked(self.X_t[:,row,col]) or np.ma.is_masked(self.X[:,:,row,col]) or np.ma.is_masked(self.Y[:,row,col]):
                    if self.clim_netcdf:
                        fcst_probs[:,row,col], clim_terc[:,row,col], clim_params[:,row,col] = self.fill_value, self.fill_value, self.fill_value
                        ens_mean[row,col], ens_mean_anom[row,col] = self.fill_value, self.fill_value
                    else:
                        None
                        
                    mu_cal[row,col], sigma_cal[row,col] = self.fill_value, self.fill_value
                    fcst_pre[row,col], fcst_non[row,col] = self.fill_value, self.fill_value
                    
                # if not masked
                else:
                    # all obs show event has always already occurred at the time of initializatoin
                    if np.all(self.Y[:,row,col]==self.a):
                        mu_cal[row,col], sigma_cal[row,col] = self.a - eps_mu, eps_sigma
                        
                    # all obs show the event never occurrs over the course of the forecast/season
                    elif np.all(self.Y[:,row,col]==self.b):
                        mu_cal[row,col], sigma_cal[row,col] = self.b + eps_mu, eps_sigma
                    
                    # else calibrate
                    else:                    
                        # build model
                        predictors_tau, predictors_t, coeffs0 = ngp.build_model(self.X_t[:,row,col], self.X[:,:,row,col], self.Y[:,row,col], 
                                                                                self.tau, self.t,
                                                                                self.sigma_eqn, self.pred_pval)
                        # minimize CRPS and get regression coefficients
                        coeffs = ngp.optimizer(self.Y[:,row,col], predictors_tau, coeffs0, self.es_tol)
                        
                        # apply real-time predictors and coefficients to get calibrated distribution for the forecast
                        mu_cal[row,col], sigma_cal[row,col] = ngp.forecast_mode(predictors_t, coeffs)
                                            
                    fcst_pre[row,col], fcst_non[row,col] = fvc.fcst_prenon(mu_cal[row,col], sigma_cal[row,col])
                                           
                    if self.clim_netcdf:
                        fcst_probs[:,row,col], clim_terc[:,row,col], clim_params[:,row,col] = fvc.event_probs(mu_cal[row,col], sigma_cal[row,col], 
                                                                                                     self.Y_clim[:,row,col],
                                                                                                     self.terc_interp)
                        
                        ens_mean[row,col], ens_mean_anom[row,col] = fvc.fcst_deterministic(mu_cal[row,col], sigma_cal[row,col], 
                                                                                                     self.Y_clim[:,row,col])
                    else:
                        None
                                               
                # update progress bar
                update_progress(float(count)/(self.nrow*self.ncol))
                count+=1
                
         
        if self.clim_netcdf:
            if self.terc_interp==None:
                result = np.array([mu_cal, sigma_cal, fcst_probs, fcst_pre, fcst_non, clim_terc, ens_mean, ens_mean_anom, clim_params])
            else:
                result = np.array([mu_cal, sigma_cal, fcst_probs, fcst_pre, fcst_non, clim_terc, ens_mean, ens_mean_anom])
        else:
            result = np.array([mu_cal, sigma_cal])

        return result
    
    def write_output(self, result): 
        '''
        Writes the variables provided as arguments to this function to the NetCDF file ``out_netcdf``,
        an argument provided to :py:class:`ncgr_fullfield`. Note that this will remove and replace any previous version of 
        ``out_netcdf``.
        
        Args:
            mu_cal (ndarray), shape (`nrow`, `ncol`):
                Calibrated :math:`\mu` parameter for the predictive 
                DCNORM distribution, where `nrow` is the number of rows and
                `ncol` is the number of columns provided as spatial coordinates
                in the NetCDF files given to :py:meth:`ncgr_fullfield`.
                
            sigma_cal (ndarray), shape (`nrow`, `ncol`):
                Calibrated :math:`\sigma` parameter for the predictive 
                DCNORM distribution, where `nrow` and `ncol` are defined as
                in ``mu_cal``.
                
            fcst_probs (ndarray), shape (`3`, `nrow`, `ncol`):
                The three forecast probabilities for early, near-normal, and late
                sea-ice retreat/advance, where `nrow` and `ncol` are defined as
                in ``mu_cal``.    
                
            fcst_pre (ndarray), shape (`nrow`, `ncol`):
                Probability for the pre-occurrence of the event.
                
            fcst_non (ndarray), shape (`nrow`, `ncol`):
                Probability for the non-occurrence of the event.
                
            clim_terc (ndarray), shape (`2`, `nrow`, `ncol`):
                The two terciles (i.e. 1/3 and 2/3 quantiles) for the observed climatology,
                where `nrow` and `ncol` are defined as
                in ``mu_cal``.                

            ens_mean (float):
                Expected value of the forecast DCNORM distribution.
                
            ens_mean_anom (float):
                Anomaly of ``ens_mean`` relative to the climatological mean.
        '''
        if self.clim_netcdf:
            if self.terc_interp==None:
                mu_cal, sigma_cal, fcst_probs, fcst_pre, fcst_non, clim_terc, ens_mean, ens_mean_anom, clim_params = result
                mu_clim, sigma_clim = clim_params
            else:
                mu_cal, sigma_cal, fcst_probs, fcst_pre, fcst_non, clim_terc, ens_mean, ens_mean_anom = result
        else:
            mu_cal, sigma_cal = result
            
        # Load raw forecast netcdf to get the time variable
        dsin = Dataset(self.fcst_netcdf)
        
        # check if output path/file already exist; if so, delete it
        if os.path.exists(self.out_netcdf):
            os.remove(self.out_netcdf)
        else:
            None
        
        # Write file
        dsout = Dataset(self.out_netcdf,'w',format='NETCDF4_CLASSIC')
        
        # copy dimensions from original file to new file, except the dimension for the ensemble
        for dname, the_dim in dsin.dimensions.items():
            if dname==self.model_dict[1]['ens_dn']:
                None
            else:
                dsout.createDimension(dname, the_dim.size)  
                
        dnames_keep = [] # empty list to be filled with th dimensions that were kept (i.e. all but the ensemble dimension)
        for dname in dsin.variables[self.model_dict[0]['event_vn']].dimensions:
            if dname==self.model_dict[1]['ens_dn']:
                None
            else:
                dnames_keep.append(dname)
                
        dnames_keep = tuple(dnames_keep) # convert from list to tuple for passing into the createVariable function
                
        # copy all variables from original forecast file to new file
        # except the raw ifd or fud field; in place of the raw ifd or fud field,
        # write several new variables to the file
        for v_name, varin in dsin.variables.items(): 
            if v_name==self.model_dict[0]['event_vn']:
                # Write new variables
                outVar1 = dsout.createVariable('mu_cal', np.float32, dnames_keep, fill_value=self.fill_value)
                outVar1.long_name = 'calibrated mu parameter for DCNORM distribution'
                outVar1.valid_min = self.a - eps_mu
                outVar1.valid_max = self.b + eps_mu            
                outVar1[:] = mu_cal.astype(np.float32)              

                outVar2 = dsout.createVariable('sigma_cal', np.float, dnames_keep, fill_value=self.fill_value)
                outVar2.long_name = 'calibrated sigma parameter for DCNORM distribution'
                outVar2.valid_min = eps_sigma
                outVar2.valid_max = np.inf
                outVar2[:] = sigma_cal.astype(np.float) 
            
                if self.clim_netcdf:
                    outVar3 = dsout.createVariable('prob_EN', np.float32, dnames_keep, 
                                                   fill_value=self.fill_value)
                    outVar3.long_name = 'calibrated probability for early '+self.model_dict[0]['event_vn']
                    outVar3.valid_min = 0.0
                    outVar3.valid_max = 1.0
                    outVar3.units = "1"
                    outVar3[:] = fcst_probs[0].astype(np.float32) 
    
                    outVar4 = dsout.createVariable('prob_NN', np.float32, dnames_keep, fill_value=self.fill_value)
                    outVar4.long_name = 'calibrated probability for normal '+self.model_dict[0]['event_vn']
                    outVar4.valid_min = 0.0
                    outVar4.valid_max = 1.0
                    outVar4.units = "1"
                    outVar4[:] = fcst_probs[1].astype(np.float32) 
    
                    outVar5 = dsout.createVariable('prob_LN', np.float32, dnames_keep, fill_value=self.fill_value)
                    outVar5.long_name = 'calibrated probability for late '+self.model_dict[0]['event_vn']
                    outVar5.valid_min = 0.0
                    outVar5.valid_max = 1.0
                    outVar5.units = "1"
                    outVar5[:] = fcst_probs[2].astype(np.float32) 

                    outVar6 = dsout.createVariable('prob_pre', np.float32, dnames_keep, fill_value=self.fill_value)
                    outVar6.long_name = 'calibrated probability for the pre-occurrence of the '+self.model_dict[0]['event_vn']
                    outVar6.valid_min = 0.0
                    outVar6.valid_max = 1.0
                    outVar6.units = "1"
                    outVar6[:] = fcst_pre.astype(np.float32) 
            
                    outVar7 = dsout.createVariable('prob_non', np.float32, dnames_keep, fill_value=self.fill_value)
                    outVar7.long_name = 'calibrated probability for the non-occurrence of the '+self.model_dict[0]['event_vn']
                    outVar7.valid_min = 0.0
                    outVar7.valid_max = 1.0
                    outVar7.units = "1"
                    outVar7[:] = fcst_non.astype(np.float32) 
                
                    outVar8 = dsout.createVariable('clim_1_3',np.float32, dnames_keep, fill_value=self.fill_value)
                    outVar8.long_name = 'observed climatological 1/3 '+self.model_dict[0]['event_vn']+' quantile'
                    outVar8.valid_min = self.a
                    outVar8.valid_max = self.b
                    outVar8[:] = clim_terc[0].astype(np.float32) 
    
                    outVar9 = dsout.createVariable('clim_2_3', np.float32, dnames_keep, fill_value=self.fill_value)
                    outVar9.long_name = 'observed climatological 2/3 '+self.model_dict[0]['event_vn']+' quantile'
                    outVar9.valid_min = self.a
                    outVar9.valid_max = self.b
                    outVar9[:] = clim_terc[1].astype(np.float32) 
    
                    outVar10 = dsout.createVariable('ens_mean',np.float32, dnames_keep, fill_value=self.fill_value)
                    outVar10.long_name = 'calibrated ensemble mean '+self.model_dict[0]['event_vn']
                    outVar10.valid_min = self.a
                    outVar10.valid_max = self.b
                    outVar10[:] = ens_mean.astype(np.float32) 
    
                    outVar11 = dsout.createVariable('ens_mean_anom', np.float32, dnames_keep, fill_value=self.fill_value)
                    outVar11.long_name = 'calibrated ensemble mean '+self.model_dict[0]['event_vn']+' anomaly relative to climatology'
                    outVar11[:] = ens_mean_anom.astype(np.float32) 

                    if self.terc_interp==None:
                        outVar12 = dsout.createVariable('mu_clim', np.float32, dnames_keep, fill_value=self.fill_value)
                        outVar12.long_name = 'mu parameter for DCNORM distribution fit to climatology'
                        outVar12.valid_min = self.a - eps_mu
                        outVar12.valid_max = self.b + eps_mu            
                        outVar12[:] = mu_clim.astype(np.float32)              
        
                        outVar13 = dsout.createVariable('sigma_clim', np.float, dnames_keep, fill_value=self.fill_value)
                        outVar13.long_name = 'sigma parameter for DCNORM distribution fit to climatology'
                        outVar13.valid_min = eps_sigma
                        outVar13.valid_max = np.inf
                        outVar13[:] = sigma_clim.astype(np.float) 
                
            else:
                # keeps time and space coordinate variables
                outVar = dsout.createVariable(v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                outVar[:] = dsin.variables[v_name][:]

        dsin.close()
        dsout.close()
        
        return None             


class crps_funcs():
    '''
    This class contains functions needed to perform CRPS minimization for the DCNORM distribution.
    It also contains a function for computing the CRPS when the forecast distribution
    takes the form of a DCNORM distribution (as it does for NCGR).
    
    Args:
        a (float or int):
            Minimum possible date for the event in non leap year
            day-of-year units; e.g. 1=Jan 1, 91=April 1, 365=Dec 31). A value
            larger than 365 is regarded as a date for the following year.
            
        b (float or int):
            Maximum possible date for the event in non leap year 
            day-of-year units; e.g. 1=Jan 1, 91=April 1, 365=Dec 31). A value
            larger than 365 is regarded as a date for the following year. The 
            ``b`` argument must be larger than the ``a`` argument.

    
    The methods contained in this class are:

    ``crps_dcnorm()``
        Computes the CRPS for a set of forecsts and observations
        when the predictive distribution takes the form of a 
        DCNORM distribution.
                
    ``crps_ncgr()``
        The cost function used when executing ``scipy.optimize.mimize``
        in the ``calibrate`` method. Computes the mean CRPS as a function of a set
        of hindcast CDFs (modelled by NCGR) and observed dates.
        
    ``crps_ncgr_jac()``
        Called on in the ``calibrate`` method. 
        Computes the jacobian matrix for the CRPS cost function.
        
    ``crps_singleyear()``
        Called on in the ``calibrate`` method. 
        Computes the CRPS for a single forecast CDF (modelled as a DCNORM
        distribution) and observation.
           
    '''
    
    def __init__(self,a,b):
        self.a = a
        self.b = b   
  

    def crps_dcnorm_single(self,y,mu,sigma):
        '''
        Continuous rank probability score (CRPS) for a single forecast when the distribution
        takes the form of a DCNORM distribution.

        Args:
            y (float or int):
                Observed date.
            
            mu (float or int):
                DCNORM parameter :math:`\mu`.
                
            sigma (float or int):
                DCNORM parameter :math:`\sigma`
                
        Returns:
            result (float):
                CRPS
                
        '''

        rv = norm()        
        a_star = (self.a-mu)/sigma
        b_star = (self.b-mu)/sigma
        y_star = (y-mu)/sigma
    
        t1 = -sigma*(a_star*rv.cdf(a_star)**2. + 2*rv.cdf(a_star)*rv.pdf(a_star) -1./np.sqrt(np.pi)*rv.cdf(np.sqrt(2)*a_star))
        t2 = sigma*(b_star*rv.cdf(b_star)**2. + 2*rv.cdf(b_star)*rv.pdf(b_star) -1./np.sqrt(np.pi)*rv.cdf(np.sqrt(2)*b_star))
        t3 = 2*sigma*(y_star*rv.cdf(y_star) +rv.pdf(y_star)) - 2*sigma*(b_star*rv.cdf(b_star) +rv.pdf(b_star)) 
        t4 = sigma*(b_star - y_star)
        
        result = t1 + t2 + t3 + t4
    
        return result[0]     

    def crps_dcnorm(self,y,mu,sigma):
        '''
        Time mean continuous rank probability score (CRPS) when the distribution
        takes the form of a DCNORM distribution.

        Args:
            y (ndarray), shape (`n`,):
                Observed dates, where `n` is the number of
                forecast/observation pairs.
            
            mu (ndarray), shape (`n`,):
                DCNORM parameter :math:`\mu` for each of the `1,...,n` forecast distributions.
                
            sigma (ndarray), shape (`n`,):
                DCNORM parameter :math:`\sigma` for each of the `1,...,n` forecast distributions.
                
        Returns:
            result (float):
                Time mean CRPS.
                
        '''

        N = len(y)
        crps = np.zeros(N)
        rv = norm()
        for ii in np.arange(N):           
            a_star = (self.a-mu[ii])/sigma[ii]
            b_star = (self.b-mu[ii])/sigma[ii]
            y_star = (y[ii]-mu[ii])/sigma[ii]
        
            t1 = -sigma[ii]*(a_star*rv.cdf(a_star)**2. + 2*rv.cdf(a_star)*rv.pdf(a_star) -1./np.sqrt(np.pi)*rv.cdf(np.sqrt(2)*a_star))
            t2 = sigma[ii]*(b_star*rv.cdf(b_star)**2. + 2*rv.cdf(b_star)*rv.pdf(b_star) -1./np.sqrt(np.pi)*rv.cdf(np.sqrt(2)*b_star))
            t3 = 2*sigma[ii]*(y_star*rv.cdf(y_star) +rv.pdf(y_star)) - 2*sigma[ii]*(b_star*rv.cdf(b_star) +rv.pdf(b_star)) 
            t4 = sigma[ii]*(b_star - y_star)
            
            crps[ii] = t1 + t2 + t3 + t4
        
        result = np.mean(crps)
        return result
    
    
    def crps_ncgr(self, coeffs, predictors, y):  
        '''
        Args:            
            coeffs (list), shape (`m`,):
                Coefficients in the NCGR regression equations, 
                where `m` is the total number of coefficients/predictors. The first two values
                correspond to those for :math:`\mu` and the remaining values
                correspond to those for :math:`\sigma`.
                
                
            predictors (object), shape (`n`,):
                Object containing the predictors, where `n=2` is the number of distribution parameters.
                The shape of either predictors[0] or predictors[1] is (`m,p`), where
                `m` is the number of coefficients/predictors for the corresponding parameter, and `p` is the number of
                years in the training period ``self.tau_t``.
                
            y (ndarray), shape (`p`,):
                Array of observed dates, where `p` is the number of
                years in the training period ``self.tau_t``.
                
        Returns:
                The time-averaged continuous rank probability score (CRPS).

        '''
        
        N_pred_m = predictors[0].shape[0] # number of predictors for mu
        N_pred_s = predictors[1].shape[0] # numebr of preidctors for sigma
        
        # get the coefficients and predictors for the regression equation for mu
        params_m = coeffs[:N_pred_m]     
        predictors_m = predictors[0]
        
        # get the coefficients and predictors for the regression equation for sigma
        params_s = coeffs[N_pred_m:N_pred_m+N_pred_s]
        predictors_s = predictors[1]    
        
        mu = np.dot(params_m.T,predictors_m) # take linear combination of preidictors and coeffs for mu
        sigma = np.dot(params_s.T,predictors_s) # "" "" for sigma
    
        return self.crps_dcnorm(y,mu,sigma)
    
    def crps_ncgr_jac(self, coeffs, predictors, y):
        '''
        Args:
            coeffs (list), shape (`n+m`):
                Coefficients in the NCGR regression equations, 
                where `n=2` is the number of distribution parameters 
                and `m` is the number of predictors for a given parameter. The first
                two values are the coefficients for :math:`\mu` and the 
                remaining values are the coefficeints for :math:`\sigma`.
                
                
            predictors (object), shape (`n`,):
                Object containing the predictors, where `n=2` is the number of distribution parameters.
                The shape of either predictors[0] or predictors[1] is (`m,p`), where
                `m` is the number of coefficients/predictors for the corresponding parameter, and `p` is the number of
                years in the training period ``self.tau_t``.
                
            y (ndarray), shape (`p`):
                Array of observed dates, where `p` is the number of
                years in the training period ``self.tau_t``.
                
        Returns:
            (ndarray), shape (m,):
                The jacobian matrix of the time-averaged continuous rank probability score.
        '''

        N = len(y) # number of years the CRPS will averaged over
        N_pred_m = predictors[0].shape[0] # number of predictors for mu
        N_pred_s = predictors[1].shape[0] # numebr of preidctors for sigma
        
        # get the coefficients and predictors for the regression equation for mu
        params_m = coeffs[:N_pred_m]     
        predictors_m = predictors[0]
        
        # get the coefficients and predictors for the regression equation for sigma
        params_s = coeffs[N_pred_m:N_pred_m+N_pred_s]
        predictors_s = predictors[1]    
        
        mu = np.dot(params_m.T,predictors_m) # take linear combination of preidictors and coeffs for mu
        sigma = np.dot(params_s.T,predictors_s) # "" "" for sigma

        def T_mu(z):
            return rv.cdf(z)**2. + 2*rv.pdf(z)**2. 
        
        def T_std(z):
            return z*rv.cdf(z)**2. + 2*z*rv.pdf(z)**2. 

        rv = norm()
        jac = np.zeros((N,N_pred_m+N_pred_s))
    
        for ii in np.arange(N):        
            a_star = (self.a-mu[ii])/sigma[ii]
            b_star = (self.b-mu[ii])/sigma[ii]
            y_star = (y[ii]-mu[ii])/sigma[ii]

        
            jac_mu = T_mu(a_star) - T_mu(b_star) \
                    +np.sqrt(2.)/np.sqrt(np.pi) * (rv.pdf(b_star*np.sqrt(2)) - rv.pdf(a_star*np.sqrt(2))) \
                    +2.*(rv.cdf(b_star) - rv.cdf(y_star))
                    
                    
            jac_std = self.crps_ncgr_sy(np.array([mu[ii],sigma[ii]]), y[ii])/sigma[ii] + T_std(a_star) - T_std(b_star) \
                      +np.sqrt(2.)/np.sqrt(np.pi) * (b_star*rv.pdf(np.sqrt(2)*b_star) - a_star*rv.pdf(np.sqrt(2)*a_star)) \
                      +2.*(b_star*rv.cdf(b_star) - y_star*rv.cdf(y_star)) \
                      + y_star - b_star
                    
            jac[ii,:N_pred_m] = predictors_m[:,ii]*jac_mu
            jac[ii,N_pred_m:] = predictors_s[:,ii]*jac_std
                
    
        return np.mean(jac,axis=0)    
    
    def crps_ncgr_sy(self, params, y):
        '''
        Computes the continuous rank probability score
        for a single forecast with DCNORM distribution.
        
        Args:     
            params (list), shape (2,):
                List containing the two DCNORM distribution parameters 
                :math:`\mu` and :math:`\sigma`.
                
            y (float):
                The observation.
                
        Returns:
            result (float):
                The CRPS.

        '''        
        mu, sigma = params.T
        rv = norm()
        a_star = (self.a-mu)/sigma
        b_star = (self.b-mu)/sigma
        y_star = (y-mu)/sigma
        
        t1 = -sigma*(a_star*rv.cdf(a_star)**2. + 2*rv.cdf(a_star)*rv.pdf(a_star) -1./np.sqrt(np.pi)*rv.cdf(np.sqrt(2)*a_star))
        t2 = sigma*(b_star*rv.cdf(b_star)**2. + 2*rv.cdf(b_star)*rv.pdf(b_star) -1./np.sqrt(np.pi)*rv.cdf(np.sqrt(2)*b_star))
        t3 = 2*sigma*(y_star*rv.cdf(y_star) +rv.pdf(y_star)) - 2*sigma*(b_star*rv.cdf(b_star) +rv.pdf(b_star)) 
        t4 = sigma*(b_star - y_star)
        
        result = t1 + t2 + t3 + t4    
        return result
    
    
class fcst_vs_clim():
    '''
    Contains functions for computing forecast quantities relative to observed climataology.
    
    Args:
        a (float or int):
            Minimum possible date for the event in non leap year
            day-of-year units; e.g. 1=Jan 1, 91=April 1, 365=Dec 31). A value
            larger than 365 is regarded as a date for the following year.
            
        b (float or int):
            Maximum possible date for the event in non leap year 
            day-of-year units; e.g. 1=Jan 1, 91=April 1, 365=Dec 31). A value
            larger than 365 is regarded as a date for the following year. The 
            ``b`` argument must be larger than the ``a`` argument.
        
        fill_value (float):
            The flag value given to an event probability when it doesn't make sense to compute one; this
            occurs when the climatological terciles are equal to each other, for example.
                   
    '''
    
    def __init__(self,a,b):
        self.a = a
        self.b = b

    def event_probs(self, mu, sigma, y_clim, terc_interp=None):
        '''
        Computes the forecast probabilities for an early, normal, or late
        event relative to some defined climatology. 
        
        Args:
            mu (float):
                The mu parameter for the DCNORM distribution
            
            sigma (float):
                The sigma parameter for the DCNORM distribution
                
            y_clim (ndarray), shape (`N,`):
                Array of climatological dates, where `N` is the number of dates (equiavalently years)
                used to compute climatological statistics.

            terc_interp (str or None, optional):
                Interpolation scheme used to compute the terciles for the observed climatology. Default is None.
                Can be one of the following:
                    * None: By default y_clim data are fit to the DCNORM distribution,
                    and terciles are computed using its :py:meth:`_ppf` method.
                    
                    * 'HD': Estimate terciles using the Harrell-Davis estimator (see :py:class:scipy.stats.mstats.hdquantiles)
                    
                    * 'nearest-rank': Nearest rank or rank order method (see 
                    https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method)
                    
                    * Any of the interpolation arguments for :py:class:`numpy.percentile`.                
                
        Returns:
            result (object ndarray):
                An object array containing 2 arrays. The first array has shape (3,) and contains the forecast
                probabilities for being the event occuring early, near-normal, or late, respectively. The
                second array has shape (2,) and contains the climatological terciles deliniating 
                the event categories.
        '''
        dcnorm = dcnorm_gen(a=self.a,b=self.b) # create a generic DCNORM distribution object
        rv = dcnorm(mu, sigma)   # freeze a DCNORM distribution object with parameters mu and sigma


        if terc_interp is None:
            if np.all(y_clim==self.a):
                terc_low, terc_high = self.a, self.a
                mu_clim, sigma_clim = self.a-eps_mu, eps_sigma
            elif np.all(y_clim==self.b):
                terc_low, terc_high = self.b, self.b
                mu_clim, sigma_clim = self.b+eps_mu, eps_sigma                
            else:
                mu_clim, sigma_clim = dcnorm.fit(y_clim)
                rv_clim = dcnorm(mu_clim,sigma_clim)
                terc_low, terc_high = rv_clim.ppf(1./3.), rv_clim.ppf(2./3.)
        
        elif terc_interp=='HD': # use Harrell-Davis estimator
            if np.std(y_clim)==0.0:
                terc_low, terc_high = y_clim[0], y_clim[0]
            else:
                terc_low, terc_high = hdquantiles(y_clim, [1./3.,2./3.]) 
                terc_low = max(min(terc_low,self.b),self.a)
                terc_high = max(min(terc_high,self.b),self.a)
            
        elif terc_interp=='nearest-rank':
            if np.std(y_clim)==0.0:
                terc_low, terc_high = y_clim[0], y_clim[0]
            else:
                terc_low, terc_high = np.sort(y_clim)[int(np.ceil(1./3.*len(y_clim)))-1], np.sort(y_clim)[int(np.ceil(2./3.*len(y_clim)))-1]
        
        else:
            if np.std(y_clim)==0.0:
                terc_low, terc_high = y_clim[0], y_clim[0]
            else:
                terc_low, terc_high = np.percentile(y_clim, [100.*1./3., 100.*2./3.], interpolation=terc_interp)
            
            
        if round(terc_low)==round(terc_high):
            # This happens when the climatology is either always
            # a or b; in those cases the event probabilities can't be defined
            # set to flag value
            prob_early, prob_norm, prob_late = np.nan, np.nan, np.nan
        else:
            # probability for earlier than normal (includes lower tercile) 
            prob_early = rv.cdf(terc_low) 
            #probabliilty for later than normal (excludes upper tercile)    
            prob_late = 1.0 - rv.cdf(terc_high) 
            # probability for normal (excludes lower tercile, includes upper tercile)    
            prob_norm = 1.0 - (prob_late + prob_early) 

        if terc_interp==None:
            result_tup = namedtuple('result', ('probs', 'terciles',
                                               'params'))
            return result_tup(np.array([prob_early, prob_norm, prob_late]), np.array([terc_low, terc_high]), np.array([mu_clim,sigma_clim]))
        else:
            result_tup = namedtuple('result', ('probs', 'terciles'))
            
            return result_tup(np.array([prob_early, prob_norm, prob_late]), np.array([terc_low, terc_high]))

    def fcst_prenon(self, mu, sigma):
        '''
        Computes the probabilities for the pre-occurrence and non-occurrence of the IFD/FUD
        event.
        
        Args:
            mu (float):
                The mu parameter for the DCNORM distribution
            
            sigma (float):
                The sigma parameter for the DCNORM distribution
                
        Returns:
            pre_occur (float):
                Probability for the pre-occurrence of the IFD/FUD.

            non_occur (float):
                Probability for the pre-occurrence of the IFD/FUD.                
                
                
        '''
        dcnorm = dcnorm_gen(a=self.a,b=self.b) # create a generic DCNORM distribution object
        rv = dcnorm(mu, sigma)   # freeze a DCNORM distribution object with parameters mu and sigma  
        pre_occur = rv.pdf(self.a)
        non_occur = rv.pdf(self.b)
        
        return pre_occur, non_occur

        
    def fcst_deterministic(self, mu, sigma, y_clim):
        '''
        Computes the calibrated forecast ensemble mean and ensemble mean anomaly
        relative to climatology. Means for the forecast are calculated as the
        expected value of the calibrated DCNORM distribution. 
        
        Args:
            mu (float):
                The mu parameter for the DCNORM distribution
            
            sigma (float):
                The sigma parameter for the DCNORM distribution
                
            y_clim (array):
                Array of climatological dates.
                
        Returns:
            fcst_mean (float):
                The expected value of the forecast DCNORM distribution round to
                the nearest day.
                
            fcst_mean_anom (float):
                The anomaly of ``mean`` relative to the mean of ``y_clim``.
        '''
        dcnorm = dcnorm_gen(a=self.a,b=self.b) # create a generic DCNORM distribution object
        rv = dcnorm(mu, sigma)   # freeze a DCNORM distribution object with parameters mu and sigma
        fcst_mean = np.around(rv.mean())      
        fcst_mean_anom = fcst_mean - y_clim.mean()
                             
        return fcst_mean, fcst_mean_anom