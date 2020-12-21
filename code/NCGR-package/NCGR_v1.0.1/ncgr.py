#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:55:31 2020

@author: arlan
"""

from NCGR.dcnorm import dcnorm_gen

import numpy as np
from scipy.stats import norm, pearsonr
from scipy.stats.mstats import hdquantiles
from scipy.signal import detrend
from scipy.optimize import minimize
from netCDF4 import Dataset
import netCDF4 as nc4
import os


class ncgr_fullfield():    
    r'''
    * See <url> for an example and for a description of the NetCDF files used as inputs.
    
    * Performs non-homogeneous censored gaussian regression (NCGR) [1]
      on a forecast ice-free date (IFD) or freeze-up date (FUD) field.
    
    * An output NetCDF file is created that contains several quantities relevant to calibration,
      including the probability fields for each of the early, normal, and late IFD/FUD  
      categories.
    
    Args:
        hc_netcdf (str):
            '~/filename.nc' points to the path of the NetCDF file
            containing the model hindcast IFD or FUD fields to be used for training NCGR. 
            This file should contain several years of ensemble IFD or FUD hindcast fields
            for a single start date. 
            
        obs_netcdf (str):
            '~/filename.nc' points to the path of the NetCDF file
            containing the observed IFD or FUD fields to be used for training NCGR. 
            This file should contain several years of observed IFD or FUD fields corresponding
            to those being forecast in the ``hc_netcdf`` file argument. Note the conventions for
            assigning end-dates to the forecasts must also be assigned to the observations.

        fcst_netcdf (str):
            '~/filename.nc' points to the path of the NetCDF file
            containing the forecast IFD or FUD field to be calibrated. 
            This file should contain an ensemble IFD or FUD forecast field
            for a single start date. 

        out_netcdf (str):
            '~/filename.nc' a NetCDF file will containing relevant outputs will
            be written here. 

        event (str): 
            Can be either 'ifd' for an ice-free date forecast, or 'fud' for
            a freeze-up date forecast. This should match the variable name in
            all NetCDF files for the date values provided.
            
        a (float or int):
            Minimum possible date for the event in non leap year
            day-of-year units; e.g. 1=Jan 1, 91=April 1, 365=Dec 31). A value
            larger than 365 is regarded as a date for the following year.
            
        b (float or int):
            Maximum possible date for the event in non leap year 
            day-of-year units; e.g. 1=Jan 1, 91=April 1, 365=Dec 31). A value
            larger than 365 is regarded as a date for the following year. The 
            ``b`` argument must be larger than the ``a`` argument.

        clim_netcdf (str, optional):
            '<directory>/<filename.nc>' pointing to the climatology NetCDF file.
            If this is included, forecast probabilities and anomalies are 
            computed and included when writing the ``out_netcdf`` file.
            
        terc_interp (str or None):
            Interpolation scheme used to compute the terciles for the observed climatology.
            Can be one of the following:
                * None: By default, terciles are estimated using the Harrell-Davis estimator (see :py:class:scipy.stats.mstats.hdquantiles)
                
                * 'nearest-rank': Nearest rank or rank order method (see 
                https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method)
                
                * Any of the interpolation arguments for :py:class:`numpy.percentile`.
             
        sigma_eqn (str, optional): 
            Refers to the regression equation to be used for the :math:`\sigma`
            parameter in the NCGR model. This can be one of 's1', 's2', or 's3'.
            These are defined by the regression equations below as :math:`\sigma_{I}`,
            :math:`\sigma_{II}`, and :math:`\sigma_{III}`, respectively. By default,
            ``sigma`` is set to ``'s3'``. 
             
        es_tol (float or None, optional):
            Early stopping threshold used for minimizing the CRPS. 
            By default ``es_tol`` is set to ``0.001``. Specificlaly, this argument
            sets the ``tol`` argument in :py:class:`scipy.optimize.minimize(method=’SLSQP’)`. 

        pred_pval (float, optional):
            The p-value for determining statistical significance of the second 
            predictor for the  :math:`\sigma_{II}` or :math:`\sigma_{III}`
            regression equations. By default, ``pred_pval`` is set to ``0.05``.
            
        disp (True or False, optional):
            Set to True to display the numerical optimization message. By default,
            ``disp`` is set to ``False``. 

    

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
        
        
        
    %(after_notes)s
    
    References
    ----------
    .. [1] Dirkson et al., (2020): to be filled in with reference following publication.

    '''

    def __init__(self, hc_netcdf, obs_netcdf, fcst_netcdf, out_netcdf, event, a, b, 
                 clim_netcdf=None, terc_interp=None, sigma_eqn='s3', es_tol=1e-3, 
                 pred_pval=0.05, disp=False):
        
        # make objects that can be used throughout the class
        self.event = event 
        
        # path and forecast file obects needed for writing output
        self.out_netcdf = out_netcdf
        self.fcst_netcdf = fcst_netcdf 
        

        # get hindcasts, observations, and forecast IFDs or FUDs from netcdf files        
        file_hc = Dataset(hc_netcdf)
        file_obs = Dataset(obs_netcdf)
        file_fcst = Dataset(fcst_netcdf)
        
        self.X = file_hc.variables[self.event][:]
        self.Y = file_obs.variables[self.event][:]
        self.X_t = file_fcst.variables[self.event][:][0]
        
        # Get missing_value or _FillValue from X_t
        self.fill_value = file_fcst.variables[self.event][:].fill_value
        
        # get spatial dimensions from netcdf (use forecast for this)
        self.nrow = self.X_t.shape[1]
        self.ncol = self.X_t.shape[2]
        
        # get time variables relevant to the forecast
        fcst_time_var = file_fcst.variables['time']
        fcst_time_var = nc4.num2date(fcst_time_var[:], units=fcst_time_var.units, calendar=fcst_time_var.calendar)
        self.t = fcst_time_var[0].year
        
        # get time variables relevant to training 
        hc_time_var = file_hc.variables['time']
        hc_time_var = nc4.num2date(hc_time_var[:], units=hc_time_var.units, calendar=hc_time_var.calendar)
        self.tau_t = np.array([])
        for curr_time in hc_time_var:            
            self.tau_t = np.append(self.tau_t, curr_time.year)
        
        # all years
        self.t_all = np.sort(np.append(np.array(self.t),self.tau_t)) # array of all years included in both `tau_t` and `t`        

        self.a = a # minimum date possible
        self.b = b # maximum date possible
       
        self.clim_netcdf = clim_netcdf
        if self.clim_netcdf:
            file_clim = Dataset(self.clim_netcdf)
            self.Y_clim = file_clim.variables[self.event] #dimensions (time, grid rows, grid columns)
        
        # optional agrumetns
        self.terc_interp = terc_interp
        self.sigma_eqn = sigma_eqn 
        self.es_tol = es_tol
        self.pred_pval = pred_pval
        self.disp = disp
        
        # instantiate crps_funcs class to be used in calibrate_fullfield
        self.crps_funcs = crps_funcs(self.a,self.b)
        
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
                        
        '''
        N_t_all = len(self.t_all) # number of years in t_all
        tau_ind = np.where(self.t_all!=self.t)[0] # indices of t_all array corresponding to tau_t
        t_ind = np.where(self.t_all==self.t)[0][0] # index of t_all array corresponding to t
                
        mu_cal = np.zeros((self.nrow,self.ncol))
        sigma_cal = np.zeros((self.nrow,self.ncol))
        fcst_probs = np.zeros((3,self.nrow,self.ncol))
        clim_terc = np.zeros((2,self.nrow,self.ncol))
        ens_mean = np.zeros((self.nrow,self.ncol))
        ens_mean_anom = np.zeros((self.nrow,self.ncol))

        for row in np.arange(self.nrow):
            for col in np.arange(self.ncol):    
                X_all_curr = self.X[:,:,row,col]
                X_all_curr = np.insert(X_all_curr, t_ind, self.X_t[:,row,col], axis=0)
                Y_curr = self.Y[:,row,col]
                        
                if np.ma.is_masked(X_all_curr) or np.ma.is_masked(Y_curr): 
                    # if any dates are masked values for current grid cell, set distribution parameters to fill_value
                    mu_cal[row,col], sigma_cal[row,col] = self.fill_value, self.fill_value
                    
                elif np.all(Y_curr==self.a):
                    # if all training observations are "a", set distribution parameters accordingly
                    mu_cal[row,col], sigma_cal[row,col] = self.a, 1e-12 
                    
                elif np.all(Y_curr==self.b):
                    # if all training observations are "b", set distribution parameters accordingly
                    mu_cal[row,col], sigma_cal[row,col] = self.b, 1e-12
                    
                else:
                    # Calibrate with NCGR if all of the above if statements are False
                    
                    pval_y = pearsonr(self.tau_t,Y_curr)[1] # p-value for observed trend
                    if pval_y<0.05:
                        # if significant set mu predictor to the observed trend
                        coeffs = np.polyfit(self.tau_t, Y_curr, deg=1)
                        fp = np.poly1d(coeffs)
                        mu_clim = fp(self.t_all)
                        mu_clim[mu_clim>self.b] = self.b
                        mu_clim[mu_clim<self.a] = self.a
                        
                    else:
                        # if not significant, set mu predictor to observed climatology
                        mu_clim = Y_curr.mean()*np.ones(N_t_all)
        
                    # set sigma predictor to the standard deviation of the detrended observations
                    std_clim = np.ones(N_t_all)*detrend(Y_curr).std(ddof=1) 
         
                    if mu_clim[t_ind]==self.a:
                        # if the observed trend regressed onto the forecast year is "a", set 
                        # distribution parameters accordingly
                        mu_cal[row,col], sigma_cal[row,col] = self.a, 1e-12 
                    
                    elif mu_clim[t_ind]==self.b:
                        # if the observed trend regressed onto the forecast year is "b", set 
                        # distribution parameters accordingly
                        
                        mu_cal[row,col], sigma_cal[row,col] = self.b, 1e-12 
                        
                    else:
                        # Set up other predictors for regression equations and find the parameters
                        # that minimize the CRPS
                        
                        ############# Predictors for mu ###############################                               
                        X_d = detrend(X_all_curr.mean(axis=1))  
                        
                        # ensure that the first guess for mu will never be outside the interval [a,b]
                        X_d[mu_clim+X_d<self.a] = self.a - mu_clim[mu_clim+X_d<self.a] 
                        X_d[mu_clim+X_d>self.b] = self.b - mu_clim[mu_clim+X_d>self.b]
                    
                        # holds the two predictors for all years
                        predictors_all_mu = np.array([mu_clim,
                                                          X_d])
                
                        # holds the two predictors for all training years
                        predictors_tau_mu = predictors_all_mu[:,tau_ind]
                        
                        # holds the two predictors for the forecast year
                        predictors_t_mu = predictors_all_mu[:,t_ind]  
                
                
                        ############# Predictors for sigma ##############################           
                        predictors_all_std = np.array([std_clim])
                        
                        # trend-corrected hindcasts (note: values on [a,b] by construction of X_d)
                        X_tc = np.around(mu_clim + X_d,4) 
                        # unbiased ensemble-mean error
                        error = np.abs(X_tc[tau_ind] - Y_curr)   
                        
                        if self.sigma_eqn=='s1': # no second predictor for sigma
                            None 
                            
                        elif self.sigma_eqn=='s2': # second predictor is the ensemble standard deviation 
                            pred = np.std(X_all_curr,ddof=1,axis=1)
                            p_val_x = pearsonr(pred[tau_ind], error)[1]
                            
                            if p_val_x<self.pred_pval:
                                predictors_all_std = np.concatenate((predictors_all_std, np.array([pred])),axis=0)                                
                            else:
                                None
                        
                        elif self.sigma_eqn=='s3':
                            pred = X_tc # second predictor is the trend-corrected ensemble mean (default)
                            p_val_x = pearsonr(pred[tau_ind], error)[1]
                            if p_val_x<self.pred_pval:
                                predictors_all_std = np.concatenate((predictors_all_std, np.array([pred])),axis=0)            
                            else:
                                None
                       
                        predictors_tau_std = predictors_all_std[:,tau_ind]
                        predictors_t_std = predictors_all_std[:,t_ind]                                             
        
                        # predictors for both mu and sigma in one big array
                        predictors_tau = np.array([predictors_tau_mu,predictors_tau_std])       
        
                        # get number of predictors for each parameter
                        N_pred_mu = predictors_tau[0].shape[0]
                        N_pred_s = predictors_tau[1].shape[0] 
        
                        # set initial parameter guesses 
                        params0 = np.concatenate((np.ones(N_pred_mu),np.ones(N_pred_s)))
                        
                        if N_pred_s==1: 
                            None
                        elif N_pred_s==2: # if second predictor for sigma is signficant, set guess for coefficient 
                            params0[-1] = std_clim[0]/np.mean(predictors_all_std[1])
        
                        ############################################################  
                                              
                        # perform minimization of the CRPS
                        res_beinf = minimize(self.crps_funcs.crps_ncgr, params0, args=(predictors_tau,Y_curr),
                                             jac=self.crps_funcs.crps_ncgr_jac,
                                             tol=self.es_tol,
                                             options={'disp':self.disp}, 
                                             constraints=self.build_cons(predictors_tau,Y_curr))
                        
        
                           
                        if np.isnan(res_beinf.fun) or res_beinf.fun==np.inf or res_beinf.fun<0.0:
                            print("Minimization couldn't converge - this shouldn't ever happen; if it does,"+ 
                                  "please contact Arlan Dirkson at arlan.dirkson@gmail.com")
                        else:
                            # retrieve optimal coefficients from minimization result object
                            params_es = res_beinf.x
                
                            # seperate these out into coefficients for mu and sigma
                            params_mu, params_std = params_es[0:N_pred_mu], params_es[N_pred_mu:N_pred_mu+N_pred_s]
                            
                            # apply them to the regression equations for real-time (i.e. the forecast) predictors
                            mu_cal[row,col] = min(max(self.a,np.dot(params_mu.T,predictors_t_mu)),self.b) # constrain to [a,b]
                            sigma_cal[row,col] = max(1e-12,np.dot(params_std.T,predictors_t_std)) # constrain to (0,inf]
                    
                if self.clim_netcdf: # If a NetCDF file containing dates to compute climatologies is provided
                    if mu_cal[row,col]==self.fill_value:
                        # set to fill_value if on a masked grid cell
                        fcst_probs[:,row,col], clim_terc[:,row,col] = self.fill_value, self.fill_value
                        ens_mean[row,col], ens_mean_anom[row,col]  = self.fill_value, self.fill_value
                     
                    else:
                        # Call on function to compute forecast probabilities for 
                        # early, near-normal, and late ice retreat or advance
                        fcst_probs[:,row,col], clim_terc[:,row,col] = fcst_vs_clim(self.a,self.b,self.fill_value).fcst_event_probs(mu_cal[row,col],sigma_cal[row,col],self.Y_clim[:,row,col], self.terc_interp)
                        # Call on function to compute the ensemble mean date
                        # and the ensemble mean anomaly (deterministic forecasts)
                        ens_mean[row,col], ens_mean_anom[row,col]  = fcst_vs_clim(self.a,self.b,self.fill_value).fcst_deterministic(mu_cal[row,col],sigma_cal[row,col],self.Y_clim[:,row,col])               
                    
                else:
                    # if no climatology netcdf was provided, just return the calibrated distribution parameters
                    None
         
        if self.clim_netcdf:
            result = np.array([mu_cal, sigma_cal, fcst_probs, clim_terc, ens_mean, ens_mean_anom])
        else:
            result = np.array([mu_cal, sigma_cal])

        return result
                    

            
    def build_cons(self,predictors,y):
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
            
            return mu_hat - self.a

        def con_mu2(params, predictors,y):    
            params_mu = params[:N_pred_m]
                
            predictors_mu = predictors[0]
                                   
            mu_hat = np.dot(params_mu.T,predictors_mu)
            
            return self.b - mu_hat
        
        def con_std(params, predictors,y):
            params_s = params[N_pred_m:N_pred_m+N_pred_s]
            
            predictors_s = predictors[1]    

            s_hat = np.dot(params_s.T,predictors_s)     
            
            return s_hat - 1e-12
        
        cons = ({'type': 'ineq', 'fun': con_mu1, 'args':(predictors,y)},
                {'type': 'ineq', 'fun': con_mu2, 'args':(predictors,y)},
                {'type': 'ineq', 'fun': con_std, 'args':(predictors,y)})
        
        return cons

    
    
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
                in the NetCDF files given to :py:class:`ncgr_fullfield`.
                
            sigma_cal (ndarray), shape (`nrow`, `ncol`):
                Calibrated :math:`\sigma` parameter for the predictive 
                DCNORM distribution, where `nrow` and `ncol` are defined as
                in ``mu_cal``.
                
            fcst_probs (ndarray), shape (`3`, `nrow`, `ncol`):
                The three forecast probabilities for early, near-normal, and late
                sea-ice retreat/advance, where `nrow` and `ncol` are defined as
                in ``mu_cal``.                

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
            mu_cal, sigma_cal, fcst_probs, clim_terc, ens_mean, ens_mean_anom = result
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
            if dname=='ensemble':
                None
            else:
                dsout.createDimension(dname, the_dim.size)  
                
        dnames_keep = [] # empty list to be filled with th dimensions that were kept (i.e. all but the ensemble dimension)
        for dname in dsin.variables[self.event].dimensions:
            if dname=='ensemble':
                None
            else:
                dnames_keep.append(dname)
                
        dnames_keep = tuple(dnames_keep) # convert from list to tuple for passing into the createVariable function
                
        # copy all variables from original forecast file to new file
        # except the raw ifd or fud field; in place of the raw ifd or fud field,
        # write several new variables to the file
        for v_name, varin in dsin.variables.items(): 
            if v_name==self.event:
                # Write new variables
                outVar1 = dsout.createVariable('mu_cal', np.float32, dnames_keep, fill_value=self.fill_value)
                outVar1.long_name = 'calibrated mu parameter for DCNORM distribution'
                outVar1.valid_min = self.a
                outVar1.valid_max = self.b             
                outVar1[:] = mu_cal.astype(np.float32)              

                outVar2 = dsout.createVariable('sigma_cal', np.float, dnames_keep, fill_value=self.fill_value)
                outVar2.long_name = 'calibrated sigma parameter for DCNORM distribution'
                outVar2.valid_min = 1e-12
                outVar2.valid_max = np.inf
                outVar2[:] = sigma_cal.astype(np.float) 
            
                if self.clim_netcdf:
                    outVar3 = dsout.createVariable('prob_EN', np.float32, dnames_keep, fill_value=self.fill_value)
                    outVar3.long_name = 'calibrated probability for earlier-than-normal '+self.event
                    outVar3.valid_min = 0.0
                    outVar3.valid_max = 1.0
                    outVar3.units = "1"
                    outVar3[:] = fcst_probs[0].astype(np.float32) 
    
                    outVar4 = dsout.createVariable('prob_NN', np.float32, dnames_keep, fill_value=self.fill_value)
                    outVar4.long_name = 'calibrated probability for near-normal '+self.event
                    outVar4.valid_min = 0.0
                    outVar4.valid_max = 1.0
                    outVar4.units = "1"
                    outVar4[:] = fcst_probs[1].astype(np.float32) 
    
                    outVar5 = dsout.createVariable('prob_LN', np.float32, dnames_keep, fill_value=self.fill_value)
                    outVar5.long_name = 'calibrated probability for later-than-normal '+self.event
                    outVar5.valid_min = 0.0
                    outVar5.valid_max = 1.0
                    outVar5.units = "1"
                    outVar5[:] = fcst_probs[2].astype(np.float32) 
                
                    outVar6 = dsout.createVariable('clim_1_3',np.float32, dnames_keep, fill_value=self.fill_value)
                    outVar6.long_name = 'observed climatological 1/3 '+self.event+' quantile'
                    outVar6.valid_min = self.a
                    outVar6.valid_max = self.b
                    outVar6[:] = clim_terc[0].astype(np.float32) 
    
                    outVar7 = dsout.createVariable('clim_2_3', np.float32, dnames_keep, fill_value=self.fill_value)
                    outVar7.long_name = 'observed climatological 2/3 '+self.event+' quantile'
                    outVar7.valid_min = self.a
                    outVar7.valid_max = self.b
                    outVar7[:] = clim_terc[1].astype(np.float32) 
    
                    outVar8 = dsout.createVariable('ens_mean',np.float32, dnames_keep, fill_value=self.fill_value)
                    outVar8.long_name = 'calibrated ensemble mean '+self.event
                    outVar8.valid_min = self.a
                    outVar8.valid_max = self.b
                    outVar8[:] = ens_mean.astype(np.float32) 
    
                    outVar9 = dsout.createVariable('ens_mean_anom', np.float32, dnames_keep, fill_value=self.fill_value)
                    outVar9.long_name = 'calibrated ensemble mean '+self.event+'anomaly relative to climatology'
                    outVar9[:] = ens_mean_anom.astype(np.float32) 
                
            else:
                # keeps time and space coordinate variables
                outVar = dsout.createVariable(v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                outVar[:] = dsin.variables[v_name][:]

        dsin.close()
        dsout.close()
        
        return None             


class ncgr_gridpoint():    
    r'''
    Args:
        X (ndarray), shape (`N,n`):
            Forecasts for training NCGR, where `N` is the number of years
            for training and `n` is the ensemble size for a 
            given forecast.
        
        Y (ndarray), shape (`N,`):
            Observations for training NCGR, where `N` is the number of years
            for training.

        X_t (ndarray), shape (`n,`):
            Forecast to be calibrated, where `n` is the ensemble size.
            
        tau_t (ndarray), shape (`N,`):
            Years corresponding to those used for training period (this should not
            contain the forecast year). The year should be based on the initialization 
            date, not the date of the IFD or FUD event.
            
        t (float or int):
            The year in which the forecast is initialized.
            
        a (float or int):
            Minimum possible date for the event in non leap year
            day-of-year units; e.g. 1=Jan 1, 91=April 1, 365=Dec 31). A value
            larger than 365 is regarded as a date for the following year.
            
        b (float or int):
            Maximum possible date for the event in non leap year 
            day-of-year units; e.g. 1=Jan 1, 91=April 1, 365=Dec 31). A value
            larger than 365 is regarded as a date for the following year. The 
            ``b`` argument must be larger than the ``a`` argument.

        Y_clim (ndarray, optional), shape (`m,`):
            Dates used to compute climatologies for forecast probabilities
            and forecast anomalies. By default, ``Y_clim`` is `None`. 

        terc_interp (str or None):
            Interpolation scheme used to compute the terciles for the observed climatology.
            Can be one of the following:
                * None: By default, terciles are estimated using the Harrell-Davis estimator (see :py:class:scipy.stats.mstats.hdquantiles)
                
                * 'nearest-rank': Nearest rank or rank order method (see 
                https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method)
                
                * Any of the interpolation arguments for :py:class:`numpy.percentile`.

        sigma_eqn (str, optional): 
            Refers to the regression equation to be used for the :math:`\sigma`
            parameter in the NCGR model. This can be one of 's1', 's2', or 's3'.
            These are defined by the regression equations below as :math:`\sigma_{I}`,
            :math:`\sigma_{II}`, and :math:`\sigma_{III}`, respectively. By default,
            ``sigma`` is set to ``'s3'``. 
             
        es_tol (float or None, optional):
            Early stopping threshold used for minimizing the CRPS. 
            By default ``es_tol`` is set to ``0.001``. Specificlaly, this argument
            sets the ``tol`` argument in :py:class:`scipy.optimize.minimize(method=’SLSQP’)`. 

        pred_pval (float, optional):
            The p-value for determining statistical significance of the second 
            predictor for the  :math:`\sigma_{II}` or :math:`\sigma_{III}`
            regression equations. By default, ``pred_pval`` is set to ``0.05``.
            
        disp (True or False, optional):
            Set to True to display the numerical optimization message. By default,
            ``disp`` is set to ``False``.
            
    

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


    The relevant method contained in this class is:
         
    ``calibrate_gridpoint()``
        Performs NCGR on the forecast IFD or FUD at a single gridpoint.
        
        
    %(after_notes)s
    
    References
    ----------
    .. [1] Dirkson., et al (2020): to be filled in with reference following acceptance.

    '''

    def __init__(self, X, Y, X_t, tau_t, t, a, b, 
                 Y_clim=None, terc_interp=None, sigma_eqn='s3', es_tol=1e-3, 
                 pred_pval=0.05, disp=False):
        
        # make objects that can be used throughout the class
        self.X = X
        self.Y = Y
        self.X_t = X_t
        
        # get time variables relevant to the forecast
        self.t = t
        
        # get time variables relevant to training (hindcasts and observations)
        self.tau_t = tau_t
        
        # all years
        self.t_all = np.sort(np.append(np.array(self.t),self.tau_t)) # array of all years included in both `tau_t` and `t`        

        self.a = a # minimum date possible
        self.b = b # maximum date possible


        self.Y_clim = Y_clim
        
        self.terc_interp= terc_interp
      
        self.sigma_eqn = sigma_eqn 
        self.es_tol = es_tol
        self.pred_pval = pred_pval
        self.disp = disp
        
        self.crps_funcs = crps_funcs(self.a,self.b)
        
        self.fill_value = np.nan # to be consistent with the default _Fillvalue used in 
              

    def calibrate_gridpoint(self):
        '''
        Performs NCGR and returns relevant calibrated forecast quantities.
        
        Returns:
            mu_cal (float):
                Calibrated :math:`\mu` parameter for the predictive 
                DCNORM distribution.
                
            sigma_cal (float):
                Calibrated :math:`\sigma` parameter for the predictive 
                DCNORM distribution.
                
            fcst_probs (ndarray), shape (3,):
                The three forecast probabilities for early, near-normal, and late
                sea-ice retreat/advance.     

            clim_terc (ndarray), shape (2,):
                The two terciles (i.e. 1/3 and 2/3 quantiles) for the observed climatology.
                
            mean (float):
                Expected value of the forecast DCNORM distribution.
                
            mean_anom (float):
                Anomaly of ``mean`` relative to climatology.       
                
        '''
        N_t_all = len(self.t_all) # number of years in t_all
        tau_ind = np.where(self.t_all!=self.t)[0] # indices of t_all array corresponding to tau_t
        t_ind = np.where(self.t_all==self.t)[0][0] # index of t_all array corresponding to t
                  
        X_all_curr = self.X
        X_all_curr = np.insert(X_all_curr, t_ind, self.X_t, axis=0)
        Y_curr = self.Y
        
        #### See commenting in ncgr_fullfield.calibrate_fullfield for description of steps
        
        if np.ma.is_masked(X_all_curr) or np.ma.is_masked(Y_curr):
            mu_cal, sigma_cal = self.fill_value, self.fill_value
            
        elif np.all(Y_curr==self.a):
            mu_cal, sigma_cal = self.a, 1e-12 
            
        elif np.all(Y_curr==self.b):
            mu_cal, sigma_cal = self.b, 1e-12
            
        else:
            if np.all(Y_curr==Y_curr[0]): # check if array elements are constant
                pval_y = None
            else:
                pval_y = pearsonr(self.tau_t,Y_curr)[1]
                
            if pval_y<0.05:
                coeffs = np.polyfit(self.tau_t, Y_curr, deg=1)
                fp = np.poly1d(coeffs)
                mu_clim = fp(self.t_all)
                mu_clim[mu_clim>self.b] = self.b
                mu_clim[mu_clim<self.a] = self.a
            else:
                mu_clim = Y_curr.mean()*np.ones(N_t_all)

            std_clim = np.ones(N_t_all)*detrend(Y_curr).std(ddof=1) 
 
                       
            if mu_clim[t_ind]==self.a:
                mu_cal, sigma_cal = self.a, 1e-12 
            
            elif mu_clim[t_ind]==self.b:
                mu_cal, sigma_cal = self.b, 1e-12 
                
            else:
                ############# Predictors for mu ###############################
                        
                X_d = detrend(X_all_curr.mean(axis=1))     
                X_d[mu_clim+X_d<self.a] = self.a - mu_clim[mu_clim+X_d<self.a] 
                X_d[mu_clim+X_d>self.b] = self.b - mu_clim[mu_clim+X_d>self.b]
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
                
                if self.sigma_eqn=='s1': # no second predictor for sigma
                    None 
                    
                elif self.sigma_eqn=='s2': # predictor is the ensemble standard deviation 
                    pred = np.std(X_all_curr,ddof=1,axis=1)
                    
                    if np.all(pred[tau_ind]==pred[tau_ind][0]) or np.all(error==error[0]):
                        p_val_x = None
                    else:
                        p_val_x = pearsonr(pred[tau_ind], error)[1]
                    
                    if p_val_x<self.pred_pval:
                        predictors_all_std = np.concatenate((predictors_all_std, np.array([pred])),axis=0)                                
                    else:
                        None
                
                elif self.sigma_eqn=='s3':
                    pred = X_tc # predictor is the trend-corrected ensemble mean
                    if np.all(pred[tau_ind]==pred[tau_ind][0]) or np.all(error==error[0]):
                        p_val_x = None
                    else:
                        p_val_x = pearsonr(pred[tau_ind], error)[1]
                        
                    if p_val_x<self.pred_pval:
                        predictors_all_std = np.concatenate((predictors_all_std, np.array([pred])),axis=0)            
                    else:
                        None
               
                predictors_tau_std = predictors_all_std[:,tau_ind]
                predictors_t_std = predictors_all_std[:,t_ind]                                             


                predictors_tau = np.array([predictors_tau_mu,predictors_tau_std])       


                N_pred_mu = predictors_tau[0].shape[0]
                N_pred_s = predictors_tau[1].shape[0] 

                # set initial parameter guesses 
                params0 = np.concatenate((np.ones(N_pred_mu),np.ones(N_pred_s)))
                
                if N_pred_s==1:
                    None
                elif N_pred_s==2:
                    params0[-1] = std_clim[0]/predictors_all_std[1,0]

                ############################################################  
                                      
                res_beinf = minimize(self.crps_funcs.crps_ncgr, params0, args=(predictors_tau,Y_curr),
                                     jac=self.crps_funcs.crps_ncgr_jac,
                                     tol=self.es_tol,
                                     options={'disp':self.disp}, 
                                     constraints=self.build_cons(predictors_tau,Y_curr))
                

           
                if np.isnan(res_beinf.fun) or res_beinf.fun==np.inf or res_beinf.fun<0.0:
                    print("Minimization couldn't converge - this shouldn't ever happen; if it does,"+ 
                          "please contact Arlan Dirkson at arlan.dirkson@gmail.com")
                else:
                    params_es = res_beinf.x
        
                    params_mu, params_std = params_es[0:N_pred_mu], params_es[N_pred_mu:N_pred_mu+N_pred_s]
                    
                    mu_cal = min(max(self.a,np.dot(params_mu.T,predictors_t_mu)),self.b) # constrain to [a,b]
                    sigma_cal = max(1e-12,np.dot(params_std.T,predictors_t_std)) # constrain to (0,inf]
            
        if self.Y_clim is not None:
            if mu_cal==self.fill_value:
                fcst_probs, clim_terc = self.fill_value*np.ones(3), self.fill_value*np.ones(2)
                ens_mean, ens_mean_anom = self.fill_value, self.fill_value
                
            else:               
                fcst_probs, clim_terc = fcst_vs_clim(self.a,self.b,self.fill_value).fcst_event_probs(mu_cal,sigma_cal,self.Y_clim,self.terc_interp)
                ens_mean, ens_mean_anom = fcst_vs_clim(self.a,self.b,self.fill_value).fcst_deterministic(mu_cal,sigma_cal,self.Y_clim)    
            
            result = np.array([np.array([mu_cal]), np.array([sigma_cal]), 
                   fcst_probs, clim_terc, np.array([ens_mean]), np.array([ens_mean_anom])])
        else:
            result = np.array([mu_cal, sigma_cal])
        return result

    def build_cons(self,predictors,y):
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
            
            return mu_hat - self.a

        def con_mu2(params, predictors,y):    
            params_mu = params[:N_pred_m]
                
            predictors_mu = predictors[0]
                                   
            mu_hat = np.dot(params_mu.T,predictors_mu)
            
            return self.b - mu_hat
        
        def con_std(params, predictors,y):
            params_s = params[N_pred_m:N_pred_m+N_pred_s]
            
            predictors_s = predictors[1]    

            s_hat = np.dot(params_s.T,predictors_s)     
            
            return s_hat - 1e-12
        
        cons = ({'type': 'ineq', 'fun': con_mu1, 'args':(predictors,y)},
                {'type': 'ineq', 'fun': con_mu2, 'args':(predictors,y)},
                {'type': 'ineq', 'fun': con_std, 'args':(predictors,y)})
        
        return cons




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
    
    def __init__(self,a,b,fill_value):
        self.a = a
        self.b = b
        self.fill_value = fill_value
    
    def fcst_event_probs(self, mu, sigma, y_clim, terc_interp=None):
        '''
        Computes the forecast probabilities for an early, normal, or late
        event relative to some defined climatology. Default is to use the past
        10 years of observations, but one can also choose to use more or less years.
        
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
                    * None: By default, terciles are estimated using the Harrell-Davis estimator (see :py:class:scipy.stats.mstats.hdquantiles)
                    
                    * 'nearest-rank': Nearest rank or rank order method (see 
                    https://en.wikipedia.org/wiki/Percentile#The_nearest-rank_method)
                    
                    * Any of the interpolation arguments for :py:class:`numpy.percentile`.                
                
        Returns:
            result (object ndarray):
                An object array containing 2 arrays. The first array has shape (3,) and contains the forecast
                probabilities for being the event occuring early, near-normal, or late, respectively. Note that
                these are set to fill_value when the forecast distribution 
                predicts the pre-occurence of the event with 100% probability. The
                second array has shape (2,) and contains the climatological terciles deliniating 
                the event categories.
        '''
        dcnorm = dcnorm_gen(a=self.a,b=self.b) # create a generic DCNORM distribution object
        rv = dcnorm(mu, sigma)   # freeze a DCNORM distribution object with parameters mu and sigma
        
        if terc_interp is None:
            terc_low, terc_high = hdquantiles(y_clim, [1./3.,2./3.]) 
        elif terc_interp=='nearest-rank':
            terc_low, terc_high = np.sort(y_clim)[int(np.ceil(1./3.*len(y_clim)))-1], np.sort(y_clim)[int(np.ceil(2./3.*len(y_clim)))-1]
        else:
            terc_low, terc_high = np.percentile(y_clim, [100.*1./3., 100.*2./3.], interpolation=terc_interp)
            
            
        if np.all(y_clim==self.a) or np.all(y_clim==self.b) or terc_low==terc_high or rv.pdf(self.a)==1.0:
            prob_early, prob_norm, prob_late = self.fill_value, self.fill_value, self.fill_value
        else:
            prob_early = rv.cdf(terc_low) # probability for earlier than normal
            prob_norm = rv.cdf(terc_high) - rv.cdf(terc_low) # probability for normal
            prob_late = 1.0 - rv.cdf(terc_high) #probabliilty for later than normal   
            
        result = np.array([prob_early, prob_norm, prob_late]), np.array([terc_low, terc_high])  
        return result    

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
                The expected value of the forecast DCNORM distribution rounded to
                the nearest day.
                
            fcst_mean_anom (float):
                The anomaly of ``mean`` relative to the mean of ``y_clim`` rounded to the nearest
                day. Set to _fillvalue if the forecast distribution 
                predicts the pre-occurence of the event with 100% probability.
        '''
        dcnorm = dcnorm_gen(a=self.a,b=self.b) # create a generic DCNORM distribution object
        rv = dcnorm(mu, sigma)   # freeze a DCNORM distribution object with parameters mu and sigma
        fcst_mean = np.around(rv.mean())
        if rv.pdf(self.a)==1.0:
            fcst_mean_anom = self.fill_value
        else:            
            fcst_mean_anom = fcst_mean - y_clim.mean()
        
                      
        return fcst_mean, fcst_mean_anom