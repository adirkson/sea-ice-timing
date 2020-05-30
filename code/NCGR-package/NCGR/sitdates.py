#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:56:59 2019

@author: arlan
"""
import numpy as np
from datetime import date
import datetime


class sitdates:
    '''
    This module contains several functions that are useful for defining minimium and 
    maximum possible dates, and for converting between day-of-year and normal date formats
    for plotting.
    
    Args:
        event (str, optional) {'ifd','fud'}:
            Can be one of either 'ifd' or 'fud'. If provided, but ``min_dates`` and ``max_dates``
            arguments aren't provdided, then :py:data:`sitdates.min_dates` and 
            :py:data:`sitdates.max_dates` arrays will be set to the values used in
            [1]. If not provided, ``min_dates`` and ``max_dates`` arguments need to be specified
            in order to use :py:data:`sitdates.min_dates` and 
            :py:data:`sitdates.max_dates`.
            
        min_dates (array, optional), shape=(12,):
            If provided, this array defines the minimum dates
            allowed for the forecast dates for each of the 12 initialization dates
            of the year. If not provided, then :py:data:`time_functions.min_dates`
            will be set to the values used in [1]. Note that you can override any
            of these default dates using the :py:meth:`set_min_date` function.

        max_dates (array, optional), shape=(12,):
            If provided, this array defines the maximum dates
            allowed for the forecast dates for each of the 12 initialization dates
            of the year. If not provided, then :py:data:`time_functions.min_dates`
            will be set to the values used in [1]. Note that you can override any
            of these default dates using the :py:meth:`set_max_date` function.    
                           
    References
    ----------
    .. [1] Dirkson et al., (2020): to be filled in with reference following publication.

    '''
    def __init__(self, event=None, min_dates=None, max_dates=None):
        if min_dates:
            self.min_dates = min_dates.astype(float)
        else:
            if event=='ifd':
                self.min_dates = np.array([90,90,90,90,120,151,455,455,455,455,455,455]).astype(float)
            elif event=='fud':
                self.min_dates =  np.array([273,273,273,273,273,273,273,273,273,273,304,334]).astype(float)
                
        if max_dates:
            self.max_dates = max_dates
        else:
            if event=='ifd':
                self.max_dates = np.array([273,273,273,273,273,273,546,577,608,638,638,638]).astype(float)
            elif event=='fud':
                self.max_dates = np.array([365,396,424,455,455,455,455,455,455,455,455,455]).astype(float)
                
    def set_min_date(self,month,value):
        '''
        Override pre-existing minimum date value in :py:meth:`time_functions.min_dates` array.
        
        Args:
            month (int):
                Value between 1 and 12 corresponding to the calendar month of the year to override.
            
            value (int)
                The date in day-of-year format that will override the pre-existing minimum date.
                
        Returns:
            None
        '''
        self.min_dates[month-1] = value                
        return None 
    
    def set_max_date(self,month,value):
        '''
        Override pre-existing maximum date value in :py:meth:`time_functions.min_dates` array.
        
        Args:
            month (int):
                Value between 1 and 12 corresponding to the calendar month of the year to override.
            
            value (int)
                The date in day-of-year format that will override the pre-existing minimum date.
                
        Returns:
            None
        '''
        self.max_dates[month-1] = value
        return None

    def pre_occurence(self,month):
        '''
        Returns the value that the ice free date 
        has been set to, given that ice is <50% at start of forecast
               
        Args:
            month (int):
                Integer beteween 1 and 12
                
        Returns:
            Day of year integer 
        '''        
        
        return self.min_dates[month-1] 
    
    def non_occurence(self,month):
        '''
        Returns the value that the ice free date 
        has been set to, given the non-occcurence of the event, for a 
        specfic initialization month.
               
        Args:
            month (int):
                Integer beteween 1 and 12
                
        Returns:
            Day of year integer 
        '''        
       
        return self.max_dates[month-1]
      
    
    def date_to_doy(self,mmdd,next_year=False):
        '''
        Compute date in day-of-year format from mm/dd format.
        
        Args:
            month (int):
                Integer beteween 1 and 12
                
            day (int):
                Integer between 1 and 31
                    
            next_year (boolean):
                True if DOY is for next year, false if DOY is for current year.
                Default is False.
                
        Returns:
            Day of year integer 
        '''
        month = int(mmdd[0:2])
        day = int(mmdd[3:])
        if next_year==False:
            f_date = date(2013,month,day)
        else: 
            f_date = date(2014,month,day)
            
        s_date = date(2013, 1,1)
        delta = f_date - s_date
        
            
        return delta.days    
    
    def date_to_doy2(self,month,day,next_year=False):
        '''
        Compute day of year from a given month and day. It is possible
        to span to the next year, so we use 2013 and 2014 to compute day of years
        for such cases as these are non leap years.
        
        Args:
            month (int):
                Integer beteween 1 and 12
                
            day (int):
                Integer between 1 and 31
                    
            next_year (boolean):
                True if DOY is for next year, false if DOY is for current year.
                Default is False.
                
        Returns:
            Day of year integer 
        '''
        if next_year==False:
            f_date = date(2013,month,day)
        else: 
            f_date = date(2014,month,day)
            
        s_date = date(2013, 1,1)
        delta = f_date - s_date
        
            
        return delta.days     
     
    
    def doy_to_date(self,doy,next_year=False):
        '''
        Compute Month and day from given julian doy. It is possible
        to span to the next year, so we use 2013 and 2014 to compute day of years
        for such cases as these are non leap years.
        
        Args:
            month (int):
                Integer beteween 1 and 12
                
            day (int):
                Integer between 1 and 31
                    
            next_year (boolean):
                True if DOY is for next year, false if DOY is for current year.
                Default is False.
                
        Returns:
            Day of year integer 
        '''
        if next_year==False:
            date=datetime.datetime(2013, 1, 1) + datetime.timedelta(doy)
        else:
            date=datetime.datetime(2014, 1, 1) + datetime.timedelta(doy)
            
            
        return date.strftime('%m/%d') 
    
    def init_doy(self,month):
        '''
        Compute day of year of the initialization date for a given
        initialization month.
        
        Args:
            month (int):
                Integer beteween 1 and 12
                
        Returns:
            Day of year integer 
        '''
        return self.date_to_doy2(month,1)         
    
