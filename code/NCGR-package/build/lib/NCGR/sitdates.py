#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:56:59 2019

@author: arlan
"""
import numpy as np
import datetime

class sitdates:
    '''
    This module contains several functions that are useful for setting and getting minimium and 
    maximum possible dates, and for converting between day-of-year and date formats
    for plotting.
    
    Args:
        event (str) {'ifd','fud'}:
            Can be either 'ifd' or 'fud'.
            
        min_dates (array, optional), shape=(12,):
            If provided, this array defines the minimum date (:math:`a` value)
            allowed for the ice-free date or freeze-up date for each calendar month corresponding to
            the initialization month.
            If not provided, then :py:data:`time_functions.min_dates`
            will be set to the values used in [1]. To only change
            a subset of those dates from [1], rather than including this argument it may be faster to 
            use the :py:meth:`set_min_date` function.

        max_dates (array, optional), shape=(12,):
            If provided, this array defines the maximum date (:math:`b` value)
            allowed for the ice-free date or freeze-up date for each calendar month corresponding to
            the initialization month.
            If not provided, then :py:data:`time_functions.min_dates`
            will be set to the values used in [1]. To only change
            a subset of those dates from [1], rather than including this argument it may be faster to 
            use the :py:meth:`set_min_date` function. 
                           
    References
    ----------
    .. [1] To be filled in with reference following publication.

    '''
    def __init__(self, event, min_dates=None, max_dates=None):
        if min_dates:
            self.min_dates = min_dates
        else:
            if event=='ifd':
                self.min_dates = np.array([90,90,90,90,120,151,455,455,455,455,455,455])
            elif event=='fud':
                self.min_dates =  np.array([273,273,273,273,273,273,273,273,273,273,304,334])
                
        if max_dates:
            self.max_dates = max_dates
        else:
            if event=='ifd':
                self.max_dates = np.array([273,273,273,273,273,273,546,577,608,638,638,638])
            elif event=='fud':
                self.max_dates = np.array([365,396,424,455,455,455,455,455,455,455,455,455])
                
    def set_min(self,month,value):
        '''
        Override pre-existing minimum date value(s) in :py:data:`time_functions.min_dates` array.
        
        Args:
            month (int or array(N,dtype='int')):
                Value(s) between 1 and 12 corresponding to the calendar month(s) for which the minimum
                date is to be overriden.
            
            value (int, float, or array(N,)):
                The date(s) in day-of-year format that will override the pre-existing minimum date(s)
                for the calendar month(s) provided in the ``month`` argument.
                
        Returns:
            None
        '''
        self.min_dates[month-1] = value                
        return None 
    
    def set_max(self,month,value):
        '''
        Override pre-existing maximum date value(s) in :py:data:`time_functions.min_dates` array.
        
        Args:
            month (int or array(N,dtype='int')):
                Value(s) between 1 and 12 corresponding to the calendar month(s) for which the maximum
                date is to be overriden.
            
            value (int, float, or array(N,)):
                The date(s) in day-of-year format that will override the pre-existing maximum date(s)
                for the calendar month(s) provided in the ``month`` argument.
                
        Returns:
            None
        '''
        self.max_dates[month-1] = value
        return None

    def get_min(self,month):
        '''
        Returns the value of the minimum possible ice-free/freeze-up date for a given initialization month. 
               
        Args:
            month (int or array(N,dtype='int')):
                Value(s) between 1 and 12 corresponding to the calendar month(s) for which the minimum
                date is to be returned.
                
        Returns:
            Day of year or days of year representing the maximum possible date for ``month``. 
        '''        
        
        return self.min_dates[month-1] 
    
    def get_max(self,month):
        '''
        Returns the value of the maximum possible ice-free/freeze-up date for a given initialization month. 
               
        Args:
            month (int or array(N,dtype='int')):
                Value(s) between 1 and 12 corresponding to the calendar month(s) for which the maximum
                date is to be returned.
                
        Returns:
            Day of year or days of year representing the maximum possible date for ``month``. 
        '''        
       
        return self.max_dates[month-1]
      
    
    def date_to_doy(self,date):
        '''
        Compute date in day-of-year format for a given date.
        
        Args:
            date (object):
                A ''date object'' made with the :py:meth:`datetime.date` function
                (see: https://docs.python.org/3/library/datetime.html#date-objects).
                
        Returns:
            Day-of-year value
        '''
        delta = date - datetime.date(date.year, 1,1)
        return delta.days    

    def dates_to_doys(self,dates):
        '''
        Compute dates in day-of-year format for a given list of dates.
        
        Args:
            dates (list):
                A list of ''date objects'' made with the :py:meth:`datetime.date` function
                (see: https://docs.python.org/3/library/datetime.html#date-objects).
                
        Returns:
            List of days-of-year values 
        '''
        doys = []
        for date in dates:
            delta = date - datetime.date(date.year, 1,1)
            doys.append(delta.days)
        return doys        
     
    def doy_to_date(self, doy, format='%m/%d'):
        '''
        Return a date for a given day-of-year value. 
        
        Args:
            doy (int or float):
                Day-of-year value
                 
            format (optional, 'str'):
                An acceptable format given to :py:meth:`strftime` (default='%m%d';
                see: https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior)
                
        Returns:
            Day of year integer 
        '''
        date = datetime.datetime(2014, 1, 1) + datetime.timedelta(float(doy))
        return date.strftime(format) 