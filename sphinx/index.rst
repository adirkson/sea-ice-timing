.. sea-ice-timing documentation master file, created by
   sphinx-quickstart on Sat May 30 14:36:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NCGR
==========================================
==========================================

``NCGR`` is a Python package for calibrating probabilistic forecasts of
ice-free dates and freeze-up dates using Non-homogenous Censored Gaussian Regression (NCGR) [1]. For plug-and-play execution of NCGR, please see:

.. toctree::
   :maxdepth: 1

   Examples/NCGR_fullfield_example.ipynb


Installation
-------------
.. code-block::

   pip install NCGR

Simple Usage
---------------
.. code-block:: Python

   from NCGR import ncgr

   ncgr.ncgr_fullfield(fcst_netcdf, hc_netcdf, obs_netcdf, out_netcdf, event,
                       a, b, model_dict, obs_dict, 
                       clim_netcdf=clim_netcdf) 


Demos with Jupyter Notebook
-----------------------------
.. toctree::
   :maxdepth: 1

   Examples/NCGR_fullfield_example.ipynb
   Examples/NCGR_gridpoint_example.ipynb
   Examples/DCNORM_distribution_example.ipynb

:download:`Download notebooks and sample data </Examples.tar.gz>`
                    
Documentation
------------------
.. toctree::
   :maxdepth: 4
   
   code_ncgr

Reference
----------
[1] Dirkson, A.​, B. Denis., M.,Sigmond., & Merryfield, W.J. (2020). Development and Calibration of Seasonal Probabilistic Forecasts of Ice-free Dates and Freeze-up Dates. ​Weather and Forecasting​. doi:10.1175/WAF-D-20-0066.1.
   
Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



