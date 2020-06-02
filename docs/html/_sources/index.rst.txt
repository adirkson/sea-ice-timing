.. sea-ice-timing documentation master file, created by
   sphinx-quickstart on Sat May 30 14:36:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to sea-ice-timing
==========================================
==========================================

This project contains code used to perform statistical postprocessing on ensemble forecasts of
sea-ice retreat and advance dates. Currently, the contents of this project are under construction and preliminary.


NCGR
=======

Installation
-------------
.. code-block::

   pip install NCGR

Simple Usage
---------------
.. code-block:: Python

   from NCGR import ncgr

   ncgr.ncgr_fullfield(hc_netcdf, obs_netcdf, fcst_netcdf, out_netcdf, event,
                    a, b, clim_netcdf=clim_netcdf) 


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

   
Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


TAQM
======

Documentation
------------------

