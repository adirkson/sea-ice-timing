#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:32:39 2020

@author: arlan
"""
import setuptools

def readme():
    with open('README.md') as f:
        README = f.read()
    return README


setuptools.setup(
    name="NCGR",
    version="1.0.1",
    description="A Python package for calibrating probabilistic sea-ice retreat" 
                 +"and advance date forecasts using non-homogeneous censored Gaussian regression",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/adirkson/sea-ice-timing",
    author="Arlan Dirkson",
    author_email="arlan.dirkson@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7"],
    packages=["NCGR"],
    include_package_data=True,
    install_requires=["netCDF4"]
)
