IntegratedAnalysisTools
========================

.. image:: https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/tests.yml/badge.svg?branch=main
   :target: https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/tests.yml?query=branch%3Amain
   :alt: SpikeData Tests

.. image:: https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/black.yml/badge.svg
   :target: https://github.com/braingeneers/IntegratedAnalysisTools/actions/workflows/black.yml
   :alt: Black Formatting

A monorepo for a suite of analysis tools supporting automated closed-loop experimentation and data analysis in neuroscience and related fields.

Overview
--------

IntegratedAnalysisTools provides a unified framework for working with neuronal spike train data. The main components are:

* **spikedata**: Core module for spike train data representation, manipulation, and analysis
* **data_loaders**: Utilities to load various file formats (HDF5, NWB, KiloSort/Phy, SpikeInterface)
* **data_exporters**: Export SpikeData to common neuroscience formats

The ``SpikeData`` class provides a unified, extensible interface for representing, manipulating, and analyzing neuronal spike train data with a focus on clarity, performance, and interoperability.

Repository Structure
--------------------

* **spikedata/** - Core module for spike train data representation, manipulation, and analysis
* **data_loaders/** - Utilities to load various file formats into ``SpikeData``
  
  - Includes exporters in ``data_loaders/data_exporters.py`` to write ``SpikeData`` back to these formats

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   spikedata
   data_loaders
   data_exporters

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

