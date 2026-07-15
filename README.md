# A project aimed at creating high-resolution spatio-temporal land surface albedo maps to enhance the modeling of snowmelt dynamics and water resource forecasting.

## Title: Using Low-Earth Orbit Instruments and Neural Networks to Downscale Geostationary Surface Albedo Products

## Abstract:
In the Upper Colorado River Basin (UCRB), USA, the seasonal mountain snowpack provides critical water resources for downstream agricultural and municipal communities. A source of uncertainty in estimates of snow water supply timing and amount can arise from poor constraints on how snow albedo varies in space and time, especially over the spring snowmelt period. Existing methods to observe snow albedo from remote sensing platforms are limited due to the trade-off between high spatial and temporal resolution. To circumvent this observational trade-off gap, we conducted a case study in the East River Basin of the UCRB  to produce a downscaled remotely sensed land surface albedo (LSA) and snow albedo product with simultaneously high spatial and temporal resolution. This product was derived by downscaling geostationary albedo to create high-spatio-temporal LSA and snow albedo maps. We utilize a U-Net neural network model to downscale spatiotemporal observations of surface albedo from 2 km GOES resolution images to 500 m MODIS resolution images. We then leverage the DOE ARM Surface Atmosphere Integrated Field Laboratory (SAIL) data for ground truthing. The final downscaled product achieves both high spatial and high temporal resolution and, therefore, enables detailed insights into snow albedo evolution during periods of rapid transition.  These albedo outputs have promising applications for assimilation into process-based land surface hydrology models, with the potential to improve near-real-time estimates of snow water storage in mountain regions 

## Repository Setup and Instructions
For the purpose of reproducibility, we provide the instructions below for anyone who is interested in using/adapting our code and/or data. First, let's go through the repository structure:

`GOES-Modis-Albedo-Postprocessing-main`: This directory contains a notebook file that visualizes our main results after modeling.

`GOES-Modis-Data-Preprocessing-main`: This directory contains GOES-MODIS preprocessing files.
