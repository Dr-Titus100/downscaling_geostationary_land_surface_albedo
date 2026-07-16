# Packages: All packages used by any of the files in this directory are listed here.
import os
import re
import sys
import time
import json
import math
import random
import numpy as np
import pandas as pd
import xarray as xr
import rasterio as rio
import geopandas as gpd
import tensorflow as tf
import rioxarray as rxr
from io import StringIO
from pathlib import Path
from affine import Affine
from netCDF4 import date2num
from typing import Dict, List
from rasterio.plot import show
import matplotlib.pyplot as plt
from shapely.geometry import box
import matplotlib.dates as mdates
from tensorflow.keras import Input
from rasterio.enums import Resampling
from datetime import datetime, timedelta
from pyproj import Proj, CRS, Transformer
from tensorflow.keras.models import Model
from rasterio.plot import plotting_extent
from matplotlib.dates import DateFormatter
from tensorflow.keras.optimizers import Adam
from typing import Dict, Tuple, Union, Sequence
from rioxarray.exceptions import NoDataInBounds
from typing import Iterable, Optional, List, Tuple
from sklearn.metrics import r2_score, mean_squared_error
from rasterio.warp import calculate_default_transform, reproject, Resampling
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, ReLU, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint




