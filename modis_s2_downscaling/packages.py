# Note some hard code things
import os
import re
import sys
import time
import json
import glob
import random
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
import rasterio as rio
import tensorflow as tf
import rioxarray as rxr
from pathlib import Path
from typing import Dict, List
from pyproj import Transformer
import matplotlib.pyplot as plt
from tensorflow.keras import Input
from rasterio.enums import Resampling
from datetime import datetime, timedelta
from tensorflow.keras.models import Model
from rasterio.plot import plotting_extent
from tensorflow.keras.optimizers import Adam
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, ReLU, LeakyReLU, concatenate



