#!/usr/bin/env python3
"""
Video 5: Energy Dependence - FAST & LONG Version
=================================================
OPTIMIZED:
- 2x faster animation (frame skip: 3 → 6)
- 2.5x longer simulation (tau: 800 → 2000)
- More frames (250 → 350)
"""

import sys
sys.path.insert(0, './src')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import time

from geodesics import SchwarzschildMetric, GeodesicSimulation

# Dark theme
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1a1a1a'
plt.rcParams['axes.facecolor'] = '#1a1a1a'
plt.rcParams['savefig.facecolor'] = '#1a1a1a'

# Colors
HORIZON_COLOR = 'black'
HORIZON_RING_COLOR = 'white'
ISCO_COLOR = '#00FF00'

