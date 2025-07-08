"""
================================================================================
MODULE NAME: sinusoidal_with_trend

DESCRIPTION:
    Provides the sinusoidal-with-trend function for modeling authorized
    positions growth with seasonal and trend components. Used as an input
    driver for System Dynamics or ABM models.
    
AUTHOR:
    Christopher M. Parrett, George Mason University
    Email: cparret2@gmu.edu

COPYRIGHT:
    © 2025 Christopher M. Parrett

LICENSE:
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
================================================================================
"""

# ----------------------------
# Define a sinusoidal + trend function
# ----------------------------
import numpy as np
def sinusoidal_with_trend(t, a, b, c, d, w, k=0):
    t = np.asarray(t, float)
    if k > 0:
        #K = a * k
        K=k #fixed gov_carrying_capacity
        A = (K - a) / a
        r = b / a
        trend = K / (1 + A * np.exp(-r * t))
    else:
        trend = a + b * t
    cycle = c * np.sin(w * t + d)
    return trend + cycle


