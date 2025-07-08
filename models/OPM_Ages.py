"""
================================================================================
MODULE NAME: OPM_Ages

DESCRIPTION:
    Defines the OPM_Ages class for handling age and length-of-service (LOS)
    bands, retirement probability calculations, and demographic utilities
    for agents in the workforce model.
    
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

import CONFIG
from numpy import clip, exp
from numpy.random import uniform, normal
import pandas as pd
import math

class OPM_Ages:
    def __init__(self):
        self.agebnds = {
            "A": (18, 20), "B": (20, 24), "C": (25, 29), "D": (30, 34),
            "E": (35, 39), "F": (40, 44), "G": (45, 49), "H": (50, 54),
            "I": (55, 59), "J": (60, 64), "K": (65, 74), "Z": (18,74)
        }
        self.losbnds = {
            "A": (0, 1), "B": (1, 2), "C": (3, 4), "D": (5, 9),
            "E": (10, 14), "F": (15, 19), "G": (20, 24), "H": (25, 29),
            "I": (30, 34), "J": (35, 99)
        }
        self.calc_age_means = {
             9: 32.7,
            11: 35.6,
            12: 38.6,
            13: 41.1,
            14: 43.9,
            15: 45.8
        }
        self.calc_los_means = {
             9: 2.5,
            11: 2.3,
            12: 3.6,
            13: 5.0,
            14: 5.4,
            15: 8.0
        }
        self.mandatory_retire_age = 70
        self.minumum_retire_age = 57
        self.max_vestment = 50
        self.min_vestment = 20
        
    def ProbRetire(self, agt):
        if agt.los < self.min_vestment:
            return 0.01

        if agt.age >= self.mandatory_retire_age:
            return 1.0

        # Add LOS-based scaling
        r_fact = 0.01
        if agt.age >= self.minumum_retire_age and agt.los >= self.min_vestment and agt.los < 2 * self.min_vestment:
            k = 0.3
            max_span = self.mandatory_retire_age - self.minumum_retire_age
            x0 = self.minumum_retire_age + (max_span / 2)
            prob = 1 / (1 + exp(-k * (agt.age - x0)))
            los_factor = min(1.0, max(0.0, (agt.los - (self.min_vestment)) / (self.max_vestment - self.min_vestment)))
            gamma = 0.5
            r_fact = min(1.0, prob + gamma * los_factor)
        else:
            k = 0.3
            x0 = 62
            prob = 1 / (1 + exp(-k * (agt.age - x0)))
            los_factor = min(1.0, max(0.0, (agt.los - 20) / (50 - 20)))
            gamma = 0.3
            r_fact = min(1.0, prob + gamma * los_factor)
        return r_fact
    
    def GetAgeLvl(self,p_age):
        for key, (lower, upper) in self.agebnds():
            if lower <= p_age <= upper:
                return key
        return None  # Return None if no category matches
    
    def GetLOSLvl(self,p_los):
        for key, (lower, upper) in self.losbnds():
            if lower <= p_los <= upper:
                return key
        return None  # Return None if no category matches
        
    def GetRandAge(self,p_agelvl):
        if p_agelvl in self.agebnds.keys():
            return uniform(self.agebnds[p_agelvl][0],self.agebnds[p_agelvl][1])
        else:
            return normal( ((self.agebnds[p_agelvl][1] - self.agebnds[p_agelvl][0]) / 2.0), 5.0)
    
    def GetProbLOS(self,level,p_std,p_avg_los=0.0,n_agts=1):
        ret_val = None
        if p_avg_los <= 0:
            ret_val = clip(normal(self.calc_los_means[level],self.calc_los_means[level]*p_std,n_agts),0.0,10.0)
        else:
            ret_val = clip(normal(p_avg_los,p_avg_los*p_std,n_agts),0.0,10.0)            
        return ret_val
        
    def GetRandAgeValue(self,level,p_std):
        return normal(self.calc_age_means[level],self.calc_age_means[level]*p_std)
    
    def GetRandAgeList(self,level,p_avg_age,p_std,n_agts):
        l_avg_ages=[]
        if type(n_agts) != int:
            n_agts = len(n_agts)
                
        if p_avg_age < self.agebnds['A'][0]:
            l_avg_ages = list(normal(self.calc_age_means[level],self.calc_age_means[level]*p_std, n_agts))
        else:
            l_avg_ages = list(normal(p_avg_age,p_avg_age*p_std, n_agts))
        return l_avg_ages