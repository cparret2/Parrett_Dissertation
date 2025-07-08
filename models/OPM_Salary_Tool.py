"""
================================================================================
MODULE NAME: OPM_Salary_Tool

DESCRIPTION:
    Provides salary estimation, grade-step calculations, and cost of living
    adjustment (COLA) handling for the OPM workforce simulation. Loads pay
    tables and generates updated salary levels over time.
    
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
import pandas as pd
import json
import numpy as np
from OPM_Localities import OPM_Localities
from random import choice

class OPM_Salary_Tool:
    def __init__(self,simyear,load=True,rest_of_us=True):
        self.paytables=None
        self.json_file_path = ""
        self.sallvl = pd.read_csv(f"{CONFIG.opm_dir}/sallvls.csv", index_col='lvl')
        
        self.locs = OPM_Localities()
        self.availgrades = [9,11,12,13,14,15]
        self.COLA_history = pd.Series({2005: 3.5,2006: 2.1,2007: 2.2,2008: 3.5,2009: 3.9,2010: 2.0,
                             2011: 0.0,2012: 0.0,2013: 0.0,2014: 1.0,2015: 1.0,2016: 1.0,
                             2017: 2.1,2018: 1.9,2019: 1.4,2020: 2.6,2021: 1.0,2022: 2.2,
                             2023: 4.1,2024: 4.7})
        #self.COLA_history.index = pd.to_datetime(self.COLA_history.index, format='%Y')
        self.curryear = simyear
        
        if load: 
            if rest_of_us:
                self.json_file_path = f"{CONFIG.sal_dir}/salaries_RUS_Only.json"
                self.LoadRUS()
            else:
                self.json_file_path = f"{CONFIG.sal_dir}/salaries_byloc.json"
                self.Load()
            
    def GetSalLvl(self,salary):
        for key, (lower, upper) in self.sallvl():
            if lower <= salary <= upper:
                return key
        return None  # Return None if no category matches

    def _read_json(self,fn=""):
        # Read the JSON file into a Python dictionary
        if fn != "": self.json_file_path = fn
            
        with open(self.json_file_path, 'r') as file:
            json_data = json.load(file)
        return json_data
        
    def LoadRUS(self):
        json_data = self._read_json()
        self.paytables = pd.DataFrame(json_data)
        if isinstance(self.paytables.index[0], str):
            new_idx = [eval(i) for i in self.paytables.index]  # Safe if input is trusted
            new_idx = [(grade, year, "RUS") for (grade, year) in new_idx]
            self.paytables.index = pd.MultiIndex.from_tuples(new_idx, names=["Grade", "Year", "Loc"])
    
    def Load(self,years=0):

        json_data = self._read_json()
        
        # Define the primary index (1 to 15)
        steps_index = range(1, 16)
        glocs = list(json_data.keys())
        
        if years == 0:
            # Define the secondary index (years)
            years = pd.date_range(start='2011', periods=14, freq='YE').year
        else:
            years = [years]
            
        # Create a MultiIndex from the primary index and years
        multi_index = pd.MultiIndex.from_product([steps_index, years, glocs], names=['Grade', 'Year', 'Loc'])
        
        # Create a DataFrame with integers as columns (1 to 10)
        columns = [f"{x}" for x in range(1, 11)]
        
        # Create the DataFrame
        self.paytables = pd.DataFrame(0.0, index=multi_index, columns=columns)
        for grd, yr, gloc in self.paytables.index:
            if str(yr) in json_data[gloc].keys():
                self.paytables.loc[(grd,yr,gloc)] = json_data[gloc][str(yr)][str(grd)]
    
    def GetNextGrade(self,grd):
        if grd in self.availgrades:
            x = self.availgrades.index(grd)
            if x < len(self.availgrades) - 1:  # Check if there's a next value
                x += 1  # Move to the next index
            return self.availgrades[x]
        else:
            return 0
        
    def GetEstimatedStepByLOS(self,length_of_service):
        if 0 <= length_of_service < 1: return 1
        elif 1 <= length_of_service < 2: return 2
        elif 2 <= length_of_service < 3: return 3
        elif 3 <= length_of_service < 4: return 4
        elif 4 <= length_of_service < 6: return 5
        elif 6 <= length_of_service < 8: return 6
        elif 8 <= length_of_service < 10: return 7
        elif 10 <= length_of_service < 13: return 8
        elif 13 <= length_of_service < 16: return 9
        elif length_of_service >= 16: return 10
    
    def UpdatePayTable(self,year):
        if self.curryear != year:
            self.curryear = year
            self.COLA_history[self.curryear] = self.GetCOLA(self.curryear)
            if not self.curryear in self.paytables.index.get_level_values('Year'):
                self.GenNewPayTable(self.COLA_history[self.curryear])
            
    def GetCOLA(self,year):
        if year in self.COLA_history:
            return self.COLA_history[year]
        else:
            return round(np.random.normal(self.COLA_history.mean(),self.COLA_history.std()),1)
            
    def GenNewPayTable(self,cola,loc='RUS'):
        prev_year = self.curryear - 1
        for grd in range(1,16):
            index = (grd, self.curryear, loc)
            self.paytables.loc[index] = self.GetSalaryRange(grd,year=self.curryear-1) * (1.0+cola/100)
            
    def GetSalary(self, grade, step, year=-1, loc='RUS'):
        df = self.GetSalaryRange(grade,year,loc)
        return round(df.loc[f"{step}"],0)

    def GetSalaryRange(self, grade, year=-1, loc='RUS'):
        df = None
        if year < 0:
            df = self.GetCurrYearTable(loc)
        else: 
            df = self.GetTableByYear(year,loc)
        return df.loc[grade]
        
    def GetTableByYear(self, year, loc = 'RUS'):
        idx = pd.IndexSlice
        df = self.paytables.loc[idx[:, year, loc], :]
        return df.droplevel(['Year', 'Loc'])
        
    def GetCurrYearTable(self,loc='RUS'):
        return self.GetTableByYear(self.curryear,loc)
        
    def GetPromotionStepSal(self, nxt_grd, curr_sal, loc='RUS'):
        # Get the salary series for the new grade
        new_grd = pd.Series(self.paytables.loc[(nxt_grd, self.curryear, loc)])
        
        # Use binary search to find the first index where salary >= sal
        new_step = new_grd.searchsorted(curr_sal, side='left')
    
        # Ensure new_step does not exceed step 10 and increments properly
        new_step = min(10, new_step + 1)
        
        # Get the new salary
        new_sal = new_grd.loc[f"{new_step}"]
    
        return new_step, new_sal

                
    def _GetLowerBound(self,df,p_sal):
        minval=None
        for grade in range(1,16):
            res = df.loc[grade]
            low_b = res[res <= p_sal]
            if not low_b.empty:
                minval= (grade, low_b[low_b==low_b.max()].index.values[0],low_b.max())
                return minval
            
    def _GetUpperBound(self,df,p_sal):
        maxval=None
        minstep=10
        for grade in range(15, 0, -1):
            res = df.loc[grade]
            up_b = res.min()
            if up_b <= p_sal:
                up_b = res[res >= p_sal]
                if len(up_b)!=0:
                    maxval = (grade, up_b[up_b==up_b.min()].index.values[0],up_b.min())
                return maxval
    
    def GetGradeStep(self,grade,sal):
        df = self.GetCurrYearTable()
        ret_val= df.loc[grade].searchsorted(sal, side='right')
        return max(1,ret_val)
        
    def GetEstGradeStepBySalary(self, year, value, loc='RUS'):
        min_bound = None
        max_bound = None
        
        glocs = [loc]
        if loc != 'RUS':
        # Just being given a state FIPS code - trusting domain
            glocs = self.locs.GetLocalities(STATEFP=str(loc))
        
        # Iterate through the DataFrame
        for loc in glocs:
            tab = self.GetLocalityTable(year,loc)
            for index in tab.index:
                for step in tab.columns:  # Assuming columns 1 to 10 are steps
                    cell_value = tab.loc[index, step]
                    # Exact match
                    if cell_value == value:
                        return {"exact": [index, step, loc]}  # Exact match found
                    
                    # Update the minimum bound
                    if cell_value < value and (min_bound is None or cell_value > min_bound[1]):
                        min_bound = (index, step, cell_value, loc)
                    
                    # Update the maximum bound
                    if cell_value > value and (max_bound is None or cell_value < max_bound[1]):
                        max_bound = (index, step, cell_value, loc)
            
            # Return the minimum and maximum bounds
        return {
            "higrade": [min_bound[0], min_bound[1], min_bound[3]] if min_bound else None,
            "lograde": [max_bound[0], max_bound[1], max_bound[3]] if max_bound else None
        }
    
    def GetEstSalaryByGradeLoS(self, year, grade, LoS, loc='RUS'):
        
        est_gstp = self.GetEstimatedStep(LoS)
        glocs = [loc]
        if loc != 'RUS':
        # Just being given a state FIPS code - trusting domain
            glocs = self.locs.GetLocalities(STATEFP=str(loc))
        
        # Iterate through the DataFrame
        l_avgsal=0.0
        for loc in glocs:
            l_avgsal += self.GetSalary(year,int(grade),est_gstp,loc)
            
        return {"est_avgsal": l_avgsal/len(glocs)}