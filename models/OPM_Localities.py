import pandas as pd
import numpy as np
from scipy.stats import norm
import CONFIG

us_states = {
	'Alabama': 'AL','Alaska': 'AK','Arizona': 'AZ','Arkansas': 'AR','California': 'CA','Colorado': 'CO','Connecticut': 'CT',
    'Delaware': 'DE','Florida': 'FL','Georgia': 'GA','Hawaii': 'HI','Idaho': 'ID','Illinois': 'IL','Indiana': 'IN','Iowa': 'IA',
    'Kansas': 'KS','Kentucky': 'KY','Louisiana': 'LA','Maine': 'ME','Maryland': 'MD','Massachusetts': 'MA','Michigan': 'MI',
    'Minnesota': 'MN','Mississippi': 'MS','Missouri': 'MO','Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH',
    'New Jersey': 'NJ','New Mexico': 'NM','New York': 'NY','North Carolina': 'NC','North Dakota': 'ND','Ohio': 'OH','Oklahoma': 'OK',
    'Oregon': 'OR','Pennsylvania': 'PA','Rhode Island': 'RI','South Carolina': 'SC','South Dakota': 'SD','Tennessee': 'TN',
    'Texas': 'TX','Utah': 'UT','Vermont': 'VT','Virginia': 'VA','Washington': 'WA','West Virginia': 'WV','Wisconsin': 'WI',
    'Wyoming': 'WY','Washington, DC': "DC",'District Of Columbia':"DC",'District of Columbia' : 'DC'
}

class OPM_Localities:
    def __init__(self):
        self.locdf = pd.read_hdf(f"{CONFIG.geo_dir}/LocationTables.hdf")
        self.states = us_states
    def GetState_by_STFP(self,st):
        if st in self.locdf['STATEFP'].values:
            return self.locdf[self.locdf['STATEFP']==st]['STATE_NAME'].iloc[0]
        else:
            return None
    def GetSTUSPSby_STFP(self,st):
        if st in self.locdf['STATEFP'].values:
            return self.locdf[self.locdf['STATEFP']==st]['STUSPS'].iloc[0]
        else:
            return None
    def GetSTFP_by_ST(self,st):
        return list(self.locdf.groupby('STUSPS').get_group(st).groupby('STATEFP').groups.keys())[0]
    def GetSTFP_by_State(self,state): 
        return list(self.locdf.groupby('STATE_NAME').get_group(state).groupby('STATEFP').groups.keys())[0]        
    def GetLocalities(self,**kwargs):
        if "STATEFP" in kwargs:
            return self.locdf.groupby('STATEFP').get_group(kwargs["STATEFP"]).groupby('LOCNAME').groups.keys()
        elif "STUSPS" in kwargs:
            return self.locdf.groupby('STUSPS').get_group(kwargs["STUSPS"]).groupby('LOCNAME').groups.keys()
        elif "STATE_NAME" in kwargs: 
            return self.locdf.groupby('STATE_NAME').get_group(kwargs["STATE_NAME"]).groupby('LOCNAME').groups.keys()
            
    def GetMeanSalary_by_Locality(self,year,loc):
        return self.opmst.GetLocalityTable(year,loc).mean(axis=1)