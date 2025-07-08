"""
================================================================================
MODULE NAME: Integration.py

DESCRIPTION:
    Master integration script for setting up, configuring, and running
    full simulations. Handles parameter loading, configuration resets,
    file I/O, directory management, and invokes MESA or SD models.
    
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
import pandas as pd
import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
from OPM_MESA_Model import OPM_MESA_Model
import time
from gen_movie_from_GML import GenMovies

# Add the relative path to the system path
sys.path.append(os.path.abspath('../../scripts'))
#from GS_Structure import run_model
import socket
hostname = socket.gethostname()
if hostname == 'MEGATRON':
    external_path = os.path.abspath("D:\\Google Drive\\My Education\\PhD_GMU\\GOLD_FINAL_DISSERTATION\\")
elif hostname == 'MECHAGODZILLA':
    external_path = os.path.abspath("G:\\My Drive\\My Education\\PhD_GMU\\GOLD_FINAL_DISSERTATION\\")
sys.path.append(external_path)

def FormatGSReturn(gs_res):
    grades = [9, 11, 12, 13, 14, 15]
    time = gs_res['time']
    index = pd.MultiIndex.from_product([grades,time], names=['Grade','time'])  
    gs_res.pop('time')
    df = pd.DataFrame(index=index, columns=list(gs_res.keys()))
    sing_vals={}
    for col in gs_res.keys():
        if type(gs_res[col]) == dict:
            for g in gs_res[col].keys():
                df.loc[(g, slice(None)), col] = gs_res[col][g]
        else:
            df = df.drop(columns=[col])
            sing_vals[col]= gs_res[col]
    return sing_vals, df

def load_avg_config(filepath):
    '''
    # Open and load the JSON configuration file.
    '''
    with open(filepath, "r", encoding="utf-8") as file:
            config = json.load(file)
    
    # Create a DataFrame for the low-level variables for the given simulation year.
    ll_vars = pd.DataFrame(config["LOW_LEVEL_VARS"])
    ll_vars.index = ll_vars.index.astype(int)
    hi_vars = config["HIGH_LEVEL_VARS"]
    return hi_vars, ll_vars

def cfg_getMinVal(cfg,val,grd="agg"):
    if grd=="agg": return cfg[val]['min']
    elif grd == "dct": return {g: cfg[val][g]['min'] for g in cfg[val].keys()}
    else: return cfg[val][grd]['min'] 
def cfg_getMaxVal(cfg,val,grd="agg"):
    if grd=="agg": return cfg[val]['max']
    elif grd == "dct": return {g: cfg[val][g]['max'] for g in cfg[val].keys()}
    else: return cfg[val][grd]['max'] 
def cfg_getMeanVal(cfg,val,grd="agg"):
    if grd=="agg": return cfg[val]['mean']
    elif grd == "dct": return {g: cfg[val][g]['mean'] for g in cfg[val].keys()}
    else: return cfg[val][grd]['mean'] 
def cfg_getStdVal(cfg,val,grd="agg"):
    if grd=="agg": return cfg[val]['std']
    elif grd == "dct": return {g: cfg[val][g]['std'] for g in cfg[val].keys()}
    else: return cfg[val][grd]['std'] 
        
def load_config(simyear, filepath):
    '''
    # Open and load the JSON configuration file.
    '''
    with open(filepath, "r", encoding="utf-8") as file:
        config = json.load(file)
    
    # Create a DataFrame for the low-level variables for the given simulation year.
    ll_vars = pd.DataFrame(config[f'{simyear}']['LOW_LEVEL_VARS'])
    # Ensure the DataFrame index is of integer type.
    ll_vars.index = ll_vars.index.astype(int)
    # Remove the low-level variables from the configuration so that only high-level variables remain.
    config[f'{simyear}'].pop('LOW_LEVEL_VARS')
    # The remaining configuration entries are considered high-level variables.
    hi_vars = config[f'{simyear}']
    # Return a tuple containing high-level variables and low-level variables DataFrame.
    return hi_vars, ll_vars

def load_data(simyear, temp_filepath):
    '''
    # Read the CSV data from the given file path.
    '''
    df = pd.read_csv(temp_filepath)
    # Group the data by 'DATECODE' and select the group corresponding to the code 201809.
    df = df.groupby('DATECODE').get_group(simyear*100+9)
    # Return the filtered DataFrame.
    return df
def load_yearly_config(simyear, filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        config = json.load(file)
    
    ll_vars = pd.DataFrame(config[f'{simyear}']['LOW_LEVEL_VARS'])
    ll_vars.index = ll_vars.index.astype(int)
    config[f'{simyear}'].pop('LOW_LEVEL_VARS')
    hi_vars = config[f'{simyear}']
    return hi_vars, ll_vars

def load_average_config(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        config = json.load(file)
    hi_vars = pd.DataFrame(config['HIGH_LEVEL_VARS'])
    lo_vars = config['LOW_LEVEL_VARS']
    # Replace NaN values with "0" (string) then convert back to numbers
    grades = [9, 11, 12, 13, 14, 15]
    subvars = ["min", "max", "mean", "std"]
    index = pd.MultiIndex.from_product([grades, subvars], names=['Grade', 'stat'])    
    # Create the DataFrame with the multi-index and the specified columns.
    ll_vars = pd.DataFrame(index=index, columns=list(lo_vars.keys()))
    # Populate the DataFrame by iterating over each column, grade, and stat.
    for col in lo_vars.keys():
        for grade in grades:
            # The keys for grade in the dictionary are strings, so convert the grade to a string.
            grade_str = str(grade)
            for stat in subvars:
                ll_vars.loc[(grade, stat), col] = lo_vars[col][grade_str][stat]
    return hi_vars, ll_vars

def GetHiVarParams(hi_vars,ret="mean"):
    params = {}
    for col in hi_vars.columns:
        params[col]=hi_vars.loc[ret,col]
    return params 

def GetLoVarParams(ll_vars,ret="mean"):
    params = {}
    for col in ll_vars.columns:
        params[col] = ll_vars.xs(ret, level="stat")[col].to_dict()
    return params    
def ResetConfig(gov_init_population, MODE="AVERAGE",sim_year=2011,authfact=1.2,carry_cap= 6.5,line_rate_adj=0.55,
               sin_amp_adj=1.0,phase_shft_adj=0.25,ang_freq_adj=1.0):  
    ########## MAIN ###############
    #MODE = "AVERAGE"
    #MODE = "YEARLY"
    # Initial values
    #sim_year=2011
    lin_growth_rate = 68.54263422494165 * line_rate_adj
    sin_amp = 56.937426387774316 * sin_amp_adj
    phase_shft = 153.5715608292432 * phase_shft_adj
    ang_freq =1.1673703853222208* ang_freq_adj

    grades = [9,11,12,13,14,15] 
    params={}
    hi_vars, ll_vars = None, None
   
    if MODE == "AVERAGE":
        config_path = f"{external_path}\\data\\GOLD_Category_Averages.json"
        #load parameters
        hi_vars, ll_vars = load_average_config(config_path)
        params = GetHiVarParams(hi_vars) #Means 
    else: 
        config_path = f"{external_path}\\data\\GOLD_CISA_model_config_data.json"
        hi_vars, ll_vars = load_yearly_config(sim_year,config_path)
        params = hi_vars
    sim_time=72
    unemp=0.10
    
    
    #### Additional Average Government Cost of Living Adjustment over the year
    params["init_gov_cola"]= params["gov_hist_cola"]
    params["carry_cap"] = carry_cap
    ##### EMPLOYEE ENGAGEMENT VARIABLES
    # Portion of the population that is money driven and more likely to leave for higher salaries
    params["money_driven"]= 0.0
    # Portion of the population that is power seeking and more likely to stay for promotion
    params["power_seeking"]= 0.0      
    # The rest are stable... mission focused
    params["mission_focused"] = 1 - params["money_driven"] - params["power_seeking"]
    
    ####### Salary and Compensation
    # Average Industry Cost of Living Adjustment over the years
    params["init_industry_cola"]= params["gov_hist_cola"]
    params['init_industry_starting_salary'] = params['industry_init_avg_salary'] * (1-unemp)
    
    
    #Stochasiscity Factor 
    params['sd_noise']= 0.0 #no noise
    # Portion of available applicant pool that will be hired
    
    params['perf_def'] = {'mean': 3.0, 'std': 1.5}
    
    params["recruitment_rate"] = 0.25
    params.update({
        'lin_growth_rate':lin_growth_rate,
        'sin_amp':sin_amp,
        'phase_shft':phase_shft,
        'ang_freq':ang_freq,
        'carry_cap':carry_cap
    })

    if MODE == "AVERAGE":
        params["LUTS"]= GetLoVarParams(ll_vars)  #Means
        params["LUTS"]['industry_demand_lut']= {9: 0.1, 11: 0.1, 12: 0.2, 13: 0.3, 14: 0.4, 15: 0.1}
        params["LUTS"]['promotion_delays_lut']= {9: 0.9, 11: 2, 12: 2, 13: 2, 14: 5, 15: 10}
        params["LUTS"]['init_applicant_pool'] = {g: round(v*unemp) for g,v in ll_vars['gov_init_population'].items()}
    else:
        params["LUTS"]= ll_vars #Means
        params["LUTS"]['industry_demand_lut']= {9: 0.1, 11: 0.1, 12: 0.2, 13: 0.3, 14: 0.4, 15: 0.1}
        params["LUTS"]['promotion_delays_lut']= {9: 0.9, 11: 2, 12: 2, 13: 2, 14: 5, 15: 10}
        params["LUTS"]['init_applicant_pool'] = {g: round(v*unemp) for g,v in ll_vars['gov_init_population'].items()}
    params['init_applicant_pool'] =  pd.Series(params["LUTS"]['init_applicant_pool']).sum()
    params['industry_demand'] =  pd.Series(params["LUTS"]['industry_demand_lut']).mean()
    
    params['LUTS']['gov_init_population']= gov_init_population
    params['LUTS']['gov_init_auth']= {g: gov_init_population[g] * authfact for g in grades}

    return params
########################################################################################
def CreateDirectory(d_name,prefix="output"):
    directory_name = f"{prefix}/{d_name}"
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    else:
        print(f"Directory '{directory_name}' already exists.")
    return directory_name

def run_simulation(df, simyear, hi_vars, ll_vars, maxper=50, m=4, temperature=-1.5, carry_cap=6.5, 
                   social=True, grphdname="", debug=True, writeonly=False, nowrite=False):
    '''
    # Initialize the simulation model with the provided data and configuration.
    # The model is configured to run for one more than max_steps (possibly for initialization purposes).
        (self,
         data,
         simyear, 
         hi_vars,ll_vars, 
         maxper=50, 
         m=4,
         temperature=-1.5,
         carry_cap=6.5,
         social=True,
         grphdname="",
         debug=False,writeonly=False,nowrite=False):
    '''
    omg_mdl = OPM_MESA_Model(df, simyear, hi_vars, ll_vars, maxper+1,social=social,carry_cap=hi_vars['carry_cap'],
                             grphdname=grphdname,writeonly=writeonly,debug=debug,nowrite=nowrite)
        
    # Run the simulation loop for a maximum number of steps.
    for i in range(maxper):
        # Perform a simulation step.
        omg_mdl.step()
        # If no agents remain in the model, break out of the loop.
        if len(omg_mdl.agents) == 0:
            break
    
    # Return the simulation model and the collected data as DataFrames:
    # - Model-level variables data.
    # - Agent-level variables data.
    return omg_mdl, omg_mdl.datacollector.get_model_vars_dataframe(), omg_mdl.datacollector.get_agent_vars_dataframe()

def plot_results(mdl_stat, agt_stat,tc,tcnum,d_name,tc_name="",writeonly=True):
    ### Plot
    plt.figure(figsize=(10,6))
    tc_title = f"Employment Levels over Time\nTest {tcnum}: {tc_name}\n{tc}"
    mdl_stat['ttl_population'].plot(label="Population")
    mdl_stat['ttl_authorization'].plot(label="Authorization")
    cisa2210_pop = pd.read_csv("../../data/cisa2210_actuals.csv")
    plt.scatter(cisa2210_pop.index,cisa2210_pop['DTG.1'],color='black')
    plt.title(tc_title)
    #plt.ylim(0,1000)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{d_name}/pop_auth_{tcnum}.png",bbox_inches='tight')
    if not writeonly:
        plt.show()
    plt.close('all')
    
    ### Plot
    plt.figure(figsize=(10,6))
    tc_title = f"GS Levels over Time\nTest {tcnum}: {tc_name}\n{tc}"
    grds = {}
    plt_df = pd.DataFrame({x: y for x,y in mdl_stat['GS_authorization'].items()})
    for g in [9, 11, 12, 13, 14, 15]:
        grds[g] = pd.Series({st: len(agt_stat.loc[st].groupby('grade').get_group(g))
                              for st in range(1, len(agt_stat)+1)
                              if st in agt_stat.index and g in agt_stat.loc[st].groupby('grade').groups.keys()})
        grds[g].plot(label=f"GS {g}")
        #plt_df.loc[g].plot(label=f"Auth GS-{g:02}",linestyle="--")
        
    #plt.ylim(0,500)
    plt.grid(True)
    plt.title(tc_title)
    plt.legend()
    plt.savefig(f"{d_name}/grade_pops_{tcnum}.png",bbox_inches='tight')
    if not writeonly:
        plt.show()
    plt.close('all')
    
    
    ### Plot
    plt.figure(figsize=(10,6))
    tc_title = f"Flows over Time\nTest {tcnum}: {tc_name}\n{tc}"
    mdl_stat['retirees_dt'].plot(label='retirees_dt')
    mdl_stat['promotees_dt'].plot(label='promotees_dt')
    mdl_stat['fired_dt'].plot(label='fired_dt')
    mdl_stat['newhires_dt'].plot(label='newhires_dt')
    mdl_stat['exiting_dt'].plot(label='exiting_dt')
    #plt.ylim(0,150)
    plt.grid(True)
    plt.title(tc_title)
    plt.legend()
    plt.savefig(f"{d_name}/flows_{tcnum}.png",bbox_inches='tight')
    if not writeonly:
        plt.show()
    plt.close('all')
    
    '''    ### Plot
    plt.figure(figsize=(10,6))
    tc_title = f"Avg Performance\nTest {tcnum}: {tc_name}\n{tc}"
    mdl_stat['avg_performance'].plot(label='Avg Performance')
    plt.grid(True)
    plt.title(tc_title)
    #plt.ylim(0,5)
    plt.legend()
    plt.savefig(f"{d_name}/avg_performance__{tcnum}.png",bbox_inches='tight')
    if not writeonly:
        plt.show()
    plt.close('all')
    '''
    # Parameters
    F = 1.5  # Red threshold
    W = 3.5  # Lower bound of orange
    P = 4.0  # Green threshold
    
    # Plot
    plt.figure(figsize=(10, 6))
    tc_title = f"Avg Performance\nTest {tcnum}: {tc_name}\n{tc}"
    series = mdl_stat['avg_performance']
    
    # Background shading
    plt.axhspan(0, F, color='red', alpha=0.2, label='Below F')
    plt.axhspan(W, P, color='orange', alpha=0.2, label='W to P')
    plt.axhspan(P, 5.0, color='green', alpha=0.2, label='Above P')
    
    # Plot performance line in black
    series.plot(color='black', label='Avg Performance')
    
    # Labels, legend, and limits
    plt.ylim(0.0, 5.0)
    plt.grid(True)
    plt.title(tc_title)
    plt.legend()
    plt.tight_layout()
    
    # Save/show
    plt.savefig(f"{d_name}/avg_performance__{tcnum}.png", bbox_inches='tight')
    if not writeonly:
        plt.show()
    plt.close('all')

    ### Plot
    plt.figure(figsize=(10,6))
    tc_title = f"Salaries over Time\nTest {tcnum}: {tc_name}\n{tc}"
    mdl_stat['avg_sal'].plot(label='Avg Salary')
    #plt.ylim(80000,220000)
    plt.grid(True)
    plt.title(tc_title)
    plt.legend()
    plt.savefig(f"{d_name}/avg_salary__{tcnum}.png",bbox_inches='tight')
    if not writeonly:
        plt.show()
    plt.close('all')
    

import csv
#len(mdl.history)
def save_history_to_csv(history, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(history)

def convert_keys(obj):
    pass
    if isinstance(obj, dict):
        return {str(k): convert_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys(i) for i in obj]
    else:
        return float(obj)


###############################################################################
#######  MAIN 
if __name__ == '__main__': 
    sim_time=50
    simyear = 2011
    nowrite = True
    writeonly = False
    social= True
    debug = False
    cisa2210_pop = pd.read_csv("../../data/cisa2210_pop.csv")
    cisa2210_pop_gs = pd.read_csv("../../data/cisa2210_pop_gs.csv")
    gov_init_population = dict(cisa2210_pop_gs[cisa2210_pop_gs['Year']==2011][['Grade','EmployeeCount']].values)
    
    params = ResetConfig(gov_init_population,line_rate_adj=1.0  ,sin_amp_adj=0.5)

    opm_data_path = f"../../data/{simyear}_opmdf.csv"
    df = load_data(simyear, opm_data_path)
    
    ###############################################################################
    ###################### Run Test Cases Below ###################################
    params['LUTS']['industry_demand_lut']= {9: 0.1, 11: 0.2, 12: 0.3, 13: 0.3, 14: 0.6, 15: 0.4}
    params['industry_demand'] = pd.Series(params['LUTS']['industry_demand_lut']).mean()
    unemp=0.10
    params['LUTS']['init_applicant_pool'] = {g: round(v*unemp) for g,v in params['LUTS']['gov_init_population'].items()}
    params['init_applicant_pool'] = sum(params['LUTS']['init_applicant_pool'].values())
    params['industry_demand']= pd.Series(params['LUTS']['industry_demand_lut']).mean()
    params["money_driven"]= 0.0
    params["power_seeking"]= 0.0
    params['sd_noise']= 0.1
    params["mission_focused"] = 1 - params["money_driven"] - params["power_seeking"]
    params['init_industry_avg_salary'] = params['gov_init_avg_salary'] * (1-unemp)
    d_name = CreateDirectory("FOOBAR___test")
    testcases = [{9: 0, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1},
                 {9: 0, 11: 2, 12: 2, 13: 2, 14: 5, 15: 10},
                 {9: 0, 11: 3, 12: 2, 13: 2, 14: 5, 15: 10},
                 {9: 0, 11: 4, 12: 2, 13: 2, 14: 5, 15: 10},
                 {9: 0, 11: 5, 12: 2, 13: 2, 14: 5, 15: 10},
                 {9: 0, 11: 2, 12: 4, 13: 4, 14: 4, 15: 2},
                 {9: 0, 11: 2, 12: 2, 13: 5, 14: 5, 15: 2},
                 {9: 0, 11: 2, 12: 2, 13: 2, 14: 10, 15: 2}]
    
    
    testcases = [{9: 0, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1}]
    
    params['LUTS']['gov_hiring_rate'] = {9: 1.5, 11: 1.5, 12: 1.5, 13: 1.5, 14: 1.5, 15: 1.5}
    params['LUTS']['gov_separation_rate'] = {9:0,11:0,12:0,13:0,14:0,15:0}
    params['LUTS']['gov_retirement_rate'] = {9:0,11:0,12:0,13:0,14:0,15:0}
    params['auth_fill_rate'] = 1.0
    params['recruitment_rate'] = 1.0
    params['carry_cap'] = 10
    tcnum=0
    execution_times ={}
    grph_d_name = CreateDirectory("graphs",d_name)

    for tc in testcases:
        print(f"************* \nTest case #{tcnum+1}:: {tc}\n************* \n")
        params['LUTS']['promotion_delays_lut'] = tc
        start = time.time()        
    
        mdl, mdl_stat, agt_stat = run_simulation(df, simyear,  params, params['LUTS'], maxper=sim_time, m=4, temperature=-1.5, 
                                                 carry_cap=params['carry_cap'], social=social, debug=debug, writeonly=writeonly, 
                                                 grphdname=CreateDirectory(f"{tcnum}",grph_d_name), nowrite=nowrite)
        
        end = time.time()
        print(f"\t Test Case #{tc}:\t Execution time: {end - start:.4f} seconds\n===================================")
        execution_times[tcnum] = {"testcase":tc, "time": (end - start), "max_agents": mdl_stat['ttl_population'].max()}
        mdl_stat.to_hdf(f"{d_name}/ABM_model_test_case_{simyear}_{tcnum}.hdf",key=f'mdl_data_{simyear}')
        agt_stat.to_hdf(f"{d_name}/ABM_agent_test_case_{simyear}_{tcnum}.hdf",key=f'agt_data_{simyear}')
        if not nowrite:
            if social :
                GenMovies(f"{grph_d_name}/{tcnum}")
        plot_results(mdl_stat, agt_stat,tc,tcnum,d_name,"promotion_delays_social")
        
        safe_params = convert_keys(params)
        with open(f"{d_name}/params_output_{tcnum}.json", "w", encoding="utf-8") as f:
            json.dump(safe_params, f, indent=4) 
        tcnum+=1
        
    pd.DataFrame(execution_times).to_csv(f"{d_name}/ABM_agent_test_case_{simyear}.csv")