"""
================================================================================
MODULE NAME: SD_Single_Component

DESCRIPTION:
    System Dynamics model of a single-component government workforce structure.
    This simplified bathtub model simulates workforce levels, authorized positions,
    recruitment, separations, retirements, and salary competitiveness over time.
    Developed as part of the dissertation research on computational models for
    analyzing policy impacts on workforce dynamics.
    
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

# Your import statements and code start here

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import math
from sinusoidal_with_trend import sinusoidal_with_trend
 
def run_model(params, sim_time, dt=1, rng=None, rand=False, temperature=-1.5, sin_growth=True, growthfile=""):
    """
    Runs a simulation of a single component (SC) workforce model based on a Vensim bathtub model.
    Safeguards and clamping have been added to avoid overflow and NaN values.
    
    This version has been refactored to properly handle negative growth_rate values.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    # ---------------------------
    # Initialize stock variables
    # ---------------------------
    authorized_positions = list(np.zeros(sim_time+1))
    authorized_positions[0] = params['gov_init_auth']
    
    employees = list(np.zeros(sim_time+1))
    employees[0] = params['gov_init_population']
    
    applicant_pool = list(np.zeros(sim_time+1))
    applicant_pool[0] = params['init_applicant_pool']
    
    # Lists to store time series for output
    time_list = []
    authorized_positions_list = []
    employees_list = []
    applicant_pool_list = []
    vacancies_list = []
    separations_list = []  #(separations)
    growing_list = []      #(growing)
    g_flag = False
    if growthfile !="":
        g_flag = True
        growing_list = list(pd.read_hdf(growthfile)["Total"].values)

    growth_factor_list = []
    internal_demand_list = []  #(internal_demand)
    retirements_list = []      #(retirements)
    accessions_list = []       #(accessions)
    recruitment_list = []      #(recruitment)
    exiting_list = []          #(exiting)
    salary_pressure_list = []  #(salary_pressure)
    vacancy_rate_list = []
    industry_avg_salary_list = []
    gov_avg_salary_list = []
    gov_cola_list = []
    industry_cola_list = []   
    
    # ---------------------------    
    # Simulation loop
    # ---------------------------
    max_industry_salary = params['industry_init_avg_salary'] * 2
    max_gov_salary = params['gov_init_avg_salary'] * 2
    gov_carrying_capacity = params['gov_init_auth'] * params["carry_cap"]  # FIXED
    min_auth = 0.25 * params['gov_init_auth']
    steps = np.int64(sim_time/dt) + 1
    for t in range(steps):
        #current_time = t  # in years
        
        # --- Compute auxiliary variables based on current stocks ---
        industry_cola = np.clip(np.random.normal(params['init_industry_cola'],params['init_industry_cola']* params['sd_noise']),0,0.1)
        gov_cola = np.clip(np.random.normal(params['init_gov_cola'],params['init_gov_cola']* params['sd_noise']),0,0.1)
        
        # Vacancies and Vacancy Rate
        curr_vacancies = authorized_positions[t] - employees[t]
        vacancy_rate = curr_vacancies / authorized_positions[t] if authorized_positions[t] != 0 else 0
        
        # Internal Demand, clamped between 0 and 1
        internal_demand = 1 / (1 + np.exp(-((employees[t] + applicant_pool[t]) - authorized_positions[t]) / (0.5 * authorized_positions[t]))) if authorized_positions[t] != 0 else 1.0
        
        # Industry and GOV Average Salary
        industry_avg_salary = min(max_industry_salary, params['industry_init_avg_salary'] * (1 + industry_cola * math.log1p(t)))
        gov_avg_salary = min(max_gov_salary, params['gov_init_avg_salary'] * (1 + gov_cola * math.log1p(t)))
        
        # Salary Comparison: the denominator is the max of the absolute salaries.
        max_abs_salary = max(abs(industry_avg_salary), abs(gov_avg_salary))
        salary_comparison = 0.5 + 0.5 * ((industry_avg_salary - gov_avg_salary) / max_abs_salary) if max_abs_salary != 0 else 0.5
        
        # Salary Pressure
        salary_pressure = ((1 - np.exp(-2 * vacancy_rate)) / params['industry_demand']) * salary_comparison

        #############################################################################################
        # Recruitment Flow and Accessions Flow
        if employees[t] <= authorized_positions[t]:
            if t > 0:
                base_hiring = params['gov_hiring_rate'] * curr_vacancies
                salary_competition_factor = np.exp(temperature * salary_pressure)
                money_driven_factor = np.exp(temperature * params['money_driven'])
                accessions = base_hiring * salary_competition_factor * money_driven_factor
            else:
                accessions = params['gov_hiring_rate'] * curr_vacancies
            
            accessions = min(accessions, curr_vacancies)
            accessions = max(0, accessions)
            
            # Update Vacancies with accessions
            curr_vacancies -= accessions
            
            recruitment = curr_vacancies * params['recruitment_rate'] * (1 + np.log1p(internal_demand))
            max_recruitment = params['auth_fill_rate'] * curr_vacancies  # Recruitment cannot exceed 25% of current vacancies
            recruitment = min(recruitment, max_recruitment)
            recruitment = max(0, recruitment)  # Ensure non-negative
            recruitment = min(curr_vacancies, recruitment)
            curr_vacancies -= recruitment
        else:
            accessions = 0
            recruitment = 0
        #############################################################################################
        
        # Retirements Flow (People/Year)
        retirements = max(0, employees[t] * params['gov_retirement_rate'] *
                          ((1 + ((salary_comparison * params['money_driven']) / (1 + salary_comparison * params['money_driven']))) / (1 + params['mission_focused'])))
        
        # Exiting Flow from Applicant Pool (People/Year)
        #exiting = applicant_pool[t] * min(1, params['money_driven'] * (params['industry_demand'] * (1 + salary_pressure)))
        base_rate = params['industry_demand'] * (1 + salary_pressure)
        effective_rate = base_rate * (1 + params['money_driven'] )
        effective_rate = min(1.0, effective_rate)
        exiting = applicant_pool[t] * effective_rate
    
        # Separations Flow (People/Year)
        separations = max(0, (employees[t] * params['mission_focused'] * params['gov_separation_rate'] ) + \
                      (employees[t] * params['power_seeking'] * params['gov_separation_rate'] * np.exp(temperature * vacancy_rate)) + \
                      (employees[t] * params['money_driven'] * salary_pressure * params['industry_demand']))
        
        # ------------------------------------------------------------------------------
        # Handle Authorized Positions Growth/Decline based on gov_growth_rate
        # ------------------------------------------------------------------------------
        # First, capture the raw growth rate, which can be negative.
        raw_growth = params['gov_growth_rate']
        if rand:
            raw_growth = np.random.normal(params['gov_growth_rate'], abs(params['gov_growth_rate'] * params['sd_noise']))
        
        # Use the absolute value for the exponential decay/growth rate
        growth_factor = abs(raw_growth)
        
        # Compute "growing" (new authorized positions) differently for growth vs. decline.
        # If initial positions are less than carrying capacity, model growth.
        # If initial positions exceed carrying capacity, model decline.
        #if authorized_positions[0] < gov_carrying_capacity:
        growing = 0.0
        if sin_growth:
            if g_flag:
                growing = growing_list[t]
            else:
                # ------------------------------------------------------------------------------
                # Integrate the Sinusoidal Predictive Model for Authorized Positions Growth
                # ------------------------------------------------------------------------------
                # Optionally, use an offset to align simulation time with the predictive model’s time base.
                #year_for_prediction = current_time + params.get('year_offset', 0)
                year_for_prediction = t+1
                # Compute the predicted authorized positions from the sinusoidal-with-trend model.
                predicted_auth = sinusoidal_with_trend(
                    year_for_prediction,
                    authorized_positions[t],
                    params['lin_growth_rate'],
                    params['sin_amp'],
                    params['phase_shft'],
                    params['ang_freq'],
                    gov_carrying_capacity
                )
                # Use the predicted value as the "growing" variable.
                growing = predicted_auth
            
        elif raw_growth >= 0.0 and not sin_growth:
            # Growth scenario (logistic growth toward gov_carrying_capacity)
            growing = gov_carrying_capacity / (1 + ((gov_carrying_capacity - authorized_positions[0]) / authorized_positions[0]) * 
                                               np.exp(-growth_factor * t))
        else:
            if t > 0:
                # Decline scenario (exponential decay toward gov_carrying_capacity)
                #growing = authorized_positions[t-1] + (authorized_positions[t-1] - gov_carrying_capacity) * np.exp(-growth_factor * t)
                growing = authorized_positions[t-1] * (1 + raw_growth)

            else:
                #growing = authorized_positions[0] + (authorized_positions[0] - gov_carrying_capacity) * np.exp(-growth_factor * t)
                growing = authorized_positions[0] * (1 + raw_growth)
                
               
        # ------------------------------------------------------------------------------
        
        # --- Store outputs for this time step ---
        time_list.append(t)
        authorized_positions_list.append(authorized_positions[t])
        employees_list.append(employees[t])
        applicant_pool_list.append(applicant_pool[t])
        vacancies_list.append(authorized_positions[t] - employees[t])
        separations_list.append(separations)
        growth_factor_list.append(raw_growth)
        if not g_flag:
            growing_list.append(growing)
            #else just keep the list
        internal_demand_list.append(internal_demand)
        retirements_list.append(retirements)
        accessions_list.append(accessions)
        recruitment_list.append(recruitment)
        exiting_list.append(exiting)
        salary_pressure_list.append(salary_pressure)
        vacancy_rate_list.append(vacancy_rate)
        industry_avg_salary_list.append(industry_avg_salary)
        gov_avg_salary_list.append(gov_avg_salary)
        gov_cola_list.append(gov_cola)
        industry_cola_list.append(industry_cola)
        
        if t < sim_time:
            # --- Update stocks using Euler integration ---
            # Applicant Pool stock: d(Applicant Pool)/dt = Separations - Exiting
            applicant_pool[t+1] = max(0, applicant_pool[t] + separations - exiting)

            #authorized_positions[t+1] = max(min_auth, growing)
            authorized_positions[t+1] = growing
            
            # Employees stock: d(Employees)/dt = Accessions + Recruitment - Retirements - Separations
            employees[t+1] = max(0, employees[t] + (accessions + recruitment - retirements - separations))
    
    # ---------------------------
    # Organize results in a DataFrame and return
    # ---------------------------
    df_out = pd.DataFrame({
        "Time (years)": time_list,
        "Authorized Positions": authorized_positions_list,
        "Employees": employees_list,
        "Applicant Pool": applicant_pool_list,
        "Vacancies": vacancies_list,
        "Separations": separations_list,
        "Growth Factor": growth_factor_list,
        "Growing": growing_list,
        "Internal Demand": internal_demand_list,
        "Retirements": retirements_list,
        "Accessions": accessions_list,
        "Recruitment": recruitment_list,
        "Exiting": exiting_list,
        "Salary Pressure": salary_pressure_list,
        "Vacancy Rate": vacancy_rate_list,
        "Industry Avg Salary": industry_avg_salary_list,
        "Gov Avg Salary": gov_avg_salary_list,
        "Gov COLA": gov_cola_list,
        "Industry COLA": industry_cola_list
    })
    return df_out
