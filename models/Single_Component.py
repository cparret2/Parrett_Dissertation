import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import math


def sinusoidal_with_trend(t, a, b, c, d, w, k=0):
    """
    # ----------------------------
    # Sinusoidal + trend function
    # Based on algorithm from Time Series Analysis with Python Cookbook (Atwan, 2022):
    #   t = time step / year_for_prediction
    #   a = authorized_positions[t]
    #   b = linear growth rate 
    #   c = sinusoidal amplitude
    #   d = phase shift
    #   w = angular frequency
    #   k = carrying capacity
    #   
    # ----------------------------
    """
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

def run_model(params, sim_time, dt=1, rng=None, rand=False, temperature=-1.5, sin_growth=True, growthfile=""):
    """
    Execute a single-component workforce dynamics simulation (System Dynamics Bathtub model).

    This function implements a simplified model of a government workforce lifecycle,
    capturing authorized positions, employees, applicant pool, and flows between them:
      - Accessions and recruitment (inflows into workforce)
      - Separations, retirements, exiting (outflows from workforce/applicant pool)
      - Growth of authorized headcount (policy-driven hiring capacity)

    Stocks:
      authorized_positions[t] : number of budgeted positions at time t
      employees[t]            : number of filled positions at time t
      applicant_pool[t]       : number of candidates waiting to be hired at time t

    Flows (per year):
      accessions   : hires into filled positions
      recruitment  : additional hires from applicant pool
      separations  : voluntary/involuntary exits (resignations, terminations)
      retirements  : exits due to age/service-defined retirement
      exiting      : drop out of applicant pool without hiring

    Auxiliary variables:
      vacancies        : authorized_positions - employees (open positions)
      vacancy_rate     : fraction of positions open
      internal_demand  : logistic function of total demand vs. capacity
      salary comparisons and pressures based on industry vs gov wages

    Key parameters (params dict):
      gov_init_auth          : initial authorized positions
      gov_init_population    : initial filled positions
      init_applicant_pool    : initial applicant pool size
      industry_init_avg_salary: starting industry average salary
      gov_init_avg_salary    : starting government average salary
      init_industry_cola     : baseline industry COLA rate
      init_gov_cola          : baseline government COLA rate
      sd_noise               : noise level for stochastic components
      industry_demand        : scaling factor for market demand
      gov_hiring_rate        : base fraction of vacancies filled by accessions
      recruitment_rate       : rate at which leftover vacancies generate direct hires
      auth_fill_rate         : max fraction of vacancies filled via recruitment per step
      gov_retirement_rate    : base per-year retirement fraction
      gov_separation_rate    : base per-year separation fraction
      gov_growth_rate        : policy-driven growth/decline rate for authorized headcount
      carry_cap              : multiplier for carrying capacity of headcount
      money_driven, power_seeking, mission_focused: fractions of behavioral traits
      lin_growth_rate, sin_amp, phase_shft, ang_freq: sinusoidal growth parameters
      year_offset            : optional shift aligning simulation time to predictive model

    Arguments:
      params      : dict of model parameters as above
      sim_time    : total simulation duration (years)
      dt          : integration time-step (years), default 1
      rng         : np.random.Generator for reproducible stochastic draws
      rand        : bool, if True apply randomness to growth rate
      temperature : float, negative constant controlling sensitivity of exp weights
      sin_growth  : bool, use sinusoidal-with-trend growth model if True
      growthfile  : HDF5 file path to supply external "growing" series

    Returns:
      pd.DataFrame time series of all stocks, flows, and auxiliaries
    """
    # Use provided RNG or create a default one
    if rng is None:
        rng = np.random.default_rng()
        
    # ---------------------------
    # Initialize core stocks over time
    # ---------------------------
    # Budgeted headcount (authorized positions) per year
    authorized_positions = list(np.zeros(sim_time+1))
    authorized_positions[0] = params['gov_init_auth']
    # Filled headcount (employees) per year
    employees = list(np.zeros(sim_time+1))
    employees[0] = params['gov_init_population']
    # Applicant pool waiting to join workforce
    applicant_pool = list(np.zeros(sim_time+1))
    applicant_pool[0] = params['init_applicant_pool']
    
    # Prepare lists to record time series outputs
    time_list = []               # simulation time steps
    authorized_positions_list = []
    employees_list = []
    applicant_pool_list = []     
    vacancies_list = []          # authorized - employees
    separations_list = []        # voluntary/involuntary exits
    growing_list = []            # authorized positions growth series
    g_flag = False               # external growthfile flag
    # If provided, load external growth series from HDF file
    if growthfile != "":
        g_flag = True
        growing_list = list(pd.read_hdf(growthfile)["Total"].values)

    # Initialize lists for other flows and auxiliaries
    growth_factor_list = []      # raw growth rate per step
    internal_demand_list = []    # logistic demand signal [0,1]
    retirements_list = []        # retirement flow
    accessions_list = []         # hires into open positions
    recruitment_list = []        # direct hires from applicant pool
    exiting_list = []            # applicant drop-outs
    salary_pressure_list = []    # market pressure index
    vacancy_rate_list = []       # fraction of open positions
    industry_avg_salary_list = []
    gov_avg_salary_list = []
    gov_cola_list = []
    industry_cola_list = []   
    
    # ---------------------------    
    # Pre-calculate carrying capacity and salary caps
    # ---------------------------
    max_industry_salary = params['industry_init_avg_salary'] * 2
    max_gov_salary = params['gov_init_avg_salary'] * 2
    # Government carrying capacity of positions
    gov_carrying_capacity = params['gov_init_auth'] * params['carry_cap']
    # Minimum allowable headcount to prevent collapse
    min_auth = 0.25 * params['gov_init_auth']
    # Number of simulation steps (including t=0)
    steps = np.int64(sim_time/dt) + 1

    # ---------------------------
    # Main simulation loop over each time step
    # ---------------------------
    for t in range(steps):
        # --- Stochastic cost-of-living adjustments (COLA) for industry and government ---
        industry_cola = np.clip(
            np.random.normal(
                params['init_industry_cola'],
                params['init_industry_cola'] * params['sd_noise']
            ), 0, 0.1
        )
        gov_cola = np.clip(
            np.random.normal(
                params['init_gov_cola'],
                params['init_gov_cola'] * params['sd_noise']
            ), 0, 0.1
        )
        
        # --- Compute vacancies and vacancy rate ---
        curr_vacancies = authorized_positions[t] - employees[t]
        vacancy_rate = (
            curr_vacancies / authorized_positions[t]
            if authorized_positions[t] != 0 else 0
        )
        
        # --- Internal demand signal (logistic) based on total candidates vs capacity ---
        if authorized_positions[t] != 0:
            internal_demand = 1 / (1 + np.exp(
                -((employees[t] + applicant_pool[t]) - authorized_positions[t])
                / (0.5 * authorized_positions[t])
            ))
        else:
            internal_demand = 1.0
        
        # --- Compute evolving average salaries with log growth, capped to avoid overflow ---
        industry_avg_salary = min(
            max_industry_salary,
            params['industry_init_avg_salary'] * (1 + industry_cola * math.log1p(t))
        )
        gov_avg_salary = min(
            max_gov_salary,
            params['gov_init_avg_salary'] * (1 + gov_cola * math.log1p(t))
        )
        
        # --- Salary Comparison (0.0–1.0) mapping industry vs gov wages to centered metric ---
        max_abs_salary = max(abs(industry_avg_salary), abs(gov_avg_salary))
        if max_abs_salary != 0:
            salary_comparison = (
                0.5 + 0.5 *
                ((industry_avg_salary - gov_avg_salary) / max_abs_salary)
            )
        else:
            salary_comparison = 0.5
        
        # --- Salary Pressure: market tightness * competition factor ---
        salary_pressure = (
            ((1 - np.exp(-2 * vacancy_rate)) / params['industry_demand']) *
            salary_comparison
        )

        #############################################################################################
        # --- Inflow calculations: Accessions & Recruitment ---
        #############################################################################################
        if employees[t] <= authorized_positions[t]:
            # Base hires proportional to open slots
            if t > 0:
                base_hiring = params['gov_hiring_rate'] * curr_vacancies
                # Exponential discount for salary competition and money-driven trait
                salary_competition_factor = np.exp(temperature * salary_pressure)
                money_driven_factor = np.exp(temperature * params['money_driven'])
                accessions = base_hiring * salary_competition_factor * money_driven_factor
            else:
                # At t=0, no competition factor
                accessions = params['gov_hiring_rate'] * curr_vacancies
            # Clamp to valid range
            accessions = min(accessions, curr_vacancies)
            accessions = max(0, accessions)
            # Vacancies reduced by hires
            curr_vacancies -= accessions

            # Recruitment hires from applicant pool based on leftover slots and demand
            recruitment = (
                curr_vacancies * params['recruitment_rate'] *
                (1 + np.log1p(internal_demand))
            )
            # Recruitment capped to a fraction of vacancies
            max_recruitment = params['auth_fill_rate'] * curr_vacancies
            recruitment = min(recruitment, max_recruitment)
            recruitment = max(0, recruitment)
            recruitment = min(curr_vacancies, recruitment)
            curr_vacancies -= recruitment
        else:
            # No inflows if headcount exceeds budget
            accessions = 0
            recruitment = 0

        # --- Outflow calculations: Retirements, Exiting, Separations ---
        # Retirements scale with headcount, base rate, salary_comparison, and mission focus
        retirements = max(0, employees[t] * params['gov_retirement_rate'] *
                          ((1 + ((salary_comparison * params['money_driven']) /
                            (1 + salary_comparison * params['money_driven']))) /
                           (1 + params['mission_focused'])))
        
        # Exiting from applicant pool: drop-out rate capped at 1.0
        base_rate = params['industry_demand'] * (1 + salary_pressure)
        effective_rate = min(1.0, base_rate * (1 + params['money_driven']))
        exiting = applicant_pool[t] * effective_rate
    
        # Separations: weighted combination of mission, power, and money-driven factors
        separations = max(0, (
            employees[t] * params['mission_focused'] * params['gov_separation_rate'] +
            employees[t] * params['power_seeking'] * params['gov_separation_rate'] *
            np.exp(temperature * vacancy_rate) +
            employees[t] * params['money_driven'] * salary_pressure *
            params['industry_demand']
        ))
        
        # ------------------------------------------------------------------------------
        # Authorized headcount growth (growing):
        # - Option 1: external HDF series if provided
        # - Option 2: sinusoidal-with-trend predictive model
        # - Option 3: logistic growth or decay based on raw_growth parameter
        # ------------------------------------------------------------------------------
        raw_growth = params['gov_growth_rate']
        # Introduce randomness if requested
        if rand:
            raw_growth = np.random.normal(
                params['gov_growth_rate'],
                abs(params['gov_growth_rate'] * params['sd_noise'])
            )
        growth_factor = abs(raw_growth)

        growing = 0.0
        if sin_growth:
            if g_flag:
                growing = growing_list[t]
            else:
                # Compute sinusoidal-with-trend projection for new auth count
                year_for_prediction = t+1  # offset time base
                predicted_auth = sinusoidal_with_trend(
                    year_for_prediction,
                    authorized_positions[t],
                    params['lin_growth_rate'],
                    params['sin_amp'],
                    params['phase_shft'],
                    params['ang_freq'],
                    gov_carrying_capacity
                )
                growing = predicted_auth
        elif raw_growth >= 0.0:
            # Logistic growth toward carrying capacity
            growing = gov_carrying_capacity / (
                1 + ((gov_carrying_capacity -
                      authorized_positions[0]) / authorized_positions[0]) *
                np.exp(-growth_factor * t)
            )
        else:
            # Exponential decline scenario
            if t > 0:
                growing = authorized_positions[t-1] * (1 + raw_growth)
            else:
                growing = authorized_positions[0] * (1 + raw_growth)

        # ------------------------------------------------------------------------------
        # Record all metrics for this time step
        # ------------------------------------------------------------------------------
        time_list.append(t)
        authorized_positions_list.append(authorized_positions[t])
        employees_list.append(employees[t])
        applicant_pool_list.append(applicant_pool[t])
        vacancies_list.append(authorized_positions[t] - employees[t])
        separations_list.append(separations)
        growth_factor_list.append(raw_growth)
        if not g_flag:
            growing_list.append(growing)
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
        
        # ---------------------------
        # Stock updates via Euler integration
        # ---------------------------
        if t < sim_time:
            # Applicant Pool: inflow = separations, outflow = exiting
            applicant_pool[t+1] = max(0, applicant_pool[t] + separations - exiting)

            # Authorized positions update to new growing value (clamped to min_auth)
            authorized_positions[t+1] = growing
            
            # Employees headcount: net inflows minus outflows
            employees[t+1] = max(
                0,
                employees[t] + (accessions + recruitment - retirements - separations)
            )
    
    # ---------------------------
    # Package time series into DataFrame for return
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
