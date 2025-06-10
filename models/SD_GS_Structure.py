import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import math

# ----------------------------
# Define a sinusoidal + trend function
# ----------------------------
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

def run_model(p, sim_time, dt=1, rng=None, rand=False, temperature=-0.75, salary_cap=2.0, growthfile=""):
    """
    Runs a simulation of a multi–grade (GS) workforce model based on a Vensim bathtub model.
    Safeguards and clamping have been added to avoid overflow and NaN values.
    """
    # -- Simulation Setup --
    gs_levels = [9, 11, 12, 13, 14, 15]
    steps = np.int64(sim_time/dt) + 1
    time = np.arange(0, sim_time + dt, dt)
    
    if rng is None:
        rng = np.random.default_rng()
        
    # -- Lookup Tables (LUTs) --
    init_gov_paytable = p['LUTS']['gov_init_avg_salary']
    gov_init_population = p['LUTS']['gov_init_population']
    gov_init_auth = p['LUTS']['gov_init_auth']
    industry_demand_lut = p['LUTS']['industry_demand_lut']
    growth_rates_lut  = p['LUTS']['gov_growth_rate']
    hiring_rates_lut = p['LUTS']['gov_hiring_rate']
    promotion_delays_lut = p['LUTS']['promotion_delays_lut']
    separation_rates_lut = p['LUTS']['gov_separation_rate']
    retirement_rates_lut = p['LUTS']['gov_retirement_rate']
    carry_cap = {level: gov_init_auth[level] * p["carry_cap"] for level in gs_levels}

    
    # Stocks for each GS level
    GS = {level: np.zeros(steps) for level in gs_levels}    
    
    # Authorized positions for each GS level
    Authorized_GS = {level: np.zeros(steps) for level in gs_levels}
    
    # Initialize stocks (ensure a minimum of 1 for both)
    for level in gs_levels:
        GS[level][0] = max(1, gov_init_population[level])
        Authorized_GS[level][0] = max(1, gov_init_auth[level])
    
    # Applicant pools exist for GS09, GS11, GS12, GS13, and GS14
    Applicant_Pool = {level: np.zeros(steps) for level in [9, 11, 12, 13, 14]}
    for level in [9, 11, 12, 13, 14]:
        Applicant_Pool[level][0] = gov_init_population[level] * 0.1  # initial pool
    
    Internal_Demand = {level: np.ones(steps) for level in gs_levels}
    
    # Exiting flows from the applicant pools
    Exiting = {level: np.zeros(steps) for level in [9, 11, 12, 13, 14]}

    # Dictionaries to store flows
    Accession = {level: np.zeros(steps) for level in gs_levels}
    Separation = {level: np.zeros(steps) for level in gs_levels}
    Retirement = {level: np.zeros(steps) for level in gs_levels}
    
    Growing = {level: np.zeros(steps) for level in gs_levels}
    g_flag=False
    if growthfile != "":  
        g_flag=True
        growth = pd.read_hdf(growthfile)
        Growing = {level: growth[level].values for level in gs_levels}
        
    # For each GS level, track vacancies, vacancy rate, and salary pressure.
    Vacancies = {level: np.zeros(steps) for level in gs_levels}      
    Vacancy_Rate = {level: np.zeros(steps) for level in gs_levels}   
    Salary_Pressure = {level: np.zeros(steps) for level in gs_levels}  
    
    # Promotion flows between grades
    Promotion = {
        11: np.zeros(steps),
        12: np.zeros(steps),
        13: np.zeros(steps),
        14: np.zeros(steps),
        15: np.zeros(steps)
    }
    
    # Recruitment applies only at GS09
    Recruitment = {9: np.zeros(steps)}
    
    #Setup Salary Trackers
    GOV_Avg_Salary = np.zeros(steps)
    Avg_Salary_ByGRD = {level: np.zeros(steps) for level in gs_levels}  
    Industry_Avg_Salary = np.zeros(steps)
    Salary_Comparison = np.zeros(steps)
   
    # ---------------------------    
    # Simulation loop
    # ---------------------------   
    max_industry_salary = p['industry_init_avg_salary'] * abs(salary_cap)
    max_gov_salary = p['gov_init_avg_salary'] * abs(salary_cap)
    recruitment_rate = p["recruitment_rate"]
    ###########################################################################################################
    GS_carrying_capacity = {g: round(p['LUTS']['gov_init_auth'][g] * p["carry_cap"]) for g in gs_levels}
    gs_final_sum = sum(GS_carrying_capacity.values())
    GS_proportion = {g: GS_carrying_capacity[g] / gs_final_sum for g in gs_levels}
    gov_carrying_capacity = sum(carry_cap.values())
    GS_Min_Auth = {g: round(p['LUTS']['gov_init_auth'][g] * 0.25) for g in gs_levels} 
    ###########################################################################################################

    # -- Main Simulation Loop --
    for t in range(steps):
        
        # Set cost of living adjustments (COLA)
        Industry_COLA = np.clip(np.random.normal(p['init_industry_cola'],p['init_industry_cola']* p['sd_noise']),0,0.1)
        GOV_COLA = np.clip(np.random.normal(p['init_gov_cola'],p['init_gov_cola']* p['sd_noise']),0,0.1)
            
        # Industry Average Salary (ACTIVE INITIAL style: always exponential)
        Industry_Avg_Salary[t] = min(max_industry_salary,p["industry_init_avg_salary"] * (1 + Industry_COLA * math.log1p(t)))

        gov_avg_sal=0.0
        ttl_pop = 0
        for level in gs_levels:
            ttl_pop += GS[level][t]
            grd_ttl = GS[level][t] * min(max_gov_salary, init_gov_paytable[level] * (1 + GOV_COLA * math.log1p(t)))
            gov_avg_sal += grd_ttl
            Avg_Salary_ByGRD[level][t] = grd_ttl / GS[level][t] if GS[level][t] > 0 else 0
        
        #GOV_Avg_Salary[t] = min(max_gov_salary, p['gov_init_avg_salary'] * np.exp(GOV_COLA * math.log1p(t)))
        GOV_Avg_Salary[t] = (gov_avg_sal / ttl_pop)

        max_abs_salary = max(abs(Industry_Avg_Salary[t]), abs(GOV_Avg_Salary[t]))
        Salary_Comparison[t] = (0.5 + 0.5 * ((Industry_Avg_Salary[t] - GOV_Avg_Salary[t]) / max_abs_salary)
                             if max_abs_salary != 0 else 0.5)
        
        # Update vacancies and calculate vacancy rate for each level
        for level in gs_levels:
            Vacancies[level][t] = Authorized_GS[level][t] - GS[level][t]
            Vacancy_Rate[level][t] = Vacancies[level][t] / Authorized_GS[level][t]
            
            # Salary pressure: note GS15 uses a slightly different formulation
            if level == 15:
                Salary_Pressure[level][t] = (Vacancy_Rate[level][t] / industry_demand_lut[level]) * Salary_Comparison[t]
                Internal_Demand[level][t] = 0.5
            else:
                Salary_Pressure[level][t] = ((1 - np.exp(-2 * Vacancy_Rate[level][t])) / industry_demand_lut[level]) * Salary_Comparison[t]
                Internal_Demand[level][t] = 1 / (1 + np.exp(-((GS[level][t] + Applicant_Pool[level][t]) -  Authorized_GS[level][t]) / (0.5 *  Authorized_GS[level][t]))) if  Authorized_GS[level][t] > 0 else 1.0
        
        # --- Compute Flows for Each GS Level ---        
        # Accession flows (for GS09, GS11, GS12, GS13, GS14 use applicant pool)
        max_possible={}
        for level in gs_levels:
            max_possible[level] = min(Vacancies[level][t], Authorized_GS[level][t]-GS[level][t])
            if max_possible[level] > 0:
                hr = hiring_rates_lut[level]
                accessions=0
                
                if time[t] > 0:
                    base_hiring = hr * Vacancies[level][t]
                    salary_competition_factor = np.exp(temperature * Salary_Pressure[level][t]) 
                    #money_driven_factor = np.exp(temperature * p['money_driven'])
                    money_driven_factor = np.exp(temperature * p['money_driven']) * Internal_Demand[level][t]
                    accessions = base_hiring * salary_competition_factor * money_driven_factor
                else:
                    accessions = hr * Vacancies[level][t]

                Accession[level][t] = max(0, min(accessions, max_possible[level]))
            else:
                Accession[level][t] = 0 
            max_possible[level] -=  Accession[level][t]
            
        ############################################################
        # Recruitment (only for level 9 / GS-09)
        #Recruitment[9][t] = max(0, (Vacancies[9][t]-Accession[9][t]) *  recruitment_rate)         
        max_recruit = max_possible[9] * p['auth_fill_rate'] *  recruitment_rate
        if max_recruit > 0:
            if max_recruit + GS[9][t] > Authorized_GS[9][t]:
                max_recruit = max(0, Authorized_GS[9][t] - GS[9][t])
            #Recruitment[9][t] = max_recruit
            Recruitment[9][t] = max_recruit * (1 + np.log1p(Internal_Demand[9][t]))
        else:
            Recruitment[9][t] = 0
                
        # Retirement flows (common formulation)
        for level in gs_levels:
            base_factor = GS[level][t] * retirement_rates_lut[level]                         
            effe_fact = (1 + ((Salary_Comparison[t]  *  p["money_driven"]) / (1 + Salary_Comparison[t] *  p["mission_focused"])))
            Retirement[level][t] =  max(0, base_factor * effe_fact)
        
        # Applicant Pool exiting flows (for GS09, GS11, GS12, GS13, GS14)
        for level in [9, 11, 12, 13, 14]:
            #Exiting[level][t] = Applicant_Pool[level][t] * min(1, p["money_driven"] *
            #                          (industry_demand_lut[level] * (1 + Salary_Pressure[level][t])))
            base_rate = industry_demand_lut[level] * (1 + Salary_Pressure[level][t])
            effective_rate = base_rate * (1 + p['money_driven'] )
            effective_rate = min(1.0, effective_rate)
            Exiting[level][t] = Applicant_Pool[level][t] * effective_rate

        # Separation flows (for GS09, GS11, GS12, GS13, GS14 add the Power Seeking term)
        Separation[9][t] = max(0,    (GS[9][t] * p["mission_focused"] * separation_rates_lut[9])  +
                                     (GS[9][t] * max(p["power_seeking"] * np.exp(temperature*Vacancy_Rate[11][t]),
                                                     p["power_seeking"])* separation_rates_lut[9]) +
                                     (GS[9][t] * p["money_driven"] * Salary_Pressure[9][t] * industry_demand_lut[9]))
                                     
        Separation[11][t] = max(0,    (GS[11][t] * separation_rates_lut[11] * p["mission_focused"]) +
                                      (GS[11][t] * max(p["power_seeking"] * np.exp(temperature*Vacancy_Rate[12][t]), 
                                                       p["power_seeking"]) * separation_rates_lut[11]) +
                                      (GS[11][t] * p["money_driven"] * Salary_Pressure[11][t] * industry_demand_lut[11]))
        Separation[12][t] = max(0,    (GS[12][t] * separation_rates_lut[12] * p["mission_focused"]) +
                                      (GS[12][t] * max(p["power_seeking"] * np.exp(temperature*Vacancy_Rate[13][t]),
                                                       p["power_seeking"]) * separation_rates_lut[12]) +
                                      (GS[12][t] * p["money_driven"] * Salary_Pressure[12][t] * industry_demand_lut[12]))
        Separation[13][t] = max(0,    (GS[13][t] * separation_rates_lut[13] * p["mission_focused"]) +
                                      (GS[13][t] * max(p["power_seeking"] * np.exp(temperature*Vacancy_Rate[14][t]),
                                                       p["power_seeking"]) * separation_rates_lut[13]) +
                                      (GS[13][t] * p["money_driven"] * Salary_Pressure[13][t] * industry_demand_lut[13]))
        Separation[14][t] = max(0,    (GS[14][t] * separation_rates_lut[14] * p["mission_focused"]) +
                                      (GS[14][t] * max(p["power_seeking"] * np.exp(temperature*Vacancy_Rate[15][t]),
                                                       p["power_seeking"]) * separation_rates_lut[14]) +
                                      (GS[14][t] * p["money_driven"] * Salary_Pressure[14][t] * industry_demand_lut[14]))
                                      
        # For GS15 the Power Seeking term is subtractive. CHANGED
        Separation[15][t] = max(0, (GS[15][t] * separation_rates_lut[15] * p["mission_focused"]) -
                                      (GS[15][t] * p["power_seeking"] * separation_rates_lut[15]) +
                                      (GS[15][t] * p["money_driven"] * Salary_Pressure[15][t] * industry_demand_lut[15]))
        

        # Promotion flows between grades
        #prom_delay_11 = np.exp(-1 / promotion_delays_lut[11])
        prom_delay_11=promotion_delays_lut[11]
        Promotion[11][t] = max(0, min( GS[9][t] / prom_delay_11, Vacancies[11][t] / prom_delay_11))
        if promotion_delays_lut[11] >= 1000:
            Promotion[11][t] = 0
        
        #prom_delay_12 = np.exp(-1 / promotion_delays_lut[12])
        prom_delay_12=promotion_delays_lut[12]
        Promotion[12][t] = max(0, min(GS[11][t] / prom_delay_12, Vacancies[12][t] / prom_delay_12))
        if promotion_delays_lut[12] >= 1000:
            Promotion[12][t] = 0
        
        #prom_delay_13 = np.exp(-1 / promotion_delays_lut[13])
        prom_delay_13=promotion_delays_lut[13]
        Promotion[13][t] = max(0, min(GS[12][t] / prom_delay_13, Vacancies[13][t] / prom_delay_13))
        if promotion_delays_lut[13] >= 1000:
            Promotion[13][t] = 0
        
        #prom_delay_14 = np.exp(-1 / promotion_delays_lut[14])
        prom_delay_14=promotion_delays_lut[14]
        Promotion[14][t] = max(0, min(GS[13][t] / prom_delay_14, Vacancies[14][t] / prom_delay_14))
        if promotion_delays_lut[14] >= 1000:
            Promotion[14][t] = 0
        
        #prom_delay_15 = np.exp(-1 / promotion_delays_lut[15])
        prom_delay_15=promotion_delays_lut[15]
        Promotion[15][t] = max(0, min(GS[14][t] / prom_delay_15, Vacancies[15][t] / prom_delay_15))
        if promotion_delays_lut[15] >= 1000:
            Promotion[15][t] = 0
        
        # Growing flows for Authorized positions
        for level in gs_levels:
            if not g_flag:
                predicted_auth = sinusoidal_with_trend(
                    t,
                    Authorized_GS[level][t],
                    p['lin_growth_rate'],
                    p['sin_amp'],
                    p['phase_shft'],
                    p['ang_freq'],
                    GS_carrying_capacity[level]
                )
                # Use the predicted value as the "growing" variable.
                Growing[level][t] = max(GS_Min_Auth[level],predicted_auth)
                    
        # --- Stock Updates (Euler integration) ---
        debug=False
        if t == 0:
            if debug: print(f"time\tLVL\tGS[t]\tAccession\tRecruit/Promote\tSeparation\tRetirement\tPromotion")
        if t < (steps - 1):
            # Update GS stocks
            lvl=9
            if debug: print(f"{t}\t{lvl}\t{GS[lvl][t]}\t{Accession[lvl][t]}\t{Recruitment[lvl][t]}\t{Separation[lvl][t]}\t{Retirement[lvl][t]}\t{Promotion[lvl+2][t]}")
            GS[9][t+1]  = GS[9][t] + Accession[9][t] + Recruitment[9][t] - Separation[9][t] - Retirement[9][t] - Promotion[11][t]
            GS[9][t+1] = max(GS[9][t+1], 0)
            
            lvl=11
            if debug: print(f"{t}\t{lvl}\t{GS[lvl][t]}\t{Accession[lvl][t]}\t{Promotion[lvl][t]}\t{Separation[lvl][t]}\t{Retirement[lvl][t]}\t{Promotion[lvl+1][t]}")
            GS[11][t+1] = GS[11][t] + Accession[11][t] + Promotion[11][t] - Retirement[11][t] - Separation[11][t] - Promotion[12][t]
            GS[11][t+1] = max(GS[11][t+1], 0)
            
            lvl=12
            if debug: print(f"{t}\t{lvl}\t{GS[lvl][t]}\t{Accession[lvl][t]}\t{Promotion[lvl][t]}\t{Separation[lvl][t]}\t{Retirement[lvl][t]}\t{Promotion[lvl+1][t]}")
            GS[12][t+1] = GS[12][t] + Accession[12][t] + Promotion[12][t] - Retirement[12][t] - Separation[12][t] - Promotion[13][t]
            GS[12][t+1] = max(GS[12][t+1], 0)
            
            lvl=13
            if debug: print(f"{t}\t{lvl}\t{GS[lvl][t]}\t{Accession[lvl][t]}\t{Promotion[lvl][t]}\t{Separation[lvl][t]}\t{Retirement[lvl][t]}\t{Promotion[lvl+1][t]}")
            GS[13][t+1] = GS[13][t] + Accession[13][t] + Promotion[13][t] - Retirement[13][t] - Separation[13][t] - Promotion[14][t]
            GS[13][t+1] = max(GS[13][t+1], 0)
            
            lvl=14
            if debug: print(f"{t}\t{lvl}\t{GS[lvl][t]}\t{Accession[lvl][t]}\t{Promotion[lvl][t]}\t{Separation[lvl][t]}\t{Retirement[lvl][t]}\t{Promotion[lvl+1][t]}")
            GS[14][t+1] = GS[14][t] + Accession[14][t] + Promotion[14][t] - Retirement[14][t] - Separation[14][t] - Promotion[15][t]
            GS[14][t+1] = max(GS[14][t+1], 0)
            
            lvl=15
            if debug: print(f"{t}\t{lvl}\t{GS[lvl][t]}\t{Accession[lvl][t]}\t{Promotion[lvl][t]}\t{Separation[lvl][t]}\t{Retirement[lvl][t]}")
            GS[15][t+1] = GS[15][t] + Accession[15][t] + Promotion[15][t] - Retirement[15][t] - Separation[15][t]
            GS[15][t+1] = max(GS[15][t+1], 0)
            
        
            # Update Applicant Pools
            for level in [9, 11, 12, 13, 14]:
                Applicant_Pool[level][t+1] = Applicant_Pool[level][t] + dt * (Separation[level][t] - Exiting[level][t])
                
            # Update Authorized positions
            for level in gs_levels:
                Authorized_GS[level][t+1] = Growing[level][t]
            
            
            #Adjust Recruitment Rate    
            #recruitment_rate[t+1] = recruitment_rate[t] # NO ADJUSTMENT AT THIS TIME
            
    return {
        "time": time,
        "GS": GS,
        "Authorized_GS": Authorized_GS,
        "Applicant_Pool_GS": Applicant_Pool,
        "Accession": Accession,
        "Separation": Separation,
        "Retirement": Retirement,
        "Growing": Growing,
        "Promotion": Promotion,
        "Recruitment": Recruitment,
        "Exiting": Exiting,
        "Vacancies": Vacancies,
        "Vacancy_Rate": Vacancy_Rate,
        "Salary_Pressure": Salary_Pressure,
        "Salary_Comparison": Salary_Comparison,
        "Industry_Avg_Salary": Industry_Avg_Salary,
        "GOV_Avg_Salary": GOV_Avg_Salary,
        'Avg_Salary_ByGRD': Avg_Salary_ByGRD,
        "Recruitment_Rate": recruitment_rate,
        "Carry_Capacity": GS_carrying_capacity
    }


