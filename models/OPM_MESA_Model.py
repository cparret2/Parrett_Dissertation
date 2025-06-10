import mesa
from OPM_Agent import GLBL_AGT_STATUS
from OPM_MESA_Agent import *
from OPM_Ages import OPM_Ages
from OPM_Salary_Tool import *
from OPM_VacancyBoard import VacancyBoard, Vacancy
from gen_movie_from_GML import GenFrames
from OPM_SocialNet import OPM_SocialNet
import networkx as nx
import numpy as np
import pandas as pd
from random import choices
import os

#####################################################################################
#####################################################################################

class OPM_MESA_Model(mesa.Model):
    def __init__(self,data,simyear, hi_vars,ll_vars, maxper=50, m=4,temperature=-1.5,
                 carry_cap=6.5,social=True,grphdname="",debug=False,writeonly=False,nowrite=False):
        super().__init__()
        self.DEBUG = debug
        self.writeonly=writeonly
        self.nowrite=nowrite
        self.gs_levels = [9,11,12,13,14,15]
        self.curryear=simyear
        self.m_rates = hi_vars
        self.noise= self.m_rates['sd_noise']
        self.grd_rates = ll_vars
        self.pref_attach = m
        self.temperature=temperature
        self.socialmodel = social
        self.agemodel = OPM_Ages()
        self.paytable = OPM_Salary_Tool(self.curryear)
        self.grphdname=grphdname
        
        # Default Model Values
        self.wgi_schedule = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3}
        self.performance = hi_vars['perf_def']
        self.ext_avgsal = self.m_rates["industry_init_avg_salary"]
        
        #self.gov_growth_rate= self.m_rates["GOV Growth Rate"]
        self.maxper=maxper
        self.mean_power_seeking = self.m_rates['power_seeking']
        self.mean_money_driven  = self.m_rates['money_driven']
        self.mean_mission_focus = 1 - self.mean_power_seeking - self.mean_money_driven
        self.industry_cola = pd.Series(np.zeros(self.maxper+1))
        self.gov_cola = pd.Series(np.zeros(self.maxper+1))

       
        # External economics
        self.salary_comparison = 0.5
        self.salary_pressure = {9: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
        self.industry_demand = {9: 0.1, 11: 0.2, 12: 0.6, 13: 0.70, 14: 0.50, 15: 0.40}

        # STATS
        self.retirees_stat = 0
        self.fired_stat = 0
        self.promotees_stat = 0
        self.transfers_stat = 0
        self.exiting_stat = 0
        self.quitting_stat = 0
        self.newhires_stat = 0
        self.recruits_stat = 0
        
        ## Internal Model controls
        self._actions ={}
        self._perf_curv = []
        
        #------------------------------------------
        #Setup Vacancy Board
        # Get the initial ORG size
        self.vacbrd = VacancyBoard(self, data)
        
        for _, rec in data.iterrows():
            a = self._add_agents_with_probability(rec)
        
        # Create Social Network
        self.social_network = OPM_SocialNet(self)

        # Generate the performance distribution
        self.Initialize()

        self.vacbrd.InitializeTalentReqs()
        
        self.vacbrd.CalcVacancyRates()
        
        # Register in Mesa --> Not using to move though...
        self.history = [["step","agent_id","action","pop","auth"]]
        self.datacollector = mesa.DataCollector(
            model_reporters={"mean_age": lambda m: m.agents.agg("age", np.mean),
                            "avg_sal": lambda m: m.agents.agg("salary", np.mean),
                            "sal_compare": lambda m: m.salary_comparison,
                            "sal_press": lambda m: sum(m.salary_pressure.values()),
                            "avg_tenure": lambda m: m.agents.agg("los", np.mean),
                            "avg_performance": lambda m: m.agents.agg("perf", np.mean),
                            "GS_population" : lambda m: m.vacbrd.GetGSPop_Dict(),
                            "ttl_population": lambda m: len(m.agents),
                            "GS_authorization": lambda m: m.vacbrd.GetAuthGS_Dict(),
                            "ttl_authorization": lambda m: m.vacbrd.GetTotalAuthorizedLvl(),
                            "retirees_dt": lambda m: m.retirees_stat,
                            "fired_dt": lambda m: m.fired_stat,
                            "promotees_dt": lambda m: m.promotees_stat,
                            "exiting_dt":lambda m: m.exiting_stat,
                            "quitting_dt": lambda m: m.quitting_stat,
                            "transfers_dt":lambda m: m.transfers_stat,
                            "newhires_dt":lambda m: m.newhires_stat,
                            "recruits_dt":lambda m: m.recruits_stat,
                            "vacancies_dt": lambda m: m.vacbrd.GetTotalVacancies(),
                            "avg_influence": lambda m: m.agents.agg("influence", np.mean),
                            "std_influence": lambda m: m.agents.agg("influence", np.std)
                            #"max_influence": lambda m: m.agents.agg("influence", np.max),
                            #"min_influence": lambda m: m.agents.agg("influence", np.min)
                            },
            agent_reporters={"age": "age", "grade": "grade", "status": "status", 
                             'location': 'location','agelvl': 'agelvl','edlevel':'edlevel',
                             'gstp': 'gstp','loslvl': 'loslvl','sallvl': 'sallvl','salary':'salary',
                             'los': 'los','perf':'perf','money_driven':'money_driven',
                             'power_seeking':'power_seeking','influence':'influence'}
        )
    
    def _add_agents_with_probability(self,rec):
        md= np.clip(np.random.normal(self.mean_money_driven, self.mean_money_driven *self.noise),0.0,0.9)
        ps= np.clip(np.random.normal(self.mean_power_seeking,self.mean_power_seeking*self.noise),0.0,0.9)
        total = md + ps
        if total >= 0.95:
            scale = 0.95 / total
            md *= scale
            ps *= scale
        mf = 1.0 - md - ps
        if mf <= 0:
            # Redistribute to guarantee minimum mf
            mf = 0.05
            remaining = 0.95
            md_ratio = md / (md + ps)
            md = remaining * md_ratio
            ps = remaining * (1 - md_ratio)
        agt = OPM_MESA_Agent(self,rec,ps,md)
        #FOR NOW JUST RANDOM NORMAL... FUTURE CAN IMPROVE, etc
        agt.perf = np.random.normal(self.performance["mean"],self.performance["std"])
        return agt
 
    ######################################################################            
    #def _calcRetirementCurve(self,gr=0.4,infl=25):
    #    growth_rate=gr
    #    inflection=infl
        
        # Shift midpoint to control where the logistic curve starts rising rapidly
    #    mid_point = self.min_vestment + inflection  # Logistic inflection point is around min_los + Y

        # Logistic Model: Gradual increase with controlled inflection and start
    #    self.prob_retire = [
    #        1 / (1 + np.exp(-growth_rate * (x - mid_point))) if x > self.min_vestment else 0
    #        for x in range(self.min_vestment, self.max_vestment + 1)
    #    ]
    
    ######################################################################
    def _reset_stats(self):
        self.retirees_stat = 0
        self.fired_stat = 0
        self.promotees_stat = 0
        self.exiting_stat = 0
        self.transfers_stat = 0
        self.newhires_stat = 0
        self.recruits_stat=0
        self.quitting_stat = 0
        
        self._perf_curv=[]
        
    ######################################################################
    def Initialize(self):
        #self._calcRetirementCurve()
        self.perf_curv = list(np.random.normal(self.performance["mean"],self.performance["std"],len(self.agents)))
        np.random.shuffle(self.perf_curv) # Convert to list for shuffling
        for agt, perf in zip(self.agents, self.perf_curv):
            #initialize the agents... use the estimated age
            agt.perf = perf
            agt.Reflect()
            
        self.social_network.Generate(self.pref_attach)
        if self.grphdname != "" and not self.nowrite:
            file_path = f"{self.grphdname}/ABM_SocNet_test_case_0.gml"
            if not os.path.exists(file_path):
                self.social_network.DumpGraphGML(file_path)
                GenFrames(f"{self.grphdname}",writeonly=self.writeonly,spec_frame=0)
    
      
    ######################################################################            
    def _gen_COLA_curves(self,n=50,mode="static"):
        def genrandnum(n, mean, std, lower=0.0, upper=0.08):
            numbers = np.random.normal(loc=mean, scale=std, size=n)
            numbers = np.clip(numbers, lower, upper)
            return numbers
        if mode == "static":
            self.industry_cola += self.m_rates["init_industry_cola"]
            self.GOV_COLA += self.m_rates["init_gov_cola"]
        elif mode == "mirror_rand":
            self.industry_cola += genrandnum(n,self.industry_cola,self.noise)
            self.gov_cola = self.industry_cola
        elif mode == "indep_rand":
            self.industry_cola += genrandnum(n,self.industry_cola,self.noise)
            self.gov_cola += genrandnum(n,self.gov_cola,self.noise)
    
   
    ######################################################################            
    def _update_perfcurve(self,fire_t=1.5,promo_t=1.0):
        self._perf_curv= pd.Series(self._perf_curv)
        mean_t = self._perf_curv.mean()
        std_t = self._perf_curv.std()
        
        # Define bounds
        fire_thrshld = mean_t - fire_t * std_t
        promote_thrshld = mean_t + promo_t * std_t
        
        return {"fire": fire_thrshld, "promote": promote_thrshld, "WGI": mean_t}
    
    def Separations(self):
        pass
        
    # 1 #####################################################################    
    def Start_Period(self):
        '''
        Start Period resets the statistics and prepares the peformance curve.
        '''
        self._reset_stats()
        for l_grd in [9,11,12,13,14,15]:
            agts = self.agents.select(lambda m: m.grade == l_grd)
            #for a in agts.shuffle(inplace=True):
            rand_agts = agts.shuffle()
            for a in rand_agts:
                # Update each agent 'a'
                a.step()
                self._perf_curv.append(a.perf)       
        self._actions = self._update_perfcurve()
        if self.DEBUG: print(f"Step {self.steps}: Num Agents {len(self.agents)}")
    # 2 #####################################################################
    def Fire(self,actions):
        '''
        FIRE - For those that have under performed flag for firing...
        '''
        f_stat = {}
        fired = self.agents.select(lambda m: m.perf < actions)
        self.fired_stat = len(fired)
        if self.fired_stat > 0:
            fired_agts = fired.select(at_most=self.fired_stat)
            for agt in fired_agts:
                if not agt.grade in f_stat: f_stat[agt.grade] = 0
                f_stat[agt.grade]+=1
                agt.Fire()
                self.vacbrd.AddVacancy(agt,"F") #fired --> Replace with new agent
                
        if self.DEBUG: print(f"\t Fired:\t{sum(f_stat.values())}==>{f_stat}")
    # 2B #####################################################################
    def Separate(self): ## **** ONLY FOR COMPARISON           
        rand_sep = self.agents.select(at_most=len(self.agents)*self.m_rates['gov_separation_rate'])
        f_stat = {}
        for agt in rand_sep:
            agt.Exit()
            self.vacbrd.AddVacancy(agt,"X")
            
            if not agt.grade in f_stat: f_stat[agt.grade] = 0
            f_stat[agt.grade]+=1
            
        if self.DEBUG: print(f"\t Separated:\t{sum(f_stat.values())}==>{f_stat}")

    # 3 #####################################################################           
    def Promote(self,actions):
        
        f_stat = {}
        # Get all agents that are performing above the perf curve
        promote_eligible = self.agents.select(lambda m: m.isActive() and m.perf > actions and m.grade < 15)
        if len(promote_eligible) > 0:
            for agt in promote_eligible:
                promo_grade = self.paytable.GetNextGrade(agt.grade)    
                if promo_grade > 0:
                    #authorized = self.authorized_gs[promo_grade]
                    authorized = self.vacbrd.GetAuthorizedGradeLvl(promo_grade)
                    current = len(self.agents.select(lambda m: m.grade == promo_grade and m.status == 'A'))
    
                    if current < authorized and self.vacbrd.GetNumAvailPromotions(promo_grade) > 0:
                        #rand_p = np.random.randint(self.grd_rates['promotion_delays_lut'][promo_grade])
                        if agt.timeinstatus > self.grd_rates['promotion_delays_lut'][promo_grade]:
                            self.promotees_stat+=1
                            agt.Promote() #Status only... there may not be space
                            self.vacbrd.AddVacancy(agt,"P",promo_grade)
                            if not agt.grade in f_stat: f_stat[agt.grade] = 0
                            f_stat[agt.grade]+=1
                else:
                    if self.DEBUG: print(f"Number of promotions to {promo_grade} is {self.vacbrd.GetGradeVacancies(promo_grade)}")
                    
        if self.DEBUG: print(f"\t Promoted:\t{sum(f_stat.values())}==>{f_stat}")

    # 4 #####################################################################    
    def UpdateIndustryParameters(self):
        # Random draws for COLA (using a normal distribution then clipping)
        Industry_Avg_Salary = self.m_rates["industry_init_avg_salary"] * np.exp(self.industry_cola[self.steps] * self.steps) #F(t)
        
        # GOV Average Salary (ACTIVE INITIAL):
        mdl_out = self.datacollector.get_model_vars_dataframe()
        if len(mdl_out) > 0:
            GOV_Avg_Salary = mdl_out.iloc[-1]["avg_sal"]
        else:
            GOV_Avg_Salary= self.m_rates["gov_init_avg_salary"] 
            
        # Salary Comparison: a normalized measure of pay difference.
        max_abs_salary = max(abs(Industry_Avg_Salary), abs(GOV_Avg_Salary))
        
        self.salary_comparison = Industry_Avg_Salary - GOV_Avg_Salary
        
        sal_fact = (0.5 + 0.5 * (self.salary_comparison / max_abs_salary) if max_abs_salary != 0 else 0.5)
        
        # Vacancy_Rate
        vacrate =self.vacbrd.CalcVacancyRates()
        for grd in vacrate.keys():
            if grd == 15:
                self.salary_pressure[grd] = sal_fact * (vacrate[grd] / self.industry_demand[grd])
            else:
                self.salary_pressure[grd] = (vacrate[grd] / self.industry_demand[grd]) * sal_fact
    
    # 5 #####################################################################    
    def Work(self,max_comms=6):
        #for agt in self.agents.shuffle(inplace=True):
        for agt in self.agents.shuffle():
            ngbrs = list(nx.neighbors(self.social_network.G,agt.unique_id))
            nnum = min(max_comms,len(ngbrs))
            agt.UpdatePreferences(ngbrs)
            agt.CareerDecision()
            
    # 6 #####################################################################    
    def ExecuteMoves(self):
        """
        Executes end-of-step personnel movements:
        - Promoted agents generate vacancies at their current grade.
        - Transfers are processed first to fill vacancies.
        - Promotions are then applied, followed by any agents who are passed over.
        """
     
        for l_grd in [9, 11, 12, 13, 14, 15]:
            # --- TRANSFERS FIRST ---
            tfrs = self.vacbrd.ProcessTransfers(l_grd)
            self.transfers_stat = len(tfrs)
            for t in tfrs:               
               #self.history.append([self.steps, t[1].unique_id, 'M', len(self.agents), self.vacbrd.GetTotalAuthorizedLvl()])
                t[1].GetNewPos(t[0])


        agt_promos = self.vacbrd.ProcessAllPromotions()
        self.promotees_stat=0
        for l_grd in agt_promos.keys():
            promo_grade = self.paytable.GetNextGrade(l_grd)
            for agt in agt_promos[promo_grade]:    
                if self.vacbrd.GetGradePopulation(promo_grade) < self.vacbrd.GetAuthorizedGradeLvl(promo_grade):
                    agt[1].GetPromotion(promo_grade)
                    self.promotees_stat+=1
                else:
                    print(f"Promotion to {promo_grade} is going to exceed auth ceiling {self.vacbrd.GetAuthorizedGradeLvl(promo_grade)} at t={self.steps}")        
                    break

        #DEBUG ONLY
        leftoveragts = self.agents.select(lambda m: m.isPromoted())
        leftoveragts.map("GetPassedOverForPromo")

        
    # 7 #####################################################################
    def Hire(self):
        f_stat={}
        for l_grd in [9,11,12,13,14,15]:
            if self.vacbrd.GetNumNewHires(l_grd) > 0:
                hires = self.vacbrd.ProcessNewHires(l_grd)
                
                #n_agts = self.agents.select(lambda m: m.grade == l_grd)
                #current_pop = len(n_agts)
                #allowed_slots = int(self.authorized_gs[l_grd] - current_pop)
                allowed_slots = int(self.vacbrd.GetAuthorizedGradeLvl(l_grd) - self.vacbrd.GetGradePopulation(l_grd))
                selected_hires = []
                if allowed_slots > 0:
                    selected_hires = hires[0:allowed_slots]
                
                avg_ages = self.agemodel.GetRandAgeList(l_grd,self.grd_rates['gov_avg_acc_age'][l_grd],
                                                        self.m_rates['sd_noise'],hires)
                for vac in selected_hires:
                    self.newhires_stat+=1
                    l_age = avg_ages.pop()
                    a = self._add_agents_with_probability(vac[1].get_rec(l_age))                                        
                    
                    #self.history.append([self.steps,a.unique_id,'H',len(self.agents),sum(self.vacbrd.GetTotalAuthorizedLvl())])
                
                    # Add to Social Network
                    self.social_network.AddNewAgtToNet(a)

                    if not l_grd in f_stat: f_stat[l_grd] = 0
                    f_stat[l_grd]+=1
        
        if self.DEBUG: print(f"\t Hired:\t{sum(f_stat.values())}==>{f_stat}")
            
    # 8 #####################################################################    
    def Recruit(self):
        #n_agts = self.agents.select(lambda m: m.grade == 9)
        allowed_pop = max(0, self.vacbrd.GetAuthorizedGradeLvl(9) - self.vacbrd.GetGradePopulation(9))
        
        sel_hires= round(self.m_rates['auth_fill_rate'] * self.m_rates['recruitment_rate'] * allowed_pop)

        #avg_ages = self.agemodel.GetRandAgeList(9,self.grd_rates['gov_avg_newhire_age'][9],
        #                                        self.m_rates['sd_noise'],sel_hires)
        avg_ages = self.agemodel.GetRandAgeList(9, np.random.randint(self.agemodel.agebnds['B'][0],self.agemodel.agebnds['C'][1]),
                                                                     self.m_rates['sd_noise'],sel_hires)
        
        n_locs = self.vacbrd.GetLocationProbs(sel_hires)
        
        for l_age,l_loc in zip(avg_ages, n_locs):
            self.recruits_stat+=1
            #vac[0] == LOC
            #l_age = avg_ages.pop()
            rec = { 
                'LOC': l_loc,
                'EDLVL': np.random.choice(['A','B','C']),
                'GSEGRD': 9,
                'PPGRD':'GS-09',
                'SALARY'  : self.paytable.GetSalary(9, 1),
                'LOS': 0.0,
                'AGE': l_age
                }
            a = self._add_agents_with_probability(rec)  
            
            # Add to Social Network
            self.social_network.AddNewAgtToNet(a)
        
        #self.vacbrd.ProcessRecruits(sel_hires)
        
        if self.DEBUG: print(f"\t Recruited: {sel_hires}")
    
    # 9 #####################################################################    
    def End_Period(self):
        #CLEAN UP AGENTS
        f_stat={}
        remove_agts = self.agents.select(lambda m: m.status == "R")
        self.retirees_stat=len(remove_agts)
        for agt in remove_agts:
            if not agt.grade in f_stat: f_stat[agt.grade] = 0
            f_stat[agt.grade]+=1
            
            self.social_network.G.remove_node(agt.unique_id)
            #print(f"Agent {agt.unique_id} Retiring->pop {len(self.agents)}")
            #self.history.append([self.steps,agt.unique_id,'R',len(self.agents),self.vacbrd.GetTotalAuthorizedLvl()])
            agt.remove()
        if self.DEBUG: print(f"\t Retired:\t{sum(f_stat.values())}==>{f_stat}")
        
        f_stat={}
        
        self.exiting_stat = len(self.agents.select(lambda m: m.status in ["X"]))
        self.fired_stat = len(self.agents.select(lambda m: m.status in ["F"]))
        self.quitting_stat = len(self.agents.select(lambda m: m.status in ["Q"]))
        remove_agts = self.agents.select(lambda m: m.status in ["F","Q","X"])
        for agt in remove_agts:
            if not agt.grade in f_stat: f_stat[agt.grade] = 0
            f_stat[agt.grade]+=1
            
            self.social_network.G.remove_node(agt.unique_id)
            
            #print(f"Agent {agt.unique_id} {agt.status}->pop {len(self.agents)}")
            #self.history.append([self.steps,agt.unique_id,agt.status,len(self.agents),self.vacbrd.GetTotalAuthorizedLvl()])
            
            agt.remove()
        if self.DEBUG: print(f"\t Exited:\t{sum(f_stat.values())}==>{f_stat}")
    
    # ----------------------------
    # Define a sinusoidal + trend function
    # ----------------------------
    def _sinusoidal_with_trend(self, lvl,sincrv=True):
        t = np.asarray(self.steps, float)
        #a = self.grd_rates['gov_init_auth'][lvl]
        a = self.vacbrd.GetAuthorizedGradeLvl(lvl)
        b = self.m_rates['lin_growth_rate']
        c = self.m_rates['sin_amp']
        d = self.m_rates['phase_shft']
        w = self.m_rates['ang_freq']
        K = self.vacbrd.GetCarryCapGradeLvl(lvl)
        
        if sincrv:
            A = (K - a) / a
            r = b / a
            trend = K / (1 + A * np.exp(-r * t))
        else:
            trend = a + b * t
        cycle = c * np.sin(w * t + d)
        return trend + cycle
    
    def GovGrowth(self, sincrv=True, rand=False, cap_factor=4.25):
        # Get the base growth rate as a Series (e.g., {9: 0.3, 11: 0.3, ...})
        add_growth = {}
        for level in self.gs_levels:
            sin_val = self._sinusoidal_with_trend(level)
            cap_val = self.vacbrd.GetCarryCapGradeLvl(level)
            cur_ath = self.vacbrd.GetAuthorizedGradeLvl(level)
            cur_agts = self.vacbrd.GetGradePopulation(level)
            add_growth[level] = max(0, round(min(sin_val,cap_val) - cur_ath))
            self.vacbrd.UpdateAuthGradeLvl(level, round(min(sin_val,cap_val)))
            
        self.vacbrd.UpdateVacancies(add_growth)

    ######################################################################    
    def step(self):
        self.Start_Period()
        
        ######################################################################
        # FIRE - Set status ONLY (don't remove)
        self.Fire(self._actions["fire"])
       
        ######################################################################    
        # PROMOTE - Set status ONLY (don't remove)
        self.Promote(self._actions["promote"])

        self.UpdateIndustryParameters()

        #Agents Interact through their network...
        if self.socialmodel:
            if self.steps == 0:
                print("SOCIAL NETWORK ON")
            self.Work()
        
        ######################################################################    
        # ExecuteMoves
        self.ExecuteMoves()
        
        ######################################################################    
        # HIRE
        self.Hire()

        self.Recruit()

        self.End_Period()
        
        # Capture Data
        self.datacollector.collect(self)
        if self.socialmodel and not self.nowrite:
            
            if self.grphdname != "" and not self.nowrite:
                file_path = f"{self.grphdname}/ABM_SocNet_test_case_{self.steps}.gml"
                if not os.path.exists(file_path):
                    self.social_network.DumpGraphGML(file_path)
                    GenFrames(f"{self.grphdname}",writeonly=self.writeonly,spec_frame=self.steps)
        # Update Salaries
        #if self.curryear is a whole year....
        self.curryear += 1
        self.GovGrowth()
        self.paytable.UpdatePayTable(self.curryear)
        self.social_network.Rewire(r_prob=0.05, nlinks=self.pref_attach)
        
        