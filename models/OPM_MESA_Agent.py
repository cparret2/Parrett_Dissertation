import mesa
import numpy as np
from numpy.random import normal, random
from OPM_Agent import OPM_Agent

class OPM_MESA_Agent(mesa.Agent, OPM_Agent):
    """
    OPMAgent == MESA
    """
    def __init__(self, model, rec, ps=0.1, md=0.1):
        # Initialize the agent by calling the parent class constructor.
        p_age = 18
        if "AGE" in rec: p_age = rec['AGE']
        else: p_age = model.agemodel.GetRandAge(rec['AGELVL'])
        super().__init__( model, rec, p_age)
        self.start = model.steps
        self.gstp = self.model.paytable.GetGradeStep(self.grade,self.salary)
        self.perf = -1.0 #For error checking later
        self.influence = 0.0
        self.power_seeking=ps
        self.money_driven=md
        self.mission_focus= 1.0 - self.money_driven - self.power_seeking
        if self.mission_focus <= 0.0:
            assert self.mission_focus + self.power_seeking +self.money_driven == 1.0
        self.retire_preference = 0.0
        self.r_factor = 0.0
        self.performance_drive = 0.0
        self.p_factor = 0.0
        self.delta_t =0
        
    def UpdatePreferences(self, neighbors):
        agts = self.model.agents.select(lambda agent: agent._nodal_pos in neighbors)
        for neighbor in agts:
            influence = neighbor.influence
            if neighbor.status == 'R' and neighbor.grade == self.grade:
                self.retire_preference += influence * self.r_factor
            if neighbor.status == 'P' and self.grade < neighbor.grade:
                grade_diff = neighbor.grade - self.grade
                self.performance_drive += influence * self.p_factor * grade_diff
            if neighbor.status == 'F':
                self.performance_drive -= influence * 0.5

        if self.delta_t > 1:
            self.performance_drive -= (self.timeinstatus / 20) * self.p_factor

        self.retire_preference = max(0.0, self.retire_preference)
        self.performance_drive = max(0.0, self.performance_drive)
        
    
    def _compute_influence(self):
        """
        Computes agent influence based on:
        - Grade (higher rank = more influential)
        - Length of Service (longer tenure = more influential)
        - Location Size (smaller locations amplify influence)
        """
        locsize = self.model.vacbrd.location_sizes.get(self.location, 1)  # Default size 1 if location not found
        self.influence = (self.grade * (self.los + 1)) / np.log(locsize + 1)
        return self.influence
              
    def GetPromotion(self,new_grd,new_loc=""):
        self.grade = new_grd
        if new_loc != "":
            self.location=new_loc
        self.gstp, self.salary = self.model.paytable.GetPromotionStepSal(self.grade, self.salary)
        self.ppgrd = f"GS-{self.grade:02d}"
        self.status = "A"
        self.timeinstatus=0
        self.last_wgi=self.model.steps
        self._compute_influence()
        self.performance_drive = 1.0
        self.p_factor = 0.0

    def GetPassedOverForPromo(self):
        self.status = "A"
        self.last_wgi=self.model.steps
        self._compute_influence()
    
    def GetNewPos(self,new_loc):
        self.location=new_loc
        self.model.paytable.GetSalary(self.grade, self.gstp)
        self.status = "A"
        self.timeinstatus=0
        self._compute_influence()
        
    def _compareagent(self,r,l):
        if r == l:
            return 0
        elif r < l:
            return -1
        elif r > l:
            return 1
            
    def CompareGrade(self, agt): return self._compareagent(self.grade,agt.grade)
    def CompareSalary(self, agt): return self._compareagent(self.avgsal,agt.avgsal)
    def CompareAge(self, agt): return self._compareagent(self.age,agt.age)
    def CompareLos(self, agt): return self._compareagent(self.avglos,agt.avglos) 
    def CompareEd(self, agt): return self._compareagent(self.edlevel,agt.edlevel)
     
    
    
    def _compute_delta_t(self):
        '''
            needs current_step, last_wgi, timeinstatus
        '''
        self.delta_t = (self.model.steps - self.last_wgi) / (1 + self.timeinstatus)

    def _compute_promotion_factors(self):
        '''
            needs grade, money_driven, salary_pressure, power_seeking, temperature, vacbrd
        '''
        self.r_factor = self.money_driven * self.model.salary_pressure[self.grade]
        
        promo_grd = 11 if self.grade == 9 else (self.grade + 1 if self.grade < 15 else 0)
        raw_pfactor = np.exp(self.model.temperature * self.model.vacbrd.GetVacRateByGrade(promo_grd))
        
        self.p_factor = self.power_seeking * max(0.2, min(2.0, raw_pfactor))  
    
    def _compute_retirement_decision(self):
        '''
            needs age, los, agemodel
        '''
        if self.age >= self.model.agemodel.mandatory_retire_age or self.los >= self.model.agemodel.max_vestment:
            return 'mandatory'
        elif self.age >= self.model.agemodel.minumum_retire_age and self.los >= self.model.agemodel.min_vestment:
            prob = self.model.agemodel.ProbRetire(self)
            return 'voluntary' if random() <= prob else 'none'
        return 'none'
        
    def _compute_performance_update(self):
        promo_grd = 11 if self.grade == 9 else (self.grade + 1 if self.grade < 15 else 0)
        num_promos = self.model.vacbrd.GetVacRateByGrade(promo_grd)
        
        if self.performance_drive >= 0.6:
            self.perf = min(1.0, self.perf + 0.1)
        elif self.performance_drive <= 0.3:
            self.perf = max(0.0, self.perf - 0.1)
        elif self.power_seeking > 0.6 and num_promos > 0:
            self.performance_drive += 0.1 * self.power_seeking
            self.perf = min(1.0, self.perf + 0.05)
        if self.performance_drive < 0.2 and self.retire_preference > 0.3:
            self.perf = max(0.0, self.perf - 0.05)
        if self.performance_drive < 0.4 and self.power_seeking > 0.6 and num_promos == 0:
            self.performance_drive = max(0.0, self.performance_drive - 0.05)
        
    def _compute_exit_decision(self):
        salary_signal = 1 / (1 + np.exp(-self.model.salary_comparison / 5000))  # sigmoid increase with salary gap
       
        # Influence of power-seeking and promotion opportunities
        promo_grd = 11 if self.grade == 9 else (self.grade + 1 if self.grade < 15 else 0)
        vac_rate = self.model.vacbrd.GetVacRateByGrade(promo_grd)
        vacancy_signal = 1 / (1 + np.exp(-(vac_rate - 0.1) * 10))  # sigmoid, centered at vac_rate ~0.1

        # Agents stay more when vacancies are available, especially high power seekers
        stay_incentive = self.power_seeking * vacancy_signal
        attrition_pressure = self.money_driven * (1 - vacancy_signal)

        # Aggregate drive to exit
        drive_signal = (salary_signal * self.money_driven) + attrition_pressure - stay_incentive
        prob = 1 / (1 + np.exp(-5 * (drive_signal - 0.4)))

        return prob


        
    ########################################    
    ##MAIN ROUTINES##            
    ########################################
    
    def step(self,dt=1):
        """
        OPM_Mesa_Agent:
        4) 
        5) Salary Update 
        6) Update Influence
        7) Modify Performance... how?
        """
    
    
        self.age += dt          # Increment the agent's age by dt
        self.los += dt          # Increment the agent's age by dt
        self.timeinstatus+=dt   # Increment the agent's time in status by dt
         
        ############################################################################
        ### Make some decisions on Agent's future based on age and length of service
        decision = self._compute_retirement_decision()
        if decision == 'mandatory':
            # Mandatory Retirement - MUST retire
            #print(f"Agent {self.unique_id} is MANDATORILY retiring at age {self.age} and los {self.los}")
            self.Retire()
            self.model.vacbrd.AddVacancy(self,"R")
        elif decision == 'voluntary':
            #print(f"Agent {self.unique_id} is VOLUNTARILY retiring at age {self.age} and los {self.los}")
            self.Retire()
            self.model.vacbrd.AddVacancy(self,"R")
        elif random() < self.model.m_rates["gov_separation_rate"]:
            if self.model.vacbrd.GetGradePopulation(self.grade) > 0:
                self.Move()
                self.model.vacbrd.AddVacancy(self, "M")
            else:
                self.Exit()
                self.model.vacbrd.AddVacancy(self, "Q")
        ############################################################################
        
        ############################################################################
        ### Receive a time-based salary adjustment
        if self.isActive() and self.gstp < 10:
            if self.gstp == 0: print("stop")
            if len(self.model._actions) > 0 and self.perf >= self.model._actions["WGI"]:
                wgi_sched = self.model.wgi_schedule[self.gstp]
                if wgi_sched > (self.model.steps - self.last_wgi):
                    self.gstp += 1
                    self.last_wgi=self.model.steps
            self.previous_salary = self.salary
            self.salary = self.model.paytable.GetSalary(self.grade,self.gstp)
        ############################################################################
        
        ############################################################################
        # Update Influence
        self.Reflect()
        
    def Reflect(self):
        self._compute_influence()
        self._compute_delta_t()
        self._compute_promotion_factors()

    def CareerDecision(self):
    
        #if self._compute_retirement_decision() == 'voluntary' and self.retire_preference >= 0.8 and random() <= self.retire_preference:
        #    self.Retire()
        #    self.model.vacbrd.AddVacancy(self, "R")
        #    return
        if self.isActive():
            if random() < self._compute_exit_decision():
                self.Exit()
                self.model.vacbrd.AddVacancy(self, "X")
                return

        self._compute_performance_update()
        # self.perf, self.performance_drive = 
        #    self.perf, self.performance_drive, self.power_seeking,
        #    self.model.vacbrd.GetNumAvailPromotions(self.grade + 1 if self.grade < 15 else 0),
        #    self.retire_preference)