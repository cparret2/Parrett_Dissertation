"""
================================================================================
MODULE NAME: OPM_VacancyBoard

DESCRIPTION:
    Defines the Vacancy and VacancyBoard classes for tracking authorized
    positions, new hires, promotions, transfers, and vacancies across GS
    grades and locations in the workforce simulation.
    
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

import numpy as np
import pandas as pd
from random import sample

class Vacancy:
    def __init__(self,vtype,loc,ed,grd,sal,los,p_age=18):
        self.vtype = vtype
        self.loc=loc
        self.ed=ed
        self.grd=grd
        self.sal=sal
        self.los=los
        self.age=p_age
        self.ppgrd=f"GS-{self.grd:02}"

    def get_rec(self,p_age=18):
        return {
            'LOC': self.loc,  #location,
            'EDLVL':self.ed, #edlevel
            'GSEGRD':self.grd,
            'PPGRD':self.ppgrd,
            'SALARY':self.sal, #sallvl
            'LOS':self.los,  #loslvl
            'AGE':p_age   #age
        }
    def __str__(self):
        return f"{self.vtype}::Loc {self.loc}\tGrd {self.grd}\tEd {self.ed}\tSal {self.sal}\tLOS {self.los}\tAge {self.age}"
    
class VacancyBoard:
    def __init__(self,model, data):
        self.model = model
        self.recruit_rate =  {9: self.model.m_rates["recruitment_rate"], 11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
        self._authorized_gs = {int(g): round(self.model.grd_rates['gov_init_auth'][g]) for g in self.model.grd_rates['gov_init_auth'].keys()}
        self._carry_cap_gs  = {int(g): round(self._authorized_gs[g] * self.model.m_rates["carry_cap"]) for g in self._authorized_gs.keys()}
        self._transfers  = {9: [], 11: [], 12: [], 13: [], 14: [], 15: []}
        self._promotions = {9: [], 11: [], 12: [], 13: [], 14: [], 15: []}
        self._newhires   = {9: [], 11: [], 12: [], 13: [], 14: [], 15: []}
        self.location_sizes = {f"{str(k):.02}" : len(data.groupby("LOC").get_group(k)) for k in data.groupby("LOC").groups.keys()}
        self.vacancy_rate = {9: .0, 11: .0, 12: .0, 13: .0, 14: .0, 15: .0}
        self.loc_vacs = {}
    
    def GetTotalPopulation(self): return len(self.model.agents.select(lambda m: m.status in ['A','P','M']))
    def GetGradePopulation(self, grd): return len(self.model.agents.select(lambda m: m.status in ['A','P','M'] and m.grade == grd))
    def GetGSPop_Dict(self): return {grd: len(self.model.agents.select(lambda m: m.grade == grd)) for grd in self._authorized_gs.keys()}
        
    def GetTotalAuthorizedLvl(self): return sum(self._authorized_gs.values())
    def GetAuthorizedGradeLvl(self,grd): return self._authorized_gs[grd]
    def GetAuthGS_Dict(self): return self._authorized_gs
    
    def UpdateAuthGradeLvl(self,grd,newval): self._authorized_gs[grd] = newval
    
    def GetTotalCarryCap(self): return sum(self._carry_cap_gs.values())
    def GetCarryCapGradeLvl(self,grd): return self._carry_cap_gs[grd]
    def GetCarryCap_Dict(self): return self._carry_cap_gs
    
    def GetTotalVacancies(self): return max(0, self.GetTotalAuthorizedLvl() - self.GetTotalPopulation())
    def GetGradeVacancies(self,grd): return max(0, self.GetAuthorizedGradeLvl(grd) - self.GetGradePopulation(grd))
    
    def GetNumNewHires(self,grd): return len(self._newhires[grd])
    def GetNumPromotions(self,grd): return len(self._promotions[grd])
    def GetNumTransfers(self,grd): return len(self._transfers[grd])
    def GetTotalNumTransfers(self): return sum(self.GetNumTransfers(g) for g in self._transfers.keys())
    def GetVacancyRate(self): return self.vacancy_rate
    def GetVacRateByGrade(self,grd): return self.vacancy_rate[grd] if grd in self.vacancy_rate else 0
    def GetNumAvailPromotions(self,promo_grd):
        #print("CHECK MATH... should we compensate for transfers?")
        if promo_grd > 0:
            avail_promo= max(0, self.GetGradeVacancies(promo_grd) - self.GetNumTransfers(promo_grd) )
            return avail_promo
        else:
            return 0
    ######################################################################    
    def GetLocationProbs(self,N):
        locations = list(self.location_sizes.keys())
        sizes = np.array(list(self.location_sizes.values()), dtype=np.float64)
        # Compute probabilities based on size
        probabilities = sizes / sizes.sum()
        # Randomly choose N locations based on their probabilities
        chosen_locations = np.random.choice(locations, size=int(N), p=probabilities, replace=True)
        return list(chosen_locations)
    
    
    ######################################################################
    def UpdateVacancies(self,add_growth):
        for grd in add_growth.keys():
            if add_growth[grd] > 0:
                l_locs = self.GetLocationProbs(add_growth[grd])
                l_edlvl = list(np.random.randint(grd,22,add_growth[grd]))
                l_avgsal_val = self.model.grd_rates["gov_init_avg_salary"][grd]
                for vac in range(add_growth[grd]):    
                    self._add_newvacancy(grd, p_loc=l_locs[vac], p_edlevel=l_edlvl[vac], p_sal=l_avgsal_val)
                
    
    #calculate initial vacancies
    def InitializeTalentReqs(self):
        # These are all newhires
        num_vacs = pd.Series(pd.Series(self._authorized_gs)-pd.Series(self.model.grd_rates['gov_init_population']),dtype=int)
        num_vacs.index = num_vacs.index.astype(int)

        l_locs=[]
        vacs = []
        for g in num_vacs.keys():
            grd = int(g)
            l_locs = self.GetLocationProbs(num_vacs[grd])
            
            edlvl = list(np.random.randint(grd,22,int(num_vacs[grd])))
                        
            avgsal = self.model.grd_rates["gov_init_avg_salary"][grd]
                        
            #avglos = self.model.grd_rates["gov_avg_newhire_experience"][grd]
            
            for n in range(num_vacs[grd]):
                self._add_newvacancy(grd,p_loc=l_locs[n],p_edlevel=edlvl[n],p_sal=avgsal)

    ######################################################################            
    def CalcVacancyRates(self):
        for grd in self.vacancy_rate.keys():
            self.vacancy_rate[grd] = self.GetGradeVacancies(grd) / self.GetAuthorizedGradeLvl(grd)

        return self.vacancy_rate
    ######################################################################
    def AddVacancy(self,agt,action,promo_grade=0):
        if action == "F":
            self._add_newvacancy(agt.grade, agt.location, agt.edlevel, agt.salary)
        elif action == "M": 
            self._add_transfer(agt,agt.location)
        elif action == "P":
            self._add_promotion(agt,promo_grade)
            self._add_newvacancy(agt.grade, agt.location, agt.edlevel, agt.salary)
        elif action == "R": 
            self._add_newvacancy(agt.grade, agt.location, agt.edlevel, agt.salary)
        elif action == "X":
            self._add_newvacancy(agt.grade, agt.location, agt.edlevel, agt.salary)
            
    def _inc_loc(self,lloc):
        if not f"{lloc}" in self.loc_vacs: self.loc_vacs[f"{lloc}"]=0
        self.loc_vacs[f"{lloc}"] += 1
    
    def _dec_loc(self,lloc):
        if not f"{lloc}" in self.loc_vacs: self.loc_vacs[f"{lloc}"]=0
        self.loc_vacs[f"{lloc}"] -= 1
        
    def _dec_loc_vacs(self,lst):
        for vac in lst:
            self._dec_loc(vac)

    ######################################################################      
    def _add_transfer(self,agt,orig_loc):
        #Zero Sum... not a change
        self._transfers[agt.grade].append((orig_loc, agt))
        self._inc_loc(orig_loc)
        
    ######################################################################            
    def _add_newvacancy(self,p_grade,p_loc=None,p_edlevel=None,p_sal=None):
        #new 11MAY
        #DEBUG:
        a= self.GetGradeVacancies(p_grade)
        b= self.GetGradePopulation(p_grade)
        c= self.GetAuthorizedGradeLvl(p_grade)
        e = a + b < c
        if self.GetGradeVacancies(p_grade) + self.GetGradePopulation(p_grade) <= self.GetAuthorizedGradeLvl(p_grade):
            #Generate new 
            avglos = self.model.grd_rates["gov_avg_newhire_experience"][p_grade]
            new_los = self.model.agemodel.GetProbLOS(p_grade,self.model.noise,avglos)
            v = Vacancy("NEWHIRE", p_loc, p_edlevel, p_grade, p_sal, new_los[0])
            self._newhires[p_grade].append((p_loc,v))    
            self._inc_loc(p_loc)
        
        
        
    ######################################################################            
    def _add_promotion(self,agt,trfr_grade):
        #### DO I NEED TO DECREMENT THE VACANCIES? I Don't think so...
        self._promotions[trfr_grade].append((agt.location, agt))
        self._inc_loc(agt.location)
        
    
    ######################################################################            
    def ProcessNewHires(self,grd):
        #New Hires
        
        num_to_hire = round(self.GetNumNewHires(grd) * self.model.grd_rates["gov_hiring_rate"][grd])
        num_to_hire = round(min(len(self._newhires[grd]), num_to_hire) * self.model.m_rates['auth_fill_rate'])
        ret_vals = []
        if num_to_hire > 0:
            if num_to_hire < len(self._newhires[grd]):
                #ret_vals = np.random.choice(self._newhires[grd],num_to_hire)
                pool = self._newhires[grd]
                num_to_hire = min(num_to_hire, len(pool)) 
                ret_vals = sample(self._newhires[grd], num_to_hire)
            else:
                ret_vals = self._newhires[grd].copy()
        
        #remove those values in v1 from the board
        self._newhires[grd] = [item for item in self._newhires[grd] if item not in ret_vals]
        self._dec_loc_vacs(ret_vals)
        return ret_vals
    
    ######################################################################            
    def ProcessTransfers(self,grd):
        #New Hires
        num_to_transfer = min(self.GetNumTransfers(grd),self.GetGradeVacancies(grd))
        
        np.random.shuffle(self._transfers[grd])
        ret_vals = self._transfers[grd][(-1*num_to_transfer):]
        
        #remove those values in v1 from the board
        self._transfers[grd] = [item for item in self._transfers[grd] if item not in ret_vals]
        self._dec_loc_vacs(ret_vals)
        return ret_vals
    ######################################################################
    def ProcessAllPromotions(self): 
        return {g: self.ProcessPromotions(g) for g in self._promotions.keys()}
    ######################################################################        
    def ProcessPromotions(self,grd,max_allowed=1e7):
        #promotions
        np.random.shuffle(self._promotions[grd])
        num_ret = round(len(self._promotions[grd]) * self.model.m_rates['auth_fill_rate'])
        num_ret = min(max_allowed,num_ret)
        
        #Choose n values from the list and store in a separate variable
        ret_vals = self._promotions[grd][(-1*num_ret):]
        
        #remove those values in v1 from the board
        self._promotions[grd] = [item for item in self._promotions[grd] if item not in ret_vals]
        self._dec_loc_vacs(ret_vals)
        return ret_vals
        
    ######################################################################
    def ProcessRecruits(self,n_agts,grd=9):
        print("STOP")
        raise RuntimeError
        self.vacancies_gs[grd] -= n_agts
    
    ######################################################################    
    def __GetVacancies(self,grd):
        ret_vals=[]
        n = self.vacancies_gs[grd]
        # Compute initial values
        num_to_hire = round(n * self.model.grd_rates["gov_hiring_rate"][grd])
        ret_vals += self.GetNewHires(grd,num_to_hire)
        self.vacancies_gs[grd] -= len(ret_vals)
        return ret_vals
