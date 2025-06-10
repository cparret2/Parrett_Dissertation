import CONFIG

GLBL_AGT_STATUS = {"A": "Active", "R": "Retired", "F": "Fired", "M": "Moved", "P": "Promoted", 'X': "Exited"}
class OPM_Agent:
    def __init__(self,rec,p_age=18):
        '''
        Requires: REC (KWARGS):
            'LOC'              
            'AGE'           
            'EDLVL'           
            'GSEGRD'
            'PPGRD'   
            'SALARY'
            'LOS'
        '''
        self.occ = "2210"          # Occupation of the agent
        self.location =  f"{rec['LOC']:02}"                
        self.edlevel = rec['EDLVL']           
        self.grade = int(rec['GSEGRD'])
        if 'LOSLVL' in rec: self.loslvl = rec['LOSLVL']           
        if 'SALLVL' in rec: self.sallvl = rec['SALLVL']           
        self.ppgrd = rec['PPGRD']   
        self.salary = rec['SALARY']
        self.previous_salary= rec['SALARY']
        self.los = rec['LOS']
        if 'AGELVL' in rec: rec['AGELVL']
        self.age = p_age
        self.status = "A"
        #self.supv = (rec['SUPERVIS']=="2" or rec['SUPERVIS']==2)
        self.gstp = 1              # Current GS Equivalent step
        self.last_wgi = 0          # Last wage growth increment    
        self.timeinstatus = 1
        self._nodal_pos = -1
        
    def isActive(self): return self.status == 'A'
    def isRetired(self): return self.status == 'R'
    def isFired(self): return self.status == 'F'
    def isMoving(self): return self.status == 'M'
    def isPromoted(self): return self.status == 'P'
    def isExiting(self): return self.status == 'X'    
        
    def Active(self): self._update_status('A')
    def Retire(self): self._update_status('R')
    def Fire(self): self._update_status('F')
    def Move(self): self._update_status('M')
    def Promote(self): self._update_status('P',reset_t=False)
    def Exit(self): self._update_status('X')

    def _update_status(self,stat,reset_t=True): 
        self.status = stat
        if reset_t: self.timeinstatus = 1
    #Not used
    def _update_nodal_pos(self,n): self._nodal_pos = n
    def _get_nodal_pos(self):return self._nodal_pos
        
    def Report(self):
        # Generate a report of the agent's current attributes.
        return {
            'location': self.location,         # Location of the agent
            'edlevel': self.edlevel,           # Education level of the agent
            'grade': self.grade,               # Grade of the agent
            'gstp': self.gstp,                 # Current step in the salary scale
            'lst_wgi': self.last_wgi,
            'occ': self.occ,                   # Occupation of the agent
            'ppgrd': self.ppgrd,               # Previous grade of the agent
            'avgsal': self.salary,             # Average salary of the agent
            'avglos': self.los,             # Average length of service
            'age': self.age,                   # Age of the agent
            'status': self.status
        }    

