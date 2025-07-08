"""
================================================================================
MODULE NAME: OPM_SocialNet

DESCRIPTION:
    Defines the social network structure for the agent-based model using
    preferential attachment and influence metrics. Supports adding new
    agents and updating the network dynamically during simulation.
    
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

import networkx as nx
import numpy as np

class OPM_SocialNet:
    
    def __init__(self,model):
        self.model=model
        self.G = nx.Graph()
        
    ######################################################################            
    def Generate(self, m):
        """
        Generates a social network with preferential attachment, incorporating:
        - Influence from GSEGRD (grade), LOS (length of service)
        - Big Fish in Small Pond Effect (location size adjustment)
        """
        initial_clique_size = min(5, len(self.model.agents))  # Ensure connectivity
        initial_agents = self.model.agents[:initial_clique_size]
    
        
        for agent in self.model.agents:
            self.G.add_node(agent.unique_id,agent_id=agent.unique_id)
            agent._nodal_pos = agent.unique_id
            
        # Create an initial fully connected clique
        conn_exist_nodes = []
        for i in range(len(initial_agents)):
            conn_exist_nodes.append(initial_agents[i].unique_id)
            for j in range(i + 1, len(initial_agents)):
                self.G.add_edge(initial_agents[i].unique_id, initial_agents[j].unique_id)
    
        #existing_nodes = set(self.G.nodes())
        existing_nodes = set(conn_exist_nodes)
        '''# ORIGINAL
        for agent in self.model.agents[initial_clique_size:]:
                
            candidate_connections = set()
            
            # Preferential attachment within the same location
            location_peers = [a.unique_id for a in self.model.agents if a.location == agent.location]
            candidate_connections.update(location_peers)
            
            # Preferential attachment within the same grade
            grade_peers = [a.unique_id for a in self.model.agents if a.grade == agent.grade]
            candidate_connections.update(grade_peers)
            
            # Ensure only existing nodes are considered
            candidate_connections = candidate_connections.intersection(existing_nodes)
            
            # Influence-based preferential attachment
            influence_scores = np.array([a.influence for a in self.model.agents if a.unique_id in existing_nodes])
            influence_probabilities = influence_scores / influence_scores.sum()  # Normalize
            
            # Select nodes based on combined factors
            candidate_connections = list(candidate_connections)
            num_preferential = min(m, len(candidate_connections))  # Limit to available nodes
            chosen_nodes=[]
            if len(candidate_connections) >= num_preferential:
                chosen_nodes = np.random.choice(candidate_connections, num_preferential, replace=False)
            else:
                additional_nodes = np.random.choice(
                    (existing_nodes), m - len(candidate_connections), replace=False, p=influence_probabilities
                 )
                chosen_nodes = list(candidate_connections) + list(additional_nodes)

            # Add edges
            chosen_nodes = [node for node in chosen_nodes if node != agent.unique_id]
            for node in chosen_nodes:
                self.G.add_edge(agent.unique_id, node)
            
            existing_nodes.add(agent.unique_id)
        '''
        for agent in self.model.agents[initial_clique_size:]:
            self.G.add_node(agent.unique_id, agent_id=agent.unique_id)
            agent._nodal_pos = agent.unique_id
        
            candidate_connections = set()
        
            # Preferential attachment within the same location
            location_peers = [a.unique_id for a in self.model.agents if a.location == agent.location]
            candidate_connections.update(location_peers)
        
            # Preferential attachment within the same grade
            grade_peers = [a.unique_id for a in self.model.agents if a.grade == agent.grade]
            candidate_connections.update(grade_peers)
        
            # Ensure only existing nodes are considered
            candidate_connections = candidate_connections.intersection(existing_nodes)
        
            # Expand candidate set if too small
            if len(candidate_connections) < m:
                fallback_candidates = existing_nodes - candidate_connections - {agent.unique_id}
                influence_scores = np.array([
                    a.influence for a in self.model.agents if a.unique_id in fallback_candidates
                ])
                if influence_scores.sum() == 0 or len(fallback_candidates) == 0:
                    influence_probabilities = None  # Uniform sampling fallback
                else:
                    influence_probabilities = influence_scores / influence_scores.sum()
        
                extra_needed = m - len(candidate_connections)
                extra_candidates = np.random.choice(
                    list(fallback_candidates),
                    size=min(extra_needed, len(fallback_candidates)),
                    replace=False,
                    p=influence_probabilities if influence_probabilities is not None else None
                )
                candidate_connections.update(extra_candidates)
        
            # Final sampling
            candidate_connections = list(candidate_connections - {agent.unique_id})
            num_to_connect = min(m, len(candidate_connections))
            chosen_nodes = np.random.choice(candidate_connections, size=num_to_connect, replace=False)
        
            for node in chosen_nodes:
                self.G.add_edge(agent.unique_id, node)
        
            existing_nodes.add(agent.unique_id)
           
    ######################################################################                
    def AddNewAgtToNet(self, new_agent, m):
        """
        Adds a new agent to the network while maintaining its structure.
        """
        new_agent._compute_influence()
   
        self.G.add_node(new_agent.unique_id)
   
        # Find candidate connections
        existing_nodes = set(self.G.nodes())
        candidate_connections = set()
   
        # Preferential attachment within the same location
        location_peers = [a.unique_id for a in self.model.agents if a.location == new_agent.location]
        candidate_connections.update(location_peers)
   
        # Preferential attachment within the same grade
        grade_peers = [a.unique_id for a in self.model.agents if a.grade == new_agent.grade]
        candidate_connections.update(grade_peers)
   
        # Ensure only existing nodes are considered
        candidate_connections = candidate_connections.intersection(existing_nodes)
   
        # Influence-based preferential attachment
        influence_scores = np.array([a.influence for a in self.model.agents if a.unique_id in existing_nodes])
        influence_probabilities = influence_scores / influence_scores.sum() # Normalize
   
        # Select nodes based on combined factors
        candidate_connections = list(candidate_connections)
        num_preferential = min(m, len(candidate_connections)) # Limit to available nodes
   
        if len(candidate_connections) >= num_preferential:
            chosen_nodes = np.random.choice(candidate_connections, num_preferential, replace=False)
        else:
            additional_nodes = np.random.choice(
                list(existing_nodes), m - len(candidate_connections), replace=False, p=influence_probabilities
            )
            chosen_nodes = list(candidate_connections) + list(additional_nodes)
   
        # Add edges
        chosen_nodes = [node for node in chosen_nodes if node != new_agent.unique_id]
        for node in chosen_nodes:
            self.G.add_edge(new_agent.unique_id, node)

        self.G.nodes[new_agent.unique_id]["influence"] = new_agent.influence
        self.G.nodes[new_agent.unique_id]["loc"] = new_agent.location
        self.G.nodes[new_agent.unique_id]["grade"] = new_agent.grade         
        self.G.nodes[new_agent.unique_id]["los"] = new_agent.los
    
    def Rewire(self, r_prob=0.05, nlinks=1):
        # Periodically rewire or add random long-distance connections to simulate small-world behavior.
        agents_ids = list(self.G.nodes())
        for agent_id in agents_ids:
            if np.random.rand() < r_prob:
                possible_targets = [aid for aid in agents_ids if aid != agent_id and not self.G.has_edge(agent_id, aid)]
                if len(possible_targets) == 0:
                    continue
                chosen = np.random.choice(possible_targets, size=min(nlinks, len(possible_targets)), replace=False)
                for target in chosen:
                    self.G.add_edge(agent_id, target)
                
    def DumpGraphGML(self,outfile):
        out_G = self.G.copy()
        for agent in self.model.agents:    
            out_G.nodes[agent.unique_id]["influence"] = agent.influence
            out_G.nodes[agent.unique_id]["loc"] = agent.location
            out_G.nodes[agent.unique_id]["grade"] = agent.grade         
            out_G.nodes[agent.unique_id]["los"] = agent.los
        nx.write_gml(out_G,outfile)