#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 19:00:10 202456

@author: badarinath

"""

import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
import math
import importlib
from itertools import product, combinations
import VIDTR_envs
from VIDTR_envs import GridEnv
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


#%%


sys.path.insert(0, r"C:\Users\innov\OneDrive\Desktop\IDTR-project\Code")

import IVIDTR_curr_modified_V_memoized_genetic_algo_debug as VIDTR_module
import markov_decision_processes as mdp_module
import disjoint_box_union
import constraint_conditions
import constraint_conditions as cc

#%%
                                                                                
importlib.reload(constraint_conditions)
importlib.reload(disjoint_box_union)
importlib.reload(VIDTR_module)
importlib.reload(mdp_module)
importlib.reload(VIDTR_envs)

from markov_decision_processes import MarkovDecisionProcess as MDP
from disjoint_box_union import DisjointBoxUnion as DBU
from IVIDTR_curr_modified_V_memoized_genetic_algo_debug import VIDTR


#%%

class VIDTR_grid:
    
    '''
    Build the algorithm environment for the VIDTR on a grid
    
    '''
    
    def __init__(self, dimensions, center, side_lengths, stepsizes, max_lengths,
                 max_complexity, goal, time_horizon, gamma, eta, rho,
                 max_conditions = np.inf, reward_coeff=1.0, friction = 1.0):
        
        '''
        Parameters:
        -----------------------------------------------------------------------
        dimensions : int
                     Dimension of the grid 
        
        center : np.array
                 Center of the grid
                 
        side_lengths : np.array
                       Side lengths of the grid
                       
        stepsizes : np.array
                    Stepsizes for the grid
        
        max_lengths : np.array 
                      Maximum lengths for the grid
        
        max_complexity : int
                         Maximum complexity for the tree 
        
        goal : np.array
               Location of the goal for the 2D grid problem
               
        time_horizon : int
                       Time horizon for the VIDTR problem
                       
        gamma : float
                Discount factor
        
        eta : float
              Splitting promotion constant    
        
        rho : float
              Condition promotion constant
        
        max_conditions : int
                         The maximum conditions we iterate over for the VIDTR
        
        reward_coeff : float
                       The scaling for the reward
        
        friction : float
                   The friction constant; higher values here implies greater penalization
                   for the VIDTR algorithm

        Stores:
        -----------------------------------------------------------------------
        envs : list[GridEnv]
               The 2D environments for the grid for the different timesteps
        
        VIDTR_MDP : markov_decision_processes
                    The Markov Decision Process represented in the algorithm
        
        algo : VIDTR_algo
               The algorithm representing VIDTR
        '''
        self.dimensions = dimensions
        self.center = center
        self.side_lengths = side_lengths
        self.stepsizes = stepsizes
        self.max_lengths = max_lengths
        self.max_complexity = max_complexity
        self.goal = goal
        self.time_horizon = time_horizon
        self.gamma = gamma
        self.eta = eta
        self.rho = rho
        self.friction = friction
        self.reward_coeff = reward_coeff
        self.max_conditions = max_conditions        
        
        self.env = GridEnv(dimensions, center, side_lengths, goal,
                           stepsizes = stepsizes, reward_coeff=reward_coeff,
                           friction = friction)
        
        self.transitions = [self.env.transition for t in range(time_horizon)]
        self.rewards = [self.env.reward for t in range(time_horizon)]
        
        self.actions = [self.env.actions for t in range(time_horizon)]          
        self.states = [self.env.state_space for t in range(time_horizon)]       
        
        self.VIDTR_MDP = MDP(dimensions, self.states, self.actions, time_horizon, gamma,
                             self.transitions, self.rewards)                    
        
        self.algo = VIDTR(self.VIDTR_MDP, max_lengths, eta, rho, max_complexity,
                          stepsizes, max_conditions = max_conditions)
        
    
    def generate_random_trajectories(self, N):
        '''
        Generate N trajectories from the VIDTR grid setup where we take a
        random action at each timestep and we choose a random initial state
        
        Returns:
        -----------------------------------------------------------------------
           obs_states : list[list]
                        N trajectories of the states observed
        
           obs_actions : list[list]
                         N trajectories of the actions observed
           
           obs_rewards : list[list]
                         N trajectories of rewards obtained                    
           
        '''
        
        obs_states = []
        obs_actions = []
        obs_rewards = []
        
        for traj_no in range(N):
            obs_states.append([])
            obs_actions.append([])
            obs_rewards.append([])
            s = np.squeeze(self.VIDTR_MDP.state_spaces[0].pick_random_point())  
            obs_states[-1].append(s)
            
            for t in range(self.time_horizon):
                
                a = random.sample(self.actions[t], 1)[0]
                r = self.rewards[t](s,a)
                
                s = self.env.move(s,a)
                obs_states[-1].append(s)
                obs_actions[-1].append(a)
                obs_rewards[-1].append(r)
                
            
        return obs_states, obs_actions, obs_rewards
            
    def run_single_experiment(self, T, eta, rho, N):
        """
        Run VIDTR_grid for a single hyperparameter combination.
        """
        etas = eta
        rhos = rho

        grid_class = VIDTR_grid(
            dimensions, center, side_lengths, stepsizes, max_lengths, max_complexity, goal,
            time_horizon, gamma, etas, rhos,
            max_conditions=max_conditions, reward_coeff=reward_coeff,
            friction=friction
        )

        obs_states, obs_actions, obs_rewards = grid_class.generate_random_trajectories(N)

        stored_DBUs, optimal_actions = grid_class.algo.compute_interpretable_policies(
            integration_method='trajectory_integrate',
            integral_percent=0.8, debug=False,
            obs_states=obs_states, mutation_noise=5.0, T=1000,
            initialization_limit=100, mutation_limit=100,
            crossover_limit=100, print_fitness=False,  # <-- disable spam in parallel
            fitness_adder=0.0, visualize=False
        )

        mean_error = np.mean(grid_class.algo.optimal_errors)

        return {
            "T": T,
            "eta": eta,
            "rho": rho,
            "N": N,
            "mean_error": mean_error
        }

    
#%%
'''
Tests GridEnv
'''

if __name__ == '__main__':
    
    dimensions = 2
    center = np.array([0, 0])
    side_lengths = np.array([6, 6])
    goal = np.array([-1, 0])
    time_horizon = 5
    gamma = 0.9
    max_lengths = [4 for t in range(time_horizon)]
    stepsizes = 1.0
    max_complexity = 2
    etas = 10.0
    rhos = 2.0
    # etas  = 0.05, rhos = 0.1, reward_coeff = 1.5 give decent values for the answer
    # etas = -10/12, rhos = 10.2, reward_coeff = 1.2 give decent vals for the answer
    #etas = 0.05
    #rhos = 0.1
    reward_coeff = 5.0
    friction = 2.0
    
    max_conditions = np.inf
    

    grid_class = VIDTR_grid(dimensions, center, side_lengths,
                            stepsizes, max_lengths, max_complexity, goal,
                            time_horizon, gamma, etas, rhos,
                            max_conditions = max_conditions, reward_coeff = reward_coeff,
                            friction = friction)
    
    
    #%%
    N = 10000
    obs_states, obs_actions, obs_rewards = grid_class.generate_random_trajectories(N)

    #%%
    '''
    Tests for maximum_over_actions
    '''
    t = 0   
    f = lambda s,a : np.sum((s-a)**2)
    max_f = grid_class.algo.maximum_over_actions(f, t)
    
    #function(s,a) := np.sum((s-a)**2)
    
    def test_action():
        
        s = np.array([1,0])
        assert max_f(s) == np.array([-1,0])
        
    
    def test_action2():
        
        s = np.array([0,1])
        assert max_f(s) == np.array([0,-1])
        
        
    def test_action3():
        
        s = np.array([-1,0])
        assert max_f(s) == np.array([1,0])
    
    def test_action4():
        
        s = np.array([2,0])
        assert max_f(s) == np.array([-1,0])

#%%    
    f = lambda s,a : np.sum(s+a)
    max_f = grid_class.algo.maximum_over_actions(f, t)
    
    def test_f2action():
        
        s = np.array([1,0])
        assert max_f(s) == np.array([1,0])
        
    
    def test_f2action2():
        
        s = np.array([0,1])
        assert max_f(s) == np.array([0,1])
        
        
    def test_f2action3():
        
        s = np.array([-1,0])
        assert max_f(s) == np.array([-1,0])
    
    def test_f2action4():
        
        s = np.array([2,0])
        assert max_f(s) == np.array([1,0])
        
    #%%
    
    '''
    Tests for optimal_policies and values
    '''
    
    optimal_policies, optimal_value_funcs = grid_class.algo.compute_optimal_policies()
    s_vals = [np.array([0,0]), np.array([0,1]), np.array([-2,0])]
    
    # Optimal value function, optimal policy at t = 4 for state points [0,0], [0,1], [-2, 0]   
    # Check notebook for calculations regarding the same
    
    # Make blocks in a function so that we can do better testing and refactoring
    
    def test_last_timestep_optimal():
        
        val_func = optimal_value_funcs[time_horizon-1]
        optimal_policy = optimal_policies[time_horizon-1]
        
        assert val_func(np.array([0,0])) == 1
        assert optimal_policy(np.array([0,0])) == np.array([-1,0])
        
        assert val_func(np.array([0,1])) == 0.5
        assert optimal_policy(np.array([0,1])) == np.array([-1,0])
        
        assert val_func(np.array([-2,0])) == 1
        assert optimal_policy(np.array([1,0])) == np.array([1,0])
    
    # Optimal value function, optimal policy at t = 3 for state points [0,0], [0,1], [-2, 0]
    
    def rhs_compute(t, s, 
                    next_val_func, reward_func, gamma):
        
        rhs = -math.inf
        best_a = None
        
        for a in grid_class.actions[t]:
            
            val = reward_func(s, a) + grid_class.gamma * next_val_func(s + a)
            
            if val > rhs:
                rhs = val
                best_a = a
        
        return rhs, best_a
    
    def test_next_timesteps_optimal(s_vals = s_vals):
        
        
        for t in range(grid_class.time_horizon -2, -1, -1):
            
            val_func = optimal_value_funcs[t]
            optimal_policy = optimal_policies[t]
            
            next_val_func = optimal_value_funcs[t+1]
            reward_func = grid_class.rewards[t]
        
            for i, s in enumerate(s_vals):
                
                rhs, best_a = rhs_compute(t, s, next_val_func, reward_func,
                                          grid_class.gamma)
                
                assert val_func(s) == rhs
                assert optimal_policy(s) == best_a
        

    # Picture the optimal policies and value functions
    for t in range(time_horizon):
        grid_class.env.plot_policy_2D(optimal_policies[t], title = f'Actions at time {t}')
        
   #%%%
   
    '''
    Bellman_equation_I and int_value_functions tests
    '''
    
    t = grid_class.algo.MDP.time_horizon - 1
    last_step_bellman_I = grid_class.algo.bellman_equation_I(t)
    
    # s = [0,0], [0,1], [-2, 0] and a = [-1,0], [0,1]
    def test_last_step_int_bellman():
        
        s = np.array([0,0])
        a = np.array([0,1])
        assert last_step_bellman_I(s,a) == grid_class.algo.reward_functions[t](s,a)
        
        s = np.array([-1,0])
        a = np.array([1,0])
        assert last_step_bellman_I(s,a) == grid_class.algo.reward_functions[t](s,a)
        
        s = np.array([0,2])
        a = np.array([-1,0])
        assert last_step_bellman_I(s,a) == grid_class.algo.reward_functions[t](s,a)
    
    # Can you find appropriate values for the constants \eta and \rho, and for
    # the environmental parameters so that we get the right answer here?
    
    # This would consist of looking at the grid plots for each state-action tuple
    
   #%%%
   
    '''
    Copy the tests from the IVIDTR modified V case to be implemented here

    '''
                                                                        
    stored_DBUs, optimal_actions = grid_class.algo.compute_interpretable_policies(integration_method = 'static_integrate',
                                                                                  integral_percent = 0.8, debug = False,
                                                                                  obs_states=obs_states, mutation_noise = 5.0, T = 20,
                                                                                  initialization_limit = 100,
                                                                                  mutation_limit = 100,
                                                                                  crossover_limit = 100,
                                                                                  print_fitness = False,
                                                                                  fitness_adder = 0.0,
                                                                                  visualize=False) 
          
    #%%
    
    print(stored_DBUs)
    print(optimal_actions)
    
    #%%
    
    for t in range(time_horizon):
                                                                                
        print(f'Optimal DBUs at {t} is')                                  
        for i, c in enumerate(stored_DBUs[t]):
            print(f'{i}th DBU is')                                             
            print(c)
                                                                                                                     
        for i, a in enumerate(optimal_actions[t]):
            print(f'{i}th action is')
            print(a)                                         
                                                                             
    
    #%%%
                                                                                 
    '''                                                                        
    VIDTR - plot errors                                                        
    '''                                                                        
    grid_class.algo.plot_errors()                                              
                                                                                

   #%%
    '''
    VIDTR - get interpretable policy                                           
    '''
    for t in range(grid_class.time_horizon-1):                                   
        
        int_policy = VIDTR.get_interpretable_policy_dbus(stored_DBUs[t],
                                                         optimal_actions[t])
            
        grid_class.env.plot_policy_2D(int_policy, title=f'Int. policy at time {t}',
                                      saved_fig_name = f'genetic_algo_patched_vidtr_plots_{t}.png')


#%%%
'''

# Multiprocessing tyoe runs for thr GA parameters

T_vals = [200, 500, 1000, 2000, 4000]
eta_vals = [2, 5, 10, 15, 30]
rho_vals = [2, 5, 10, 15, 30]
N_values = [500, 1000, 2000, 5000, 10000]


dimensions = 2
center = np.array([0, 0])
side_lengths = np.array([6, 6])
goal = np.array([-1, 0])
time_horizon = 5
gamma = 0.9
max_lengths = [4 for t in range(time_horizon)]
stepsizes = 1.0
max_complexity = 2

reward_coeff = 5.0
friction = 2.0

max_conditions = np.inf

results = {}



#%%%

    #VIDTR - hyperparameter plots

    for i, T in enumerate(T_vals):
        for j, eta in enumerate(eta_vals):
            for k, rho in enumerate(rho_vals):
                for l, N in enumerate(N_values):
                    
                    etas = eta
                    rhos = rho
                    
                    grid_class = VIDTR_grid(dimensions, center, side_lengths,
                                            stepsizes, max_lengths, max_complexity, goal,
                                            time_horizon, gamma, etas, rhos,
                                            max_conditions = max_conditions, reward_coeff = reward_coeff,
                                            friction = friction)
                    
                    obs_states, obs_actions, obs_rewards = grid_class.generate_random_trajectories(N)
    
                    stored_DBUs, optimal_actions = grid_class.algo.compute_interpretable_policies(integration_method = 'static_integrate',
                                                                                                  integral_percent = 0.8, debug = False,
                                                                                                  obs_states=obs_states, mutation_noise = 5.0, T = 1000,
                                                                                                  initialization_limit = 100,
                                                                                                  mutation_limit = 100,
                                                                                                  crossover_limit = 100,
                                                                                                  print_fitness = False,
                                                                                                  fitness_adder = 0.0,
                                                                                                  visualize=False) 
                    
                    mean_error = np.mean(grid_class.algo.optimal_errors)
                    
                    results.append({
                    "T": T,
                    "eta": eta,
                    "rho": rho,
                    "N": N,
                    "mean_error": mean_error
                    })
                    
                    for t in range(grid_class.time_horizon-1):                                   
                        
                        int_policy = VIDTR.get_interpretable_policy_dbus(stored_DBUs[t],
                                                                         optimal_actions[t])
                            
                        grid_class.env.plot_policy_2D(int_policy, title=f'Int. policy at time {t}',
                                                      saved_fig_name = f'genetic_algo_patched_vidtr_plots_time{t}_runs{T}_eta{eta}_rho{rho}_N{N}.png')
                    
                    
                    
    df_results = pd.DataFrame(results)
    print(df_results)
    
    # optionally save to CSV
    df_results.to_csv("vidtr_hyperparam_sweep.csv", index=False)

#%%%

#Parameter grid collect values and run multithreading 


# Collect all parameter tuples
param_grid = [(T, eta, rho, N) for T in T_vals for eta in eta_vals for rho in rho_vals for N in N_values]

results = []
with ProcessPoolExecutor(max_workers=8) as executor:  # adjust workers to your CPU
    futures = {executor.submit(grid_class.run_single_experiment, *GA_cluster_params): params for params in GA_cluster_params}

    for future in as_completed(futures):
        params = futures[future]
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            print(f"Experiment {params} failed with error: {e}")

df_results = pd.DataFrame(results)
print(df_results)

# Optionally save to CSV
df_results.to_csv("vidtr_hyperparam_sweep.csv", index=False)
'''