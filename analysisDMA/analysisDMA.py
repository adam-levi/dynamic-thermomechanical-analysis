#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 09:46:16 2022

@author: adam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from itertools import product
plt.style.use('fivethirtyeight')
import matplotlib.ticker as tck


from DMA_extract import data_to_df
from DMA_math import *

class analysisDMA:
    
    def __init__(self, filename):
        self.df = data_to_df(filename)
        self.exp_names = np.array([])
        self.exp_values = np.array([[]])
        self.exp_unit = np.array([])
        self.df_analysis = pd.DataFrame()
    
    def make_df(self):
        if not len(self.df_analysis.index) > 0:
            self.df_analysis = pd.DataFrame(self.exp_values, columns = self.exp_names)
        else:
            pass
        self.df_analysis_units = pd.DataFrame(self.exp_unit, columns = self.exp_names)
        

class strain(analysisDMA):
    
    def __init__(self, filename):
        super().__init__(filename)
        self.strain_analysis()
        self.make_df()
    
    #determines the linear viscoelastic regime (LVE) for a strain sweep experiment, LVE is defined as a 5% reduction in elastic plateu modulus
    def strain_analysis(self):
        LVE_mod =[[self.df['Storage Modulus (MPa)'][0:3].mean()*0.95]]
        df_LVE = self.df.iloc[(self.df['Storage Modulus (MPa)']-LVE_mod[0]).abs().argsort()[:4]].sort_index().reset_index()
        reg = LinearRegression().fit(df_LVE['Storage Modulus (MPa)'].to_numpy().reshape((-1, 1)),df_LVE['Strain (%)'].to_numpy())
        LVE = reg.predict(LVE_mod)
        self.exp_values = [[LVE[0]]]
        self.exp_names = ['LVE']
        self.exp_unit = ['%']

    def strain_plot(self):
        fig, ax = plt.subplots()
        plt.loglog(self.df['Strain (%)'], self.df["Storage Modulus (MPa)"])
        ax.set_xlabel("Strain (%)")
        ax.set_ylabel("Storage Modulus (MPa)")

class creep(analysisDMA):
    def __init__(self, filename):
        super().__init__(filename)
        self.stat = self.df[['Static Force (N)']].iloc[0] + 0.00001
        #stat is the minimum static force (plus a small margin for error), used for determining recovery regime
        self.df_analysis = pd.DataFrame()
        self.creep_analysis()
        #self.make_df()
    
    def creep_fn(self, df_cycle, index, perm_deform):
        #stat = self.df[['Static Force (N)']].iloc[0] + 0.00001
        #creep was orginally run assuming negative displacements so values are transformed to be negative
        neg = df_cycle[['Displacement (µm)']].values[df_cycle[["Decay Time (min)"]].idxmax()]
        if neg > 0:
            df_cycle.loc[:, 'Displacement (µm)'] *= -1

        
        #finding the first index when stress is applied (within 0.00001) and using that to determine the elastic strain
        stress = df_cycle["Stress (MPa)"][df_cycle[["Stress (MPa)"]].idxmax()].values.tolist()
        res = next(x for x, val in enumerate(df_cycle['Stress (MPa)'])
                  if val > stress[0] - 0.00001)
        elastic_strain = df_cycle.at[res + 1,'Displacement (µm)'] - perm_deform 
        #perm_deform is initialized as 0 but updated for subsequent creep experiments
        
        #displacement is negative so idxmin returns max absolute value of displacement
        #max displacement = creep strain + elastic strain + permenant deformation
        #permenant deformation is the remaining displacement at the end of the run
        creep_strain = df_cycle[['Displacement (µm)']].values[df_cycle[['Displacement (µm)']].idxmin()] - elastic_strain - perm_deform
        perm_deform = df_cycle[['Displacement (µm)']].values[df_cycle[["Time (min)"]].idxmax()]
        #creating a new dataframe that contains only data when the material is recovering after removal of stress
        #often times there are "outliers" created from the snapback of the material when stress is relieved that produces
        #lower values of displacement than is physically real, we want to remove those points from the rest of the analysis
        df_recovery = df_cycle[df_cycle['Static Force (N)']<self.stat[0]].reset_index().drop(range(0,index)).reset_index()
        df_recovery = df_recovery.drop('level_0', axis = 1)
        df_recovery, outliers = remove_outliers(df_recovery, 0,5)
        recovery_index = df_recovery.at[0, 'index']
        creep_recovery = df_cycle.at[recovery_index,'Displacement (µm)']-perm_deform
        elastic_recovery = df_cycle[['Displacement (µm)']].values[df_cycle[['Displacement (µm)']].idxmin()] - df_cycle.at[recovery_index,'Displacement (µm)']
        
        self.exp_names = ['Elastic Strain', 'Creep Strain' , 'Elastic Recovery', 'Creep Recovery', 'Permenant Deformation']
        self.exp_units = ['micron', 'micron','micron', 'micron', 'micron' ]
        self.exp_values = [elastic_strain*-1, creep_strain[0]*-1, elastic_recovery[0]*-1, creep_recovery[0]*-1, perm_deform[0]*-1]
    

        
        if len(self.df_analysis.index) > 0:
            self.df_analysis.loc[len(self.df_analysis.index)] = self.exp_values
        else:
            self.df_analysis = pd.DataFrame([self.exp_values], columns = self.exp_names)
     
        return perm_deform
    
    #some creep experiments have multiple cycles of creep that are delineated by a value of negative Time
    #this function breaks up each creep cycle and runs creep() on each cycle individually
    def creep_analysis(self):
        if -2 in self.df['Time (min)'].values:
            creep_1_index = self.df[self.df['Time (min)']==-2].index.values
            df_1 = self.df.iloc[0:creep_1_index[0]]
            df_2 = self.df.iloc[creep_1_index[0]:-1].reset_index().drop('index',axis=1)
            perm_deform = self.creep_fn(df_1, 2, 0)
            perm_deform = self.creep_fn(df_2, 3, perm_deform)
            if -3 in self.df['Time (min)'].values:
                creep_2_index = self.df[self.df['Time (min)']==-2].index.values
                df_3 = self.df.iloc[creep_1_index[0]:creep_2_index[0]].reset_index().drop('index',axis=1)
                perm_deform = self.creep_fn(df_3, stat, 3, perm_deform)
        else:
            perm_deform = self.creep_fn(self.df, 2, 0)
            
    def creep_plot(self):
        fig, ax = plt.subplots()
        plt.scatter(x = self.df["Time (min)"], y = self.df["Displacement (µm)"])
        ax2 = ax.twinx()
        ax.set_ylabel('Displacement (µm)')
        ax.set_xlabel('Time (min)')
        ax2.set_ylabel('Stress (MPa)', c = 'red')
        plt.scatter(x = self.df["Time (min)"], y = self.df["Stress (MPa)"], color ='red')
        ax2.grid(None)
        
        
    

class tempSweep(analysisDMA):
    
    def __init__(self, filename):
        super().__init__(filename)
        self.dfs = dict(tuple(self.df.groupby('Frequency (Hz)')))
        self.df_analysis = pd.DataFrame()
        self.temp_sweep_analysis()
        #self.make_df()
    
    #determining glass transition temperature by the temperature when both loss modulus and tan delta have local maxima in close proximity
    def temp_sweep_analysis(self):
        
        for key in self.dfs.keys():
        
            df_temp=self.dfs[key].reset_index()
            
            tan_delta = df_temp["Tan Delta"][3:]
            tan_max_arr, prom_tan = find_max(tan_delta, 0.01)
            
            loss_mod = df_temp["Loss Modulus (MPa)"][3:]
            loss_max_arr, prom_loss = find_max(loss_mod, 50000)
    
            tg_index=sorted(product(loss_max_arr, tan_max_arr), key=lambda t: abs(t[0]-t[1]))[0]
    
            tg_loss = df_temp["Temperature (°C)"].iloc[tg_index[0]]
            tg_tan = df_temp["Temperature (°C)"].iloc[tg_index[1]]
            
            self.exp_values = [key, tg_tan, tg_loss]
            self.exp_names = ['Frequency','Tg (Tan Delta)', 'Tg (Loss Modulus)']
            self.exp_units = ['Hz','°C', '°C']
            
            if len(self.df_analysis.index) > 0:
                self.df_analysis.loc[len(self.df_analysis.index)] = self.exp_values
            else:
                self.df_analysis = pd.DataFrame([self.exp_values], columns = self.exp_names)
    

    def temp_sweep_plot(self):
        
        for key in self.dfs.keys():
            passfig,ax =plt.subplots()
            plt.scatter(x=self.dfs[key]["Temperature (°C)"],y = self.dfs[key]["Tan Delta"], label = 'Tan Delta', color = 'blue')
            ax2 = ax.twinx()
            ax.set_ylabel('Tan Delta', c = 'blue')
            ax.set_xlabel('Temperature (°C)')
            ax2.set_ylabel('Modulus')
            plt.scatter(x= self.dfs[key]["Temperature (°C)"],y = self.dfs[key]["Storage Modulus (MPa)"], c = 'red',label='Storage Modulus')
            plt.scatter(x=self.dfs[key]["Temperature (°C)"],y = self.dfs[key]["Loss Modulus (MPa)"], c = 'green', label='Loss Modulus') 
            ax2.set_yscale('log')
            plt.legend()
            ax2.grid(None)
    
class stressRelax(analysisDMA):
    
    def __init__(self, filename):
        super().__init__(filename)
        self.stress_relax_analysis()
        self.make_df()
    
    def stress_relax_analysis(self):
        tau_stress = self.df[['Stress (MPa)']].values[self.df[['Stress (MPa)']].idxmax()]*(1/np.e)
        max_time = self.df[['Time (min)']].values[self.df[['Time (min)']].idxmax()]
        stress_end = self.df[['Stress (MPa)']].values[self.df[['Time (min)']].idxmax()]
        sr_total_perc = stress_end/self.df[['Stress (MPa)']].values[self.df[['Stress (MPa)']].idxmax()]
        self.exp_unit = [['Minutes','Minutes', '%']]
        self.exp_names = ['Charateristic Relaxation Time','Total Time', 'Total Stress Relaxed']
        
        if tau_stress < stress_end:
            self.exp_values = [[np.nan, max_time[0], 100-sr_total_perc[0]*100]]

        else:
            df_tau = self.df.iloc[(self.df['Stress (MPa)']-tau_stress[0]).abs().argsort()[:2]]
            f = interp1d(df_tau['Stress (MPa)'],df_tau['Time (min)'])
            tau = f(tau_stress)
            self.exp_values = [[tau[0], max_time[0], 100-sr_total_perc[0]*100]]

 
    def stress_relax_plot(self):
        self.df.plot.scatter(x = "Time (min)", y = "Stress (MPa)" )


