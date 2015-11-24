# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:50:25 2015

@author: edogerde
"""

"""DESCRIPTIVES ANALYSIS AND PLOTS"""

#Import systems support
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy

df = pd.read_csv('/home/edogerde/Registration_roi/hippomuseDataBase.csv')


# Select Sc episodiciteTotale
Episo=df[(df.Test=='episodicite') & (df.Phase=='P1')]
Episo1= Episo.sort_values(by = 'Age')
df_Episo = Episo1.dropna(subset=['Result'])

# Select Totale episo for P1 et P2
Phases = ["P1", "P2"]
for p in Phases: 
    Episo=df[(df.Test=='totale') & (df.Phase== p)]
    Episo1= Episo.sort_values(by = 'Age')
    df_Episo = Episo1.dropna(subset=['Result'])

#Anova 3 way Episodicite [ score on 3] , Phase, Age, RT
Episo=df[(df.Test=='totale')] 
Episo1= Episo.sort_values(by = 'Age')
df_Episo = Episo1.dropna(subset=['Result'])

sns.set(style="whitegrid")
g = sns.factorplot(x="Age", y="Result", hue="RT", col="Phase", data=df_Episo,
                   palette="YlGnBu_d", size=6, aspect=.75)
g.despine(left=True)

# Scatter Plot episodicite totale [score on 18] for P1  with mathplot
plt.plot(df_Episo.Age[df_Episo.RT == 'RT'],df_Episo.Result[df_Episo.RT == 'RT'],'o')
plt.plot(df_Episo.Age[df_Episo.RT == 'Pas RT'],df_Episo.Result[df_Episo.RT == 'Pas RT'], 'o')
plt.ylabel("Score episodicite totale en phase 1")
plt.xlabel("Age")
plt.title("Score episodicite totale en P1 en fonction de lage et de la presence de RT")

# Linear regression of episodicite totale [ score on 18] for P1  with seaborn
sns.set(style="ticks", context="talk")
pal = sns.cubehelix_palette(4, 1.5, .75, light=.6, dark=.2)
g = sns.lmplot(x="Age", y="Result", hue="RT", data=df_Episo,
               palette=pal, size=7)
g.set_axis_labels("Age (annees)", "Result (Score d'episodictite)")

# Linear regression for episodicite totale [ score on 3] for P1 and P2 according to RT
Phases = ["P1", "P2"]
for p in Phases: 
    Episo=df[(df.Test=='totale') & (df.Phase== p)]
    Episo1= Episo.sort_values(by = 'Age')
    df_Episo = Episo1.dropna(subset=['Result'])
    
    sns.set(style="ticks", context="talk")
    pal = sns.cubehelix_palette(4, 1.5, .75, light=.6, dark=.2)
    g = sns.lmplot(x="Age", y="Result", hue="RT", data=df_Episo,
                   palette=pal, size=7)
    g.set_axis_labels("Age (annees)", "Result (Score d'episodictite)")
