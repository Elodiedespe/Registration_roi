""" Create a appropriate pandas database"""
import numpy as np
import pandas as pd
dfInput = pd.read_csv('/media/edogerde/MY PASSPORT/results_38.csv')
dfInput
dfOutput = pd.DataFrame(columns=['Sujet', 'Age', 'GroupeAge', 'AgeRT', 'RT', 'Phase', 'Item', 'Test', 'Result'])
dfInput.columns[1]
dfInput.columns.shape
dfInput.columns[96]
dfOutput = pd.DataFrame(columns=['Sujet', 'Age', 'GroupeAge', 'AgeRT', 'RT', 'TypeRT', 'Phase', 'Item', 'Test', 'Result'])

compt = 0

for idx in dfInput.index:
    cur = dfInput.loc[idx]
    for j in range(1,96):
        colName = dfInput.columns[j]
        itemName = colName.split('_')[0]
        testName = colName.split('_')[1]
        phaseName = colName.split('_')[2]
        result = cur[colName]
        sujet = cur['sujet']
        age = cur['ageToday']
        groupeAge = cur['groupeAge']
        ageRT = cur['ageRT']
        rt = cur['presenceRT']
        typeRT = cur['typeRT']
        dfOutput.loc[compt] = [sujet, age, groupeAge, ageRT, rt, typeRT, phaseName, itemName, testName, result]
        compt += 1
dfOutput
import os
dfOutput.to_csv('/home/edogerde/Desktop/hippomuseDataBase.csv', index=False)
exdf = dfOutput[(dfOutput.GroupeAge == 'Grand') & (dfOutput.Phase == 'P1') & (dfOutput.Test == 'reconnaissanceOdeur')]



"""DESCRIPTIVES ANALYSIS AND PLOTS"""

#Import systems support
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy

df = pd.read_csv('/home/edogerde/Registration_roi/hippomuseDataBase.csv')

"""TOTAL EPISODICITY AND AGE AND RT during P1 & P2"""
#Select the test "episocite" and phase "1" 
Episo=df[(df.Test=='episodicite') & (df.Phase=='P1')]
Episo1= Episo.sort_values(by = 'Age')
df_Episo = Episo1.dropna(subset=['Result'])


#Select the test "episocite" and phase "2"
EpisoRT= df[(df.Test=='episodicite') & (df.Phase=='P2')]
EpisoRT2 = EpisoRT.dropna(subset=['Result'])
EpisoRT=EpisoRT["Age"].sort_values()
# Plot Episodicity score according to the RT  and the Age in P1 with mathplotlib
plt.plot(df_Episo.Age[df_Episo.RT == 'RT'],df_Episo.Result[df_Episo.RT == 'RT'],'o')
plt.plot(df_Episo.Age[df_Episo.RT == 'Pas RT'],df_Episo.Result[df_Episo.RT == 'Pas RT'], 'o')
plt.ylabel("Score épisodicité totale en phase 1")
plt.xlabel("Age")
plt.title("Score episodicité totale en P1 en fonction de l'âge et de la presence de RT")


# Plot Episodicity score according to the RT  and the Age in P1 with seaborn
sns.set(style="ticks", context="talk")
pal = sns.cubehelix_palette(4, 1.5, .75, light=.6, dark=.2)
g = sns.lmplot(x="Age", y="Result", hue="RT", data=EpisoRT2,
               palette=pal, size=7)
g.set_axis_labels("Age (annees)", "Result (Score d'episodictite)")

# Boxplot Episodicity score according to the RT  and AgeGroup in P1 with seaborn
ax = sns.boxplot(x="RT", y="Result", hue= "GroupeAge", data=EpisoRT2)
ax = sns.stripplot(x="RT", y="Result", data=EpisoRT2, size=4, jitter=True, edgecolor="gray")

# Plot Episodicity score according to the RT  and the Age in P2 with mathplotlib
plt.plot(EpisoRT2.Age[EpisoRT2.RT == 'RT'],EpisoRT2.Result[EpisoRT2.RT == 'RT'],'o')
plt.plot(EpisoRT2.Age[EpisoRT2.RT == 'Pas RT'],EpisoRT2.Result[EpisoRT2.RT == 'Pas RT'], 'o')
plt.ylabel("Score épisodicité totale en phase 2")
plt.xlabel("Age")
plt.title("Score episodicité totale en P2 en fonction de l'âge et de la presence de RT")

# Boxplot Episodicity score according to the RT  and AgeGroup in P2 with seaborn
ax = sns.boxplot(x="RT", y="Result", hue= "GroupeAge", data=EpisoRT2)
ax = sns.stripplot(x="RT", y="Result", data=EpisoRT2, size=4, jitter=True, edgecolor="gray")


""" HEDONICITY AND AGE"""

# select the Hedonicity score and Age
HedoAge=df[(df.Test=='hedonicite')]
df_hedo = HedoAge.dropna(subset=['Result'])

#Linear regression between age and hedonicity for all items.
sns.set(style="ticks", context="talk")
Item = ["champignon", "cassis", "lavande", "banane", "fenouil", "citron"]
g = sns.lmplot("Age", "Result", hue="Item", data=df_hedo,
               hue_order= Item, size=6)


#Linear regression between age and hedonicity for each item.
sns.set(style="ticks")
g= sns.lmplot(x="Age", y="Result", col="Item", hue="Item", data=df_hedo,
           col_wrap=2, ci=None, palette="muted", size=4,
           scatter_kws={"s": 50, "alpha": 1})


""" RECOGNITION IMAGE AND AGE"""
# select the Hedonicity score and Age
Reco_Im=df[(df.Test=='reconnaissanceImage')] 
Reco_Im = Reco_Im.dropna(subset=['Result'])

# Linear regression between age and recognition score for the different items
Reco_Ima = Reco_Im.loc[Reco_Im['Item'].isin(['champignon','cassis','lavande','banane', 'fenouil','citron'])]
sns.set(style="ticks")
g= sns.lmplot(x="Age", y="Result", col="Item", hue="Item", data=Reco_Ima,
           col_wrap=2, ci=None, palette="muted", size=4,
           scatter_kws={"s": 50, "alpha": 1})

# Linear regression between age and recognition score for the different items
Reco_Im=df[(df.Test=='reconnaissanceImage') & (df.Item =='totale')] 
Reco_Im = Reco_Im.dropna(subset=['Result'])
sns.set(style="ticks")
g= sns.lmplot(x="Age", y="Result", col="Item", hue="Item", data=Reco_Im,
           col_wrap=2, ci=None, palette="muted", size=4,
           scatter_kws={"s": 50, "alpha": 1})

# Histogram of the recognition score 

# Select recognition image 
Reco_Im=df[['Item', 'Test', 'Result']][(df.Test=='reconnaissanceImage')] 
Reco_Im = Reco_Im.dropna(subset=['Result'])
Reco_Ima = Reco_Im.loc[Reco_Im['Item'].isin(['champignon','cassis','lavande','banane', 'fenouil','citron'])]

#Create a new_dataframe with the mean and std 
Reco_Im=df[['Item', 'Test', 'Result']][(df.Test=='reconnaissanceImage')]

itemsAll = np.unique(Reco_Im.Item)
df_mean=pd.DataFrame(columns = ['Item', 'mean', 'std'])
df_mean.Item = itemsAll
df_mean['mean'] = [np.mean(Reco_Im.Result[Reco_Im.Item == item]) for item in itemsAll]
df_mean['std'] = [np.std(Reco_Im.Result[Reco_Im.Item == item]) for item in itemsAll]

# PLot a histogram of the mean of the recogniton image score of the different item
df_plot = df_mean[["Item","mean"]][df_mean['Item'].isin(['champignon','cassis','lavande','banane', 'fenouil','citron'])]
df_plot_rotation = df_plot.T
df_plot_rotation.columns = ["banane", "cassis ","champignon","citron","fenouil","lavande"]
df_plot_rotation2 = df_plot_rotation.drop("Item")
df_plot_rotation2.plot(kind='bar') 

# Recognition Image score and age
# Select Reconnaissance image test
Reco_Im=df[(df.Test=='reconnaissanceImage')] 
Reco_Im = Reco_Im.dropna(subset=['Result'])
Reco_Ima = Reco_Im.loc[Reco_Im['Item'].isin(['champignon','cassis','lavande','banane', 'fenouil','citron'])]
sns.set(style="ticks")

#Linear regression of the different items
g= sns.lmplot(x="Age", y="Result", col="Item", hue="Item", data=Reco_Ima,
           col_wrap=2, ci=None, palette="muted", size=4,
           scatter_kws={"s": 50, "alpha": 1})
plt.show()
#Linear regression of the all items
Reco_Im=df[(df.Test=='reconnaissanceImage') & (df.Item =='totale')] 
Reco_Im = Reco_Im.dropna(subset=['Result'])
sns.set(style="ticks")
g= sns.lmplot(x="Age", y="Result", col="Item", hue="Item", data=Reco_Im,
           col_wrap=2, ci=None, palette="muted", size=4,
           scatter_kws={"s": 50, "alpha": 1})
plt.show()

""" RECOGNITION IMAGE AND ODOR"""
# histogram with std
df_mean= pd.read_csv('/home/edogerde/Desktop/df_mean.csv')

fig = plt.figure()
ax = fig.add_subplot(111)

# data reco
N=6
Mean_item_recoIm=[0.944444,0.500000 ,0.222222,1.000000, 0.166667,0.888889]
std_item_recoIm=[0.235000,0.510000,0.427793,0.000000,0.383482,0.323381]
# data Reco
Mean_item_recoOd=[0.333333,0.500000,0.722222,0.500000, 0.388889,0.666667]
std_item_recoOd=[0.470000, 0.510000, 0.440000, 0.510000,0.487000, 0.471000]

ind=np.arange(N) 
width=0.35

## the bars
rects1 = ax.bar(ind, Mean_item_recoIm, width,
                color='blue',
                yerr=std_item_recoIm,
                error_kw=dict(elinewidth=2,ecolor='red'))

rects2 = ax.bar(ind+width, Mean_item_recoOd, width,
                    color='red',
                    yerr= std_item_recoOd,
                    error_kw=dict(elinewidth=2,ecolor='blue'))

# axes and labels
ax.set_xlim(-width,len(ind)+width)
ax.set_ylim(0,1)
ax.set_ylabel('Scores')
ax.set_title('Scores de Reconnaisance Image et Reconnaissance Odeur en fonction"
              "des items")

xTickMarks = ['champignon','cassis','lavande','banane', 'fenouil','citron']
ax.set_xticks(ind+width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=45, fontsize=10)

## add a legend
ax.legend( (rects1[0], rects2[0]), ('Reconnaissance Image', 'Reconnaissance Odeur') )

plt.show()

"""MUSIQUE ET AGE """

# Plot separately musique according to Age within P1 & P2
list_item = ["P1", "P2"]
for i in list_item:
    Musi=df[(df.Test=='musique') & (df.Phase == i) & (df.Item == "totale")]
    Musi_Age= Musi.dropna(subset=["Result"])
    sns.set(style="darkgrid", color_codes=True)
    sns.jointplot("Age", "Result",data=Musi_Age, kind="reg",
               xlim=(0, 13), ylim=(-0.5, 6), color="r", size=7)

# Plot musique according to Age in P1 and P2 
sns.set(style="ticks")

Musi1=df[(df.Test=='musique')] & (df.Item == "totale")]

Musi= Musi2.dropna(subset=["Result"])

sns.lmplot(x="Age", y="Result", col="Phase", hue="Phase", data=Musi_Age,
           col_wrap=2, ci=None, palette="muted", size=4,
           scatter_kws={"s": 50, "alpha": 1})


# Plot the different item_music according to P1 and P2
list_item = ['champignon','cassis','lavande','banane', 'fenouil','citron']

for i in list_item:
    
    Musi= df[(df.Test=='musique') & (df.Item== i) ] 
    Music= Musi.dropna(subset=['Result'])
    sns.set(style="whitegrid")

    g = sns.PairGrid(Music, y_vars="Result",
                 x_vars=['Phase'],
                 size=5, aspect=.5)

    g.map(sns.pointplot, color=sns.xkcd_rgb["blue"])
    g.set(ylim=(0, 1.5))
    sns.despine(fig=g.fig, left=True)

# PLot the different Tests score acoording to P1 and P2
Tests= ['musique', 'what']    
for test in Tests:
    df_Test= df[(df.Test== test)]    
    R = df_Test.loc[df_Test['Item'].isin(['champignon','cassis','lavande','banane', 'fenouil','citron'])]  
    sns.pointplot(x="Phase", y="Result", hue="Item", data=R)
    plt.ylabel('Score %s'%(test))
    plt.show()



"""HISTOGRAMS OF THE DEPENDANTS VARIABLES ("hedonicite", "reconnaissanceImage",
 "reconnaissanceOdeur")"""

Tests = ["hedonicite", "reconnaissanceImage", "reconnaissanceOdeur", "totale", "musique"]
for t in Tests:
Test_nan = df[['Item', 'Test', 'Result', 'Phase']][(df.Test== "musique") & (df.Phase=='P2')]
Test_nan1 = Test_nan.loc[Test_nan['Item'].isin(['champignon','cassis','lavande','banane', 'fenouil','citron'])]  
Test_sansNan = Test_nan1.dropna(subset=["Result"])
itemsAll = np.unique(Test_sansNan.Item)

df_mean=pd.DataFrame(columns = ['Item', 'mean', 'std'])
df_mean.Item = itemsAll
df_mean['mean'] = [np.mean(Test_sansNan.Result[Test_sansNan.Item == item]) for item in itemsAll]
df_mean['std'] = [np.std(Test_sansNan.Result[Test_sansNan.Item == item]) for item in itemsAll]

Mean_item_recoIm = []
std_item_recoIm= []
xTickMarks = []

for i in np.array(df_mean["Item"]):
    xTickMarks.append(i)
    
for m in np.array(df_mean['mean']):
        Mean_item_recoIm.append(m)
        
for s in np.array(df_mean['std']):
    std_item_recoIm.append(m)

# data reco
fig = plt.figure()
ax = fig.add_subplot(111)

# data reco
N=6

ind=np.arange(N) 
width=0.35

## the bars
rects1 = ax.bar(ind, Mean_item_recoIm, width,
                color='blue',
                yerr=std_item_recoIm,
                error_kw=dict(elinewidth=2,ecolor='red'))
    
# axes and labels
ax.set_xlim(-width,len(ind)+width)
ax.set_ylim(0,1)
ax.set_ylabel("Scores association musique odeur en recuperation")

ax.set_xticks(ind+width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=45, fontsize=10)



    
# Plot for the RT groupe the different Score test according to the Phase and the Age of RT     
import seaborn as sns

EpisoR=df[(df.Test=='what') & (df.RT=='RT') & (df.GroupeAge=='Moyen')]
EpisoRT= EpisoR.sort_values(by = 'Age')
EpisoRT = EpisoRT.dropna(subset=['Result'])

sns.set(style="whitegrid")

g = sns.PairGrid(EpisoRT, y_vars="Result",
                 x_vars=["AgeRT"],
                 size=5, aspect=.5)
g.map(sns.pointplot, color=sns.xkcd_rgb["blue"])
g.set(ylim=(0, 3))
sns.despine(fig=g.fig, left=True)


sns.pointplot(x="Phase", y="Result", hue="Item", data=EpisoRT)
plt.show()