# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:31:16 2015

@author: edogerde
"""

"""RECONNAISSANCE IMAGE & RECONNAISSANCE ODEUR"""
# Histogram score Reconnaissance Image , Reconnaissance Odeur

"""# histogram with mean_ and std
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

plt.show()"""

# Select the Reconnaissance and sort by age 
Reco=df[(df.Test=='reconnaissanceImage')]
Reco1= Reco.sort_values(by = 'Age')
df_Reco = Reco1.dropna(subset=['Result'])


#Linear regression between age and recognition for each item.           
Reco_Ima = df_Reco.loc[df_Reco['Item'].isin(['champignon','cassis','lavande','banane', 'fenouil','citron'])]

#Linear regression of the different items
sns.set(style="ticks")
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

# Mean and confident interval of the reconnaissance of items along age  
Items = ["champignon", "cassis", "lavande", "banane", "fenouil", "citron"] 
color = ["black", "red", "purple", "plum", "green", "yellow"]
#c = 0
for c, i in enumerate(Items): 
    
    Reco=df[(df.Test=='reconnaissanceImage') & (df.Item == i)]
    Reco1= Reco.sort_values(by = 'Age') 
    df_Reco = Reco1.dropna(subset=['Result'])
    
    sns.set(style="whitegrid")
    g = sns.PairGrid(df_Reco, y_vars="Result",
                     x_vars=["Age"],
                    size=5, aspect=.5)
                
    g.map(sns.pointplot, color=sns.xkcd_rgb[color[c]])
    #c= c + 1
    g.set(ylim=(0, 1))
    sns.despine(fig=g.fig, left=True)
    
# Mean and confident interval of the Reconnaissance of items alone and by groupAge    
Reco=df[(df.Test=='reconnaissanceOdeur')]
Reco1= Reco.sort_values(by = 'Age')
df_Reco = Reco1.dropna(subset=['Result'])
Reco_Ima = df_Reco.loc[df_Reco['Item'].isin(['champignon','cassis','lavande','banane', 'fenouil','citron'])]

# Mean and confident interval of the Reconnaissance of items  
sns.factorplot(x="Item", y="Result", data=Reco_Ima, kind= "bar") 

# Mean and confident interval of the Reconnaissance of items between GroupeAge   
sns.factorplot(x="Item", y="Result", hue = "GroupeAge", data=Reco_Ima, kind= "bar")

"""

# Spearman Corr between  Reconnaissance Image, ReconnaissanceOdor
import scipy

x = df[["Test", "Item",'Result']][(df.Test=="reconnaissanceOdeur")]
x_i = x.loc[y['Item'].isin(['champignon','cassis','lavande','banane', 'fenouil','citron'])]
x1 = x_i.dropna(subset=['Result'])
X = x1["Result"]

y = df[["Test", "Item",'Result']][(df.Test== "reconnaissanceImage")]
y_i = y.loc[y['Item'].isin(['champignon','cassis','lavande','banane', 'fenouil','citron'])]
y1 = y_i.dropna(subset=['Result'])
Y= y1["Result"]

cor_spearman = scipy.stats.spearmanr(X, Y)

NE MARCHE PAS, AFFICHE 1
"""
    
# Score episodicite en fonction du type d'hedonicite(positive,negative) en P1 & P2
Phases = ["P1", "P2"]
for p in Phases:
    Episo=df[(df.Test=='totale')&(df.Phase== p )]
    Episo1= Episo.sort_values(by = 'Age')
    df_epido = Episo1.dropna(subset=['Result'])

    sns.factorplot(x="Age", y="Result", hue="TypeReconnaissanceImage", data=df_epido)


