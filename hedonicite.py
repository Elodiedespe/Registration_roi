# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 11:40:54 2015

@author: edogerde
"""
#https://stanford.edu/~mwaskom/software/seaborn/tutorial/categorical.html

#HISTOGRAMS OF THE DEPENDANTS VARIABLES ("hedonicite", "reconnaissanceImage",
#reconnaissanceOdeur")

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

"""HEDONICITY"""
# select the Hedonicity score and Age
HedoAge=df[(df.Test=='hedonicite')]
df_hedo = HedoAge.dropna(subset=['Result'])

#Linear regression between age and hedonicity for each item.
sns.set(style="ticks")
g= sns.lmplot(x="Age", y="Result", col="Item", hue="Item", data=df_hedo,
           col_wrap=2, ci=None, palette="muted", size=4,
           scatter_kws={"s": 50, "alpha": 1})
           

# Mean and confident interval of the hedonicity of items along age  
Items = ["champignon", "cassis", "lavande", "banane", "fenouil", "citron"] 
color = ["black", "red", "purple", "plum", "green", "yellow"]
c = 0
for i in Items: 
    
    HedoAge=df[(df.Test=='hedonicite') & (df.Item == i)]
    HedoAge1= HedoAge.sort_values(by = 'Age') 
    df_hedo = HedoAge1.dropna(subset=['Result'])
    
    sns.set(style="whitegrid")
    g = sns.PairGrid(df_hedo, y_vars="Result",
                     x_vars=["Age"],
                    size=5, aspect=.5)
                
    g.map(sns.pointplot, color=sns.xkcd_rgb[color[c]])
    c= c + 1
    g.set(ylim=(0, 2))
    sns.despine(fig=g.fig, left=True)
    
# Mean and confident interval of the hedonicity of items    
sns.factorplot(x="Item", y="Result", data=df_hedo, kind= "bar") 

# Mean and confident interval of the hedonicity of items    
sns.factorplot(x="Item", y="Result", hue = "GroupeAge" data=df_hedo, kind= "bar")

# Spearman Corr between Hedonicity and Reconnaissance Image, ReconnaissanceOdor
tests = ["reconnaissanceOdeur","reconnaissanceImage"]
for t in tests:
    x = df[["Test", 'Result']][(df.Test=="hedonicite")]
    x1 = x.dropna(subset=['Result'])
    X = x1["Result"]
    y = df[["Test", "Item",'Result']][(df.Test== t)]
    y_i = y.loc[y['Item'].isin(['champignon','cassis','lavande','banane', 'fenouil','citron'])]
    y1 = y_i.dropna(subset=['Result'])
    Y= y1["Result"]
    cor_spearman = scipy.stats.spearmanr(X, Y)
    
    print("La correlation de spearman hedonicite %s " %(t) ) 
    print(cor_spearman)
    
# Score episodicite en fonction du type d'hedonicite(positive,negative) en P1 & P2
Phases = ["P1", "P2"]
for p in Phases:
    Episo=df[(df.Test=='totale')&(df.Phase== p )]
    df_epido = Episo.dropna(subset=['Result'])

    sns.factorplot(x="Age", y="Result", hue="TypeHedonicite", data=df_epido)

