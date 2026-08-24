# Within-superfamily abundance->dominance test (Approach 3) + figure.
# Shows the abundance-dominance relationship holds within each major TE class
# separately, ruling out element-type composition as the driver.
#
# Inputs (same directory):
#   kimura_subfam_summaries_tsv.gz   (species, Class/fam, subfam, Mean_Kimura, ...)
#   subfamily_blast_results_tsv.gz   (species, subfamily, orphan, count, young)
# Output:
#   within_superfamily.png  and printed table
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
 'font.size':9,'axes.labelsize':10,'xtick.labelsize':8.5,'ytick.labelsize':8.5})

# ---- load & merge ----
k=pd.read_csv('kimura_subfam_summaries_tsv.gz',sep='\t',on_bad_lines='skip')
bl=pd.read_csv('subfamily_blast_results_tsv.gz',sep='\t')
bl['subfam']=bl.subfamily.str.split('#').str[0]
m=bl.merge(k[['species','subfam','Class/fam']],on=['species','subfam'],how='left')
m=m[m['count']>0].copy()
m['super']=m['Class/fam'].astype(str).str.split('/').str[0].replace({'nan':'Unknown'})

def gini(x):
    x=np.sort(x[x>0]); n=len(x)
    return (2*np.arange(1,n+1)-n-1)@x/(n*x.sum()) if n>=2 else np.nan
def rs(y,x):
    X=np.column_stack([np.ones(len(x)),x]); return y-X@np.linalg.pinv(X.T@X)@X.T@y

MIN=5
classes=['LTR','LINE','DNA','SINE','RC']
labels={'LTR':'LTR retrotransposons','LINE':'LINEs','DNA':'DNA transposons',
        'SINE':'SINEs','RC':'Rolling-circle (Helitron)'}
res=[]; allpool=[]
for sup in classes:
    rows=[]
    for sp,g in m[m['super']==sup].groupby('species'):
        c=g['count'].values.astype(float); c=c[c>0]
        if len(c)<MIN: continue
        rows.append((sp,len(c),c.sum(),gini(c)))
    d=pd.DataFrame(rows,columns=['sp','S','N','gini']).dropna()
    if len(d)<30: continue
    lN=np.log(d.N.values); lS=np.log(d.S.values)
    pr=stats.spearmanr(rs(d.gini.values,lS), rs(lN,lS))
    X=np.column_stack([np.ones(len(d)),lN,lS]); b=np.linalg.lstsq(X,d.gini.values,rcond=None)[0]
    # SE of beta_logN
    resid=d.gini.values-X@b; sig=(resid@resid)/(len(d)-3)
    se=np.sqrt(sig*np.linalg.inv(X.T@X)[1,1])
    res.append((sup,len(d),pr[0],pr[1],b[1],se))
    d['super']=sup; allpool.append(d)
    print('%-5s n=%4d  rho=%+.3f p=%.1e  beta=%+.4f'%(sup,len(d),pr[0],pr[1],b[1]))

pooled=pd.concat(allpool); pooled['lN']=np.log(pooled.N); pooled['lS']=np.log(pooled.S)

# ---- figure: two panels ----
C={'pt':'#1B3A5C','fit':'#C98A2B','g':'#8A96A3'}
palette={'LTR':'#B0453A','LINE':'#C98A2B','DNA':'#2E7D74','SINE':'#4A6FA5','RC':'#8A6BA5'}
fig=plt.figure(figsize=(11,4.3))
gs=fig.add_gridspec(1,2,width_ratios=[1.25,1],wspace=.32)

# panel a: within-type partial-residual scatter, one colour per class
axA=fig.add_subplot(gs[0,0])
for sup in classes:
    dd=pooled[pooled.super==sup]
    if len(dd)<30: continue
    rx=rs(dd.lN.values,dd.lS.values); ry=rs(dd.gini.values,dd.lS.values)
    axA.scatter(rx,ry,s=6,c=palette[sup],alpha=.20,lw=0)
    b,a=np.polyfit(rx,ry,1); xr=np.linspace(np.quantile(rx,.02),np.quantile(rx,.98),20)
    axA.plot(xr,b*xr+a,c=palette[sup],lw=2.2,label=labels[sup])
axA.axhline(0,c=C['g'],lw=.6); axA.axvline(0,c=C['g'],lw=.6)
axA.set_xlabel(r'within-class abundance  $\log N$  (residual | $\log S$)')
axA.set_ylabel(r'within-class Gini  (residual | $\log S$)')
axA.legend(frameon=False,fontsize=7.6,loc='upper left')
axA.text(-0.13,1.04,'a',transform=axA.transAxes,fontsize=13,fontweight='bold')

# panel b: forest of within-class beta_logN
axB=fig.add_subplot(gs[0,1])
res_r=res[::-1]; y=np.arange(len(res_r))
for i,(sup,n,rho,p,b,se) in enumerate(res_r):
    axB.plot([b-1.96*se,b+1.96*se],[y[i]]*2,c=palette[sup],lw=2.6,solid_capstyle='round')
    axB.scatter(b,y[i],s=52,c=palette[sup],edgecolor='white',lw=1.1,zorder=3)
axB.axvline(0,c='k',lw=1,ls='--',alpha=.6)
axB.set_yticks(y); axB.set_yticklabels(['%s\n(n=%d)'%(labels[s],n) for s,n,_,_,_,_ in res_r],fontsize=7.6)
axB.set_xlabel(r'within-class abundance slope  $\beta_{\log N}$')
axB.set_xlim(0,axB.get_xlim()[1])
axB.text(-0.13,1.04,'b',transform=axB.transAxes,fontsize=13,fontweight='bold')

fig.savefig('within_superfamily.png',dpi=220,bbox_inches='tight',facecolor='white')
print('\nsaved within_superfamily.png')
