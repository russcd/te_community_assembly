import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
 'font.size':9,'axes.labelsize':9.5,'xtick.labelsize':8,'ytick.labelsize':8})
d=pd.read_csv('arrival_corrected.tsv',sep='\t')
C={'arr':'#2E7D74','exp':'#B0453A','obs':'#1B3A5C','g':'#8A96A3','acc':'#C98A2B'}
def rs(y,x):
    X=np.column_stack([np.ones(len(x)),x]); return y-X@np.linalg.pinv(X.T@X)@X.T@y

# threshold-sweep results (from PGLS run)
thr=pd.DataFrame({'thr':[3,4,5,6,7],'b':[-.0112,-.0117,-.0146,-.0169,-.0183],
                  't':[-2.72,-2.99,-3.70,-4.17,-4.32],
                  'se':[.0112/2.72,.0117/2.99,.0146/3.70,.0169/4.17,.0183/4.32]})

fig=plt.figure(figsize=(11,3.6)); gs=GridSpec(1,3,wspace=.40,figure=fig)

# a: arrival count vs Gini, richness-residualized
axA=fig.add_subplot(gs[0,0])
rx=rs(d.l_arr.values,d.logS.values); ry=rs(d.gini.values,d.logS.values)
axA.scatter(rx,ry,s=7,c=C['obs'],alpha=.30,lw=0)
sl=np.polyfit(rx,ry,1); xx=np.linspace(rx.min(),rx.max(),20)
axA.plot(xx,np.polyval(sl,xx),c=C['arr'],lw=2.5)
axA.axhline(0,c=C['g'],lw=.6); axA.axvline(0,c=C['g'],lw=.6)
axA.set_xlabel(r'recent arrivals  $\log(n_{arr})$  (resid | $\log S$)')
axA.set_ylabel(r'Gini  (residual | $\log S$)')
axA.text(-0.20,1.05,'a',transform=axA.transAxes,fontsize=13,fontweight='bold')

# b: two modes — arrival (neg) vs expansion (null) coefficients on Gini and J, PGLS
axB=fig.add_subplot(gs[0,1])
# PGLS coefs at thr=5
eff=pd.DataFrame({'lab':['arrival\n(Gini)','expansion\n(Gini)','arrival\n(J)','expansion\n(J)'],
   'b':[-0.0146,-0.0015,0.0079,-0.0005],'se':[0.0039,0.0041,0.0027,0.0028],
   'c':[C['arr'],C['exp'],C['arr'],C['exp']]})
y=np.arange(len(eff))[::-1]
for i,(_,r) in enumerate(eff.iterrows()):
    axB.plot([r.b-1.96*r.se,r.b+1.96*r.se],[y[i]]*2,c=r.c,lw=2.4,solid_capstyle='round')
    axB.scatter(r.b,y[i],s=48,c=r.c,edgecolor='white',lw=1,zorder=3)
axB.axvline(0,c='k',lw=1,ls='--',alpha=.6)
axB.set_yticks(y); axB.set_yticklabels(eff.lab,fontsize=7.6)
axB.set_xlabel('PGLS coefficient'); axB.set_xlim(-0.024,0.016)
axB.text(-0.20,1.05,'b',transform=axB.transAxes,fontsize=13,fontweight='bold')

# c: threshold robustness
axC=fig.add_subplot(gs[0,2])
lo=(thr.b-1.96*thr.se).min(); 
axC.errorbar(thr.thr,thr.b,yerr=1.96*thr.se,fmt='o-',c=C['arr'],lw=2,ms=5,
             capsize=3,ecolor=C['arr'],mfc='white',mec=C['arr'],mew=1.5)
axC.axhline(0,c='k',lw=1,ls='--',alpha=.6)
axC.set_ylim(lo-0.004, 0.004)
axC.set_xlim(2.6,7.4); axC.set_xticks([3,4,5,6,7])
axC.set_xlabel('Kimura threshold for "recent" (%)'); axC.set_ylabel('arrival coefficient (PGLS)')
axC.text(-0.22,1.05,'c',transform=axC.transAxes,fontsize=13,fontweight='bold')

fig.savefig('arrival_result.png',dpi=220,bbox_inches='tight',facecolor='white')
print('saved')
