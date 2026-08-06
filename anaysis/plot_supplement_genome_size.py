# Supplementary figure: the abundance-dominance relationship is driven by TE
# copy number, not genome size. Reads p1_counts.tsv (per-genome stats).
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
 'font.size':9,'axes.labelsize':10,'xtick.labelsize':8.5,'ytick.labelsize':8.5})

m=pd.read_csv('p1_counts.tsv',sep='\t').dropna(subset=['gini','asm'])
m['logN']=np.log(m.N); m['logS']=np.log(m.S); m['logA']=np.log(m.asm)

def rs(y,x):
    X=np.column_stack([np.ones(len(x)),x]); return y-X@np.linalg.pinv(X.T@X)@X.T@y

C={'obs':'#1B3A5C','acc':'#C98A2B','neg':'#B0453A','g':'#888888'}

fig,ax=plt.subplots(1,2,figsize=(9,4.2)); plt.subplots_adjust(wspace=.34)

# a: controlling richness only -> confounded POSITIVE relationship
ryA=rs(m.gini.values,m.logS.values)
ax[0].scatter(m.asm,ryA,s=6,c=C['obs'],alpha=.28,lw=0)
slA=np.polyfit(np.log(m.asm),ryA,1)
xx=np.linspace(np.log(m.asm.min()),np.log(m.asm.max()),20)
ax[0].plot(np.exp(xx),np.polyval(slA,xx),c=C['acc'],lw=2.4)
ax[0].axhline(0,c=C['g'],lw=.6); ax[0].set_xscale('log')
ax[0].set_xlabel('assembly length (bp)'); ax[0].set_ylabel(r'Gini  (residual | $\log S$)')
ax[0].text(-0.18,1.04,'a',transform=ax[0].transAxes,fontsize=13,fontweight='bold')
ax[0].text(.04,.95,'controlling richness only\npartial $\\rho=+0.46$',
           transform=ax[0].transAxes,va='top',fontsize=8.3)

# b: additionally controlling copy number -> REVERSAL
ryB=rs(m.gini.values,np.column_stack([m.logN,m.logS]))
rxB=rs(m.logA.values,np.column_stack([m.logN,m.logS]))
ax[1].scatter(rxB,ryB,s=6,c=C['obs'],alpha=.28,lw=0)
slB=np.polyfit(rxB,ryB,1); xx2=np.linspace(rxB.min(),rxB.max(),20)
ax[1].plot(xx2,np.polyval(slB,xx2),c=C['neg'],lw=2.4)
ax[1].axhline(0,c=C['g'],lw=.6); ax[1].axvline(0,c=C['g'],lw=.6)
ax[1].set_xlabel(r'genome size  (residual | $\log N,\log S$)')
ax[1].set_ylabel(r'Gini  (residual | $\log N,\log S$)')
ax[1].text(-0.18,1.04,'b',transform=ax[1].transAxes,fontsize=13,fontweight='bold')
ax[1].text(.04,.14,'also controlling copy number\npartial $\\rho=-0.45$',
           transform=ax[1].transAxes,va='bottom',fontsize=8.3,color=C['neg'])

fig.savefig('SuppFig_genomesize.png',dpi=220,bbox_inches='tight',facecolor='white')
print('saved SuppFig_genomesize.png')
