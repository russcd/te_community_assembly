import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
 'font.size':8.5,'axes.labelsize':9.5,'xtick.labelsize':8,'ytick.labelsize':8})
m=pd.read_csv('p1_counts.tsv',sep='\t')
c=pd.read_csv('/mnt/user-data/uploads/TE_subfamily_counts_by_species.tsv',sep='\t').set_index('species')
def vec(s):
    s=str(s); return np.array([int(x) for x in s.split(',')],float) if s not in('nan','') else np.array([])
C={'obs':'#1B3A5C','acc':'#C98A2B','null':'#B0453A','g':'#7A8894'}
def rs(y,x):
    X=np.column_stack([np.ones(len(x)),x]); return y-X@np.linalg.pinv(X.T@X)@X.T@y
def panel(ax,l):
    ax.text(-0.16,1.07,l,transform=ax.transAxes,fontsize=12,fontweight='bold',va='top',ha='left')

fig=plt.figure(figsize=(10,7))
gs=GridSpec(2,6,figure=fig,hspace=.40,wspace=1.5,height_ratios=[1.15,1])
# top row: a spans cols 0-2, b spans cols 3-5  (equal halves)
axA=fig.add_subplot(gs[0,0:3]); panel(axA,'a')
axB=fig.add_subplot(gs[0,3:6]); panel(axB,'b')
# bottom row: c,d,e span two cols each
axC=fig.add_subplot(gs[1,0:2]); panel(axC,'c')
axD=fig.add_subplot(gs[1,2:4]); panel(axD,'d')
axE=fig.add_subplot(gs[1,4:6]); panel(axE,'e')

# a: Gini vs N
axA.scatter(m.N,m.gini,s=6,c=C['obs'],alpha=.30,lw=0)
q=pd.qcut(m.logN,10); gm=m.groupby(q,observed=True).agg(x=('N','median'),y=('gini','median'))
axA.plot(gm.x,gm.y,'o-',c=C['acc'],lw=2,ms=4)
axA.set_xscale('log'); axA.set_xlabel('total TE copies $N$'); axA.set_ylabel('Gini coefficient')
axA.text(.03,.94,r'$\rho=0.81$',transform=axA.transAxes,va='top',fontsize=9)

# b: rank-abundance fan
sub=m.sort_values('N'); idx=np.linspace(0,len(sub)-1,70).astype(int); cmap=plt.cm.viridis
lo,hi=np.log10(sub.N.min()),np.log10(sub.N.max())
for i in idx:
    sp=sub.iloc[i].species; x=np.sort(vec(c.loc[sp,'subfamily_counts']))[::-1]; x=x[x>0]
    if len(x)<5: continue
    axB.plot(np.arange(1,len(x)+1),x/x.sum(),c=cmap((np.log10(sub.iloc[i].N)-lo)/(hi-lo)),lw=.6,alpha=.7)
axB.set_xscale('log'); axB.set_yscale('log'); axB.set_xlabel('subfamily rank'); axB.set_ylabel('relative copy number')
sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(lo,hi)); sm.set_array([])
cb=fig.colorbar(sm,ax=axB,pad=.02,fraction=.045); cb.set_label(r'$\log_{10}N$',fontsize=8); cb.ax.tick_params(labelsize=7)

# c: rank-abundance slope vs N
w=m.dropna(subset=['ra_slope'])
axC.scatter(w.N,w.ra_slope,s=5,c=C['obs'],alpha=.30,lw=0)
q=pd.qcut(np.log(w.N),10); gm=w.groupby(q,observed=True).agg(x=('N','median'),y=('ra_slope','median'))
axC.plot(gm.x,gm.y,'o-',c=C['acc'],lw=2,ms=4)
axC.set_xscale('log'); axC.set_xlabel('total TE copies $N$'); axC.set_ylabel('rank-abundance slope')
axC.text(.04,.14,r'$\rho=-0.71$',transform=axC.transAxes,fontsize=8.5)

# d: partial
rx=rs(m.logN.values,m.logS.values); ry=rs(m.gini.values,m.logS.values)
axD.scatter(rx,ry,s=5,c=C['obs'],alpha=.30,lw=0)
sl=np.polyfit(rx,ry,1); xx=np.linspace(rx.min(),rx.max(),20)
axD.plot(xx,np.polyval(sl,xx),c=C['acc'],lw=2); axD.axhline(0,c=C['g'],lw=.6)
axD.set_xlabel(r'$\log N$  (residual $|\,\log S$)'); axD.set_ylabel(r'Gini  (residual $|\,\log S$)')

# e: null
axE.scatter(m.S,m.dH_raw,s=5,c=C['obs'],alpha=.30,lw=0)
xs=np.linspace(np.log(2),np.log(m.S.max()),50); bb=np.polyfit(m.logS,m.dH_raw,1)
axE.plot(np.exp(xs),np.polyval(bb,xs),c=C['acc'],lw=2); axE.axhline(0,c=C['null'],lw=1.2,ls='--')
axE.set_xscale('log'); axE.set_xlabel('subfamily richness $S$'); axE.set_ylabel(r'$\Delta H$ (nats vs. null)')

fig.savefig('/mnt/user-data/outputs/TE_prediction1_counts.png',dpi=220,bbox_inches='tight',facecolor='white')
print('done')

