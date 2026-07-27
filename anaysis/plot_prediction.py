import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
 'font.size':8.5,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8})
m=pd.read_csv('p1_counts.tsv',sep='\t')
c=pd.read_csv('TE_subfamily_counts_by_species.tsv',sep='\t').set_index('species')
def vec(s):
    s=str(s); return np.array([int(x) for x in s.split(',')],float) if s not in('nan','') else np.array([])
C={'obs':'#1B3A5C','acc':'#C98A2B','null':'#B0453A','g':'#7A8894','sz':'#3C6E47'}
def rs(y,x):
    X=np.column_stack([np.ones(len(x)),x]); return y-X@np.linalg.pinv(X.T@X)@X.T@y
def panel(ax,l):
    ax.text(-0.20,1.06,l,transform=ax.transAxes,fontsize=12,fontweight='bold',va='top',ha='left')

fig=plt.figure(figsize=(10.5,6.6)); gs=fig.add_gridspec(2,3,hspace=.34,wspace=.40)

# A
ax=fig.add_subplot(gs[0,0]); panel(ax,'a')
ax.scatter(m.N,m.gini,s=5,c=C['obs'],alpha=.30,lw=0)
q=pd.qcut(m.logN,10); gm=m.groupby(q,observed=True).agg(x=('N','median'),y=('gini','median'))
ax.plot(gm.x,gm.y,'o-',c=C['acc'],lw=2,ms=4)
ax.set_xscale('log'); ax.set_xlabel('total TE copies $N$'); ax.set_ylabel('Gini coefficient')
ax.text(.04,.94,r'$\rho=0.81$',transform=ax.transAxes,va='top',fontsize=8.5)

# B rank-abundance curves
ax=fig.add_subplot(gs[0,1]); panel(ax,'b')
sub=m.sort_values('N'); idx=np.linspace(0,len(sub)-1,60).astype(int); cmap=plt.cm.viridis
lo,hi=np.log10(sub.N.min()),np.log10(sub.N.max())
for i in idx:
    sp=sub.iloc[i].species; x=np.sort(vec(c.loc[sp,'subfamily_counts']))[::-1]; x=x[x>0]
    if len(x)<5: continue
    ax.plot(np.arange(1,len(x)+1),x/x.sum(),c=cmap((np.log10(sub.iloc[i].N)-lo)/(hi-lo)),lw=.6,alpha=.7)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('subfamily rank'); ax.set_ylabel('relative copy number')
sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(lo,hi)); sm.set_array([])
cb=fig.colorbar(sm,ax=ax,pad=.02,fraction=.05); cb.set_label(r'$\log_{10}N$',fontsize=8); cb.ax.tick_params(labelsize=7)

# C rank-abundance slope vs N
ax=fig.add_subplot(gs[0,2]); panel(ax,'c')
w=m.dropna(subset=['ra_slope'])
ax.scatter(w.N,w.ra_slope,s=5,c=C['obs'],alpha=.30,lw=0)
q=pd.qcut(np.log(w.N),10); gm=w.groupby(q,observed=True).agg(x=('N','median'),y=('ra_slope','median'))
ax.plot(gm.x,gm.y,'o-',c=C['acc'],lw=2,ms=4)
ax.set_xscale('log'); ax.set_xlabel('total TE copies $N$'); ax.set_ylabel('rank-abundance slope')
ax.text(.04,.14,r'$\rho=-0.71$',transform=ax.transAxes,fontsize=8.5)

# D partial
ax=fig.add_subplot(gs[1,0]); panel(ax,'d')
rx=rs(m.logN.values,m.logS.values); ry=rs(m.gini.values,m.logS.values)
ax.scatter(rx,ry,s=5,c=C['obs'],alpha=.30,lw=0)
sl=np.polyfit(rx,ry,1); xx=np.linspace(rx.min(),rx.max(),20)
ax.plot(xx,np.polyval(sl,xx),c=C['acc'],lw=2); ax.axhline(0,c=C['g'],lw=.6)
ax.set_xlabel(r'$\log N$  (residual $|\,\log S$)'); ax.set_ylabel(r'Gini  (residual $|\,\log S$)')

# E null
ax=fig.add_subplot(gs[1,1]); panel(ax,'e')
ax.scatter(m.S,m.dH_raw,s=5,c=C['obs'],alpha=.30,lw=0)
xs=np.linspace(np.log(2),np.log(m.S.max()),50); bb=np.polyfit(m.logS,m.dH_raw,1)
ax.plot(np.exp(xs),np.polyval(bb,xs),c=C['acc'],lw=2); ax.axhline(0,c=C['null'],lw=1.2,ls='--')
ax.set_xscale('log'); ax.set_xlabel('subfamily richness $S$'); ax.set_ylabel(r'$\Delta H$ (nats vs. null)')

# F genome size
ax=fig.add_subplot(gs[1,2]); panel(ax,'f')
rr=rs(m.gini.values,m.logS.values); lx=np.log(m.asm.values)
ax.scatter(m.asm,rr,s=5,c=C['sz'],alpha=.28,lw=0)
sl=np.polyfit(lx,rr,1); xx=np.linspace(lx.min(),lx.max(),20)
ax.plot(np.exp(xx),np.polyval(sl,xx),c=C['acc'],lw=2); ax.axhline(0,c=C['g'],lw=.6)
ax.set_xscale('log'); ax.set_xlabel('assembly length (bp)'); ax.set_ylabel(r'Gini  (residual $|\,\log S$)')

fig.savefig('TE_prediction1_counts.png',dpi=220,bbox_inches='tight',facecolor='white')
print('done')
