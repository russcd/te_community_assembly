import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
 'font.size':9.5,'axes.labelsize':10,'xtick.labelsize':8.5,'ytick.labelsize':8.8})

d=pd.read_csv('pgls_clades.tsv',sep='\t').dropna(subset=['clade','gini','logN','logS'])
def rs(y,x):
    X=np.column_stack([np.ones(len(x)),x]); return y-X@np.linalg.pinv(X.T@X)@X.T@y
d['rx']=rs(d.logN.values,d.logS.values); d['ry']=rs(d.gini.values,d.logS.values)

order=['Mammalia','Aves','Herptiles','Fish','Insecta','OtherArthropoda',
       'Nematoda','OtherInvertebrates','Fungi','Viridiplantae','Protists']
lab={'OtherArthropoda':'Other arthropods','OtherInvertebrates':'Other invertebrates'}
pal={'Mammalia':'#8C5A3B','Aves':'#4A90A4','Herptiles':'#6BAA75','Fish':'#3E6DA6',
     'Insecta':'#C56B3E','OtherArthropoda':'#D9A24E','Nematoda':'#9C8AA5',
     'OtherInvertebrates':'#7BA8B0','Fungi':'#B0453A','Viridiplantae':'#4E8B4A','Protists':'#9B7FB5'}
navy='#1B3A5C'; gray='#9BA6B2'; gold='#C98A2B'

fig=plt.figure(figsize=(6.6,12.4))
gs=GridSpec(3,1,height_ratios=[1.15,1.25,0.62],hspace=0.34)

# ===== a: clade scatter =====
ax=fig.add_subplot(gs[0,0])
for c in order:
    s=d[d.clade==c]; ax.scatter(s.rx,s.ry,s=13,c=pal[c],alpha=.25,lw=0)
for c in order:
    s=d[d.clade==c]
    if len(s)>=15:
        b,a=np.polyfit(s.rx,s.ry,1)
        xr=np.linspace(s.rx.quantile(.03),s.rx.quantile(.97),20)
        ax.plot(xr,b*xr+a,c=pal[c],lw=1.5,alpha=.9,zorder=4)
ax.axhline(0,c='#bbb',lw=.6); ax.axvline(0,c='#bbb',lw=.6)
ax.set_xlabel(r'total TE copies  $\log N$   (residual $|\,\log S$)')
ax.set_ylabel(r'copy-number inequality, Gini'+'\n'+r'(residual $|\,\log S$)')
ax.text(-0.16,1.02,'a',transform=ax.transAxes,fontsize=13,fontweight='bold',va='top')
ax.text(.96,.05,r'overall PGLS $\beta_{\log N}=0.108$',transform=ax.transAxes,ha='right',
        fontsize=8.3,fontweight='bold')

# legend as small swatches under panel a
from matplotlib.lines import Line2D
handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=pal[c],markersize=6,
        label=lab.get(c,c)) for c in order]
ax.legend(handles=handles,frameon=False,fontsize=7,loc='upper left',ncol=2,
          handletextpad=.2,columnspacing=.9,labelspacing=.25)

# ===== b: per-clade slope forest =====
ax2=fig.add_subplot(gs[1,0])
rows=[]
for c in order:
    s=d[d.clade==c]; n=len(s)
    X=np.column_stack([np.ones(n),s.rx.values]); y=s.ry.values
    bh=np.linalg.lstsq(X,y,rcond=None)[0]; resid=y-X@bh; sig=(resid@resid)/(n-2)
    se=np.sqrt(sig*np.linalg.inv(X.T@X)[1,1]); rows.append((c,n,bh[1],se))
rows=rows[::-1]; yv=np.arange(len(rows))
ax2.axvspan(0.10768-0.008,0.10768+0.008,color='#444',alpha=.10,zorder=0)
for i,(c,n,bb,se) in enumerate(rows):
    ax2.plot([bb-1.96*se,bb+1.96*se],[yv[i]]*2,c=pal[c],lw=2.3,solid_capstyle='round',zorder=2)
    ax2.scatter(bb,yv[i],s=44,c=pal[c],edgecolor='white',lw=1,zorder=3)
ax2.axvline(0,c='k',lw=1,ls='--',alpha=.6)
ax2.set_yticks(yv); ax2.set_yticklabels([f'{lab.get(c,c)} (n={n})' for c,n,_,_ in rows])
ax2.set_ylim(-0.7,len(rows)-0.3); ax2.set_xlim(-0.07,0.30)
ax2.set_xlabel(r'within-clade abundance slope  $\beta_{\log N}$')
ax2.text(-0.16,1.02,'b',transform=ax2.transAxes,fontsize=13,fontweight='bold',va='top')

# ===== c: model-comparison forest =====
ax3=fig.add_subplot(gs[2,0])
mods=[("OLS (no phylogeny)",0.07016,0.002016,gray),
      (r"PGLS, $\lambda$-ML",0.10768,0.004106,navy),
      ("PGLS, equal branches",0.11541,0.004204,navy),
      ("Bayesian phylo. MM",0.11000,None,gold)]
bl,bh_=0.09936,0.11583; mods=mods[::-1]; ym=np.arange(len(mods))
pc=[m[1] for m in mods if m[3] in (navy,gold)]
ax3.axvspan(min(pc),max(pc),color=navy,alpha=.06,zorder=0)
for i,(lab_,est,se,col) in enumerate(mods):
    lo,hi=(bl,bh_) if se is None else (est-1.96*se,est+1.96*se)
    ax3.plot([lo,hi],[ym[i]]*2,c=col,lw=2.6,solid_capstyle='round',zorder=2)
    ax3.scatter(est,ym[i],s=58,c=col,edgecolor='white',lw=1.1,zorder=3)
    ax3.text(hi+0.004,ym[i],f'{est:.3f}',va='center',fontsize=7.8,color=col)
ax3.axvline(0,c='k',lw=1,ls='--',alpha=.55)
ax3.set_yticks(ym); ax3.set_yticklabels([m[0] for m in mods])
ax3.set_ylim(-0.7,len(mods)-0.3); ax3.set_xlim(-0.01,0.135)
ax3.set_xlabel(r'abundance slope $\beta_{\log N}$  across models')
ax3.text(-0.16,1.05,'c',transform=ax3.transAxes,fontsize=13,fontweight='bold',va='top')

fig.savefig('/mnt/user-data/outputs/phylo_stacked.png',dpi=230,bbox_inches='tight',facecolor='white')
print('saved stacked')
