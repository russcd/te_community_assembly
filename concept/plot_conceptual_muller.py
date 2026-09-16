import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,
 'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':False})

T=100; tau=np.linspace(0,T,500)
def trajectory(tau,t_arrive,rate,cap,t_death=None):
    age=t_arrive-tau; x=cap/(1+np.exp(-rate*(age-6))); x[age<0]=0
    if t_death is not None:
        x=np.where(tau<t_death,0,np.where(tau<t_death+8,x*np.clip((tau-(t_death-8))/8,0,1),x))
    return x
def shade(color,factor):
    c=np.array(mcolors.to_rgb(color))
    return tuple(c+(1-c)*(factor-1)) if factor>=1 else tuple(c*factor)

def make(fam):
    for f in fam:
        f['w']=gaussian_filter1d(trajectory(tau,f['t_arrive'],f['rate'],f['cap'],f.get('t_death')),2.5)
    return {f['id']:f for f in fam}

def stack_order(fam):
    """Build a stacking order where children sit immediately next to their parent.
    Roots ordered by arrival (oldest first); each child inserted right after its parent."""
    byid={f['id']:f for f in fam}
    children={}
    for f in fam: children.setdefault(f.get('parent'),[]).append(f['id'])
    for k in children: children[k].sort(key=lambda i:-byid[i]['t_arrive'])
    order=[]
    def visit(fid):
        order.append(fid)
        for c in children.get(fid,[]): visit(c)
    roots=sorted(children.get(None,[]),key=lambda i:-byid[i]['t_arrive'])
    for r in roots: visit(r)
    return order

def draw(ax,fam,title,xscale):
    byid=make(fam)
    order=stack_order(fam)
    W=np.zeros_like(tau)
    for f in fam: W+=f['w']
    running=(-W/2).copy(); y=tau
    for fid in order:
        s=byid[fid]; top=running+s['w']
        ax.fill_betweenx(y,running,top,color=s['color'],lw=0.3,edgecolor='white',alpha=0.96)
        if s.get('is_arrival'):
            ta=s['t_arrive']; idx=np.argmin(np.abs(tau-ta))
            stack_edge=(W/2)[idx]; mid=(running[idx]+top[idx])/2
            if mid>=0: x_out=stack_edge+xscale*0.22; x_in=top[idx]
            else: x_out=-stack_edge-xscale*0.22; x_in=running[idx]
            ax.annotate('',xy=(x_in,ta),xytext=(x_out,ta),
                arrowprops=dict(arrowstyle='-|>',color=s['color'],lw=1.8,shrinkA=0,shrinkB=1))
        running=top
    ax.set_xlim(-xscale*0.75,xscale*0.75); ax.set_ylim(0,T)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title,fontsize=11,pad=10)

np.random.seed(3)
cols_even=plt.cm.Set2(np.linspace(0,1,8))
arrival_dominated=[]
for i,ta in enumerate([88,76,64,54,44,32,22,11]):
    arrival_dominated.append(dict(id=f'a{i}',parent=None,t_arrive=ta,rate=0.28,
        cap=np.random.uniform(9,17)*0.5,color=cols_even[i%8],is_arrival=True))
arrival_dominated.append(dict(id='ax1',parent=None,t_arrive=80,rate=0.3,cap=6,color=cols_even[3],is_arrival=True,t_death=45))
arrival_dominated.append(dict(id='ax2',parent=None,t_arrive=58,rate=0.3,cap=5,color=cols_even[6],is_arrival=True,t_death=25))

base=plt.cm.Dark2(0.0)
cols_uneven=plt.cm.Dark2(np.linspace(0,1,6))
expansion_dominated=[
    dict(id='D', parent=None,t_arrive=92,rate=0.12,cap=95,color=base,is_arrival=False),
    dict(id='d1',parent='D', t_arrive=60,rate=0.17,cap=20,color=shade(base,1.4),is_arrival=False),
    dict(id='d2',parent='D', t_arrive=42,rate=0.19,cap=13,color=shade(base,0.68),is_arrival=False),
    dict(id='d3',parent='d1',t_arrive=26,rate=0.22,cap=8, color=shade(base,1.7),is_arrival=False),
    dict(id='R', parent=None,t_arrive=80,rate=0.15,cap=15,color=cols_uneven[1],is_arrival=False),
    dict(id='A', parent=None,t_arrive=40,rate=0.2, cap=7, color=cols_uneven[2],is_arrival=True),
]

def total(fam):
    for f in fam:
        if 'w' not in f: f['w']=gaussian_filter1d(trajectory(tau,f['t_arrive'],f['rate'],f['cap'],f.get('t_death')),2.5)
    W=np.zeros_like(tau)
    for f in fam: W+=f['w']
    return W.max()
xscale=max(total(arrival_dominated),total(expansion_dominated))

fig,axes=plt.subplots(1,2,figsize=(9,5.6))
draw(axes[0],arrival_dominated,'Arrival-dominated\n(even community)',xscale)
draw(axes[1],expansion_dominated,'Expansion-dominated\n(unequal community)',xscale)
fig.text(0.5,0.015,'number of elements  (stream width)',ha='center',fontsize=9.5)
plt.tight_layout(rect=[0.03,0.03,1,1])
fig.savefig('/mnt/user-data/outputs/conceptual_muller.png',dpi=200,bbox_inches='tight',facecolor='white')
print('saved')
