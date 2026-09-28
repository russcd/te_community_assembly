import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,
 'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':False})

T=100; tau=np.linspace(0,T,600)

def logistic_step(tau, t0, size, rate):
    """One growth burst: rises from 0 toward `size` as time passes t0 (moving to present)."""
    age = t0 - tau
    x = size/(1+np.exp(-rate*(age)))
    x[age<0]=0
    return x

def trajectory(tau, t_arrive, bursts, t_death=None, rate=0.3):
    """Family copy number as a sum of growth bursts.
    bursts: list of (time_before_present, added_size). First burst usually at t_arrive.
    Growth is cumulative, so the family steps up at each burst then plateaus."""
    x = np.zeros_like(tau)
    for (t0, size) in bursts:
        x = x + logistic_step(tau, t0, size, rate)
    if t_death is not None:
        x = np.where(tau<t_death, 0,
             np.where(tau<t_death+8, x*np.clip((tau-(t_death-8))/8,0,1), x))
    return x

def shade(color,factor):
    c=np.array(mcolors.to_rgb(color))
    return tuple(c+(1-c)*(factor-1)) if factor>=1 else tuple(c*factor)

def compute(fam):
    for f in fam:
        f['w']=gaussian_filter1d(
            trajectory(tau, f['t_arrive'], f['bursts'], f.get('t_death'), f.get('rate',0.3)), 3)
    return fam

def stack_order(fam):
    byid={f['id']:f for f in fam}
    children={}
    for f in fam: children.setdefault(f.get('parent'),[]).append(f['id'])
    for k in children: children[k].sort(key=lambda i:-byid[i]['t_arrive'])
    order=[]
    def visit(fid):
        order.append(fid)
        for c in children.get(fid,[]): visit(c)
    for r in sorted(children.get(None,[]),key=lambda i:-byid[i]['t_arrive']): visit(r)
    return order

def draw(ax,fam,title,xscale,label_intro=False):
    compute(fam); byid={f['id']:f for f in fam}
    order=stack_order(fam)
    W=np.zeros_like(tau)
    for f in fam: W+=f['w']
    running=(-W/2).copy(); y=tau
    for fid in order:
        s=byid[fid]; top=running+s['w']
        ax.fill_betweenx(y,running,top,color=s['color'],lw=0.3,edgecolor='white',alpha=0.96)
        if s.get('is_arrival'):
            ta=s['t_arrive']; idx=np.argmin(np.abs(tau-ta))
            stack_edge=(W/2)[idx]
            x_out=stack_edge+xscale*0.20; x_in=stack_edge
            ax.annotate('',xy=(x_in,ta),xytext=(x_out,ta),
                arrowprops=dict(arrowstyle='-|>',color=s['color'],lw=1.8,shrinkA=0,shrinkB=1))
        running=top
    if label_intro:
        ax.text(xscale*0.60, T*0.5, 'introductions', rotation=90,
                va='center', ha='center', fontsize=9, color='#555', style='italic')
    ax.set_xlim(-xscale*0.72,xscale*0.70); ax.set_ylim(0,T)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title,fontsize=11,pad=10)

# ---------- ARRIVAL-DOMINATED: many families, each grows in 1-2 modest bursts ----------
np.random.seed(7)
cols_even=plt.cm.Set2(np.linspace(0,1,8))
arrival_dominated=[]
arrivals=[88,76,66,56,46,34,24,12]
for i,ta in enumerate(arrivals):
    # 1 or 2 bursts, small total
    nb=np.random.choice([1,2])
    bursts=[(ta, np.random.uniform(1.5,3))]
    if nb==2: bursts.append((ta-np.random.uniform(15,30), np.random.uniform(3,6)))
    arrival_dominated.append(dict(id=f'a{i}',parent=None,t_arrive=ta,
        bursts=bursts,color=cols_even[i%8],is_arrival=True,rate=0.35))
# a couple that go extinct
arrival_dominated.append(dict(id='ax1',parent=None,t_arrive=82,bursts=[(82,5)],
    color=cols_even[3],is_arrival=True,t_death=48,rate=0.35))
arrival_dominated.append(dict(id='ax2',parent=None,t_arrive=60,bursts=[(60,4)],
    color=cols_even[6],is_arrival=True,t_death=28,rate=0.35))

# ---------- EXPANSION-DOMINATED: dominant family grows in several big bursts ----------
base=plt.cm.Dark2(0.0)
cols_uneven=plt.cm.Dark2(np.linspace(0,1,6))
expansion_dominated=[
    dict(id='D',parent=None,t_arrive=92,rate=0.3,is_arrival=False,
         bursts=[(92,4),(72,9),(50,20),(28,34)],color=base),        # slow start, escalating
    dict(id='d1',parent='D',t_arrive=60,rate=0.3,is_arrival=False,
         bursts=[(60,10),(38,10)],color=shade(base,1.4)),           # subfamily, 2 bursts
    dict(id='d2',parent='D',t_arrive=42,rate=0.3,is_arrival=False,
         bursts=[(42,9)],color=shade(base,0.68)),
    dict(id='d3',parent='d1',t_arrive=24,rate=0.3,is_arrival=False,
         bursts=[(24,7)],color=shade(base,1.7)),
    dict(id='R',parent=None,t_arrive=80,rate=0.3,is_arrival=False,
         bursts=[(80,9),(50,8)],color=cols_uneven[1]),
    dict(id='A',parent=None,t_arrive=40,rate=0.35,is_arrival=True,
         bursts=[(40,7)],color=cols_uneven[2]),
]

def total(fam):
    compute(fam); W=np.zeros_like(tau)
    for f in fam: W+=f['w']
    return W.max()
xscale=max(total(arrival_dominated),total(expansion_dominated))

fig,axes=plt.subplots(1,2,figsize=(9,5.6))
draw(axes[0],arrival_dominated,'Arrival-dominated\n(even community)',xscale,label_intro=True)
draw(axes[1],expansion_dominated,'Expansion-dominated\n(unequal community)',xscale,label_intro=True)
fig.text(0.5,0.015,'number of elements  (stream width)',ha='center',fontsize=9.5)
plt.tight_layout(rect=[0.03,0.03,1,1])
fig.savefig('/mnt/user-data/outputs/conceptual_muller.png',dpi=200,bbox_inches='tight',facecolor='white')
print('saved')
