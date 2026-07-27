import pandas as pd, numpy as np
from scipy import stats
rng=np.random.default_rng(17)
c=pd.read_csv('TE_subfamily_counts_by_species.tsv',sep='\t').set_index('species')
dv=pd.read_csv('TE_diversity_stats__1_.tsv',sep='\t').set_index('species')
def vec(s):
    s=str(s); 
    return np.array([int(x) for x in s.split(',')],float) if s not in('nan','') else np.array([],float)

rows=[]
for sp in c.index:
    cnt=vec(c.loc[sp,'subfamily_counts'])
    cnt=cnt[cnt>0]
    if len(cnt)<2: continue
    S=len(cnt); N=cnt.sum(); p=cnt/N
    H=-(p*np.log(p)).sum(); J=H/np.log(S)
    # Gini on copy numbers
    x=np.sort(cnt); n=len(x); gini=(2*np.arange(1,n+1)-n-1)@x/(n*x.sum())
    # top-family dominance
    d1=x[-1]/N; d5=x[-5:].sum()/N if n>=5 else np.nan
    eff=np.exp(H)  # effective number of subfamilies
    rows.append((sp,S,N,H,J,gini,d1,d5,eff))
m=pd.DataFrame(rows,columns=['species','S','N','H','J','gini','dom1','dom5','effN']).set_index('species')
m['asm']=dv.assembly_length.reindex(m.index)
m['logN']=np.log(m.N); m['logS']=np.log(m.S); m['genus']=m.index.str.split('_').str[0]

# ---- broken-stick null in H, at matched RICHNESS S ----
ca={}
for S in np.sort(m.S.unique()):
    S=int(S); r=250 if S<800 else 100
    pp=rng.dirichlet(np.ones(S),size=r); Hn=-(pp*np.log(pp)).sum(1)
    gn=np.array([ (2*np.arange(1,S+1)-S-1)@np.sort(row)/(S*row.sum()) for row in pp[:80] ])
    ca[S]=(Hn.mean(),Hn.std(ddof=1),gn.mean(),gn.std(ddof=1))
m['Hnull']=m.S.map(lambda s:ca[int(s)][0]); m['Hnull_sd']=m.S.map(lambda s:ca[int(s)][1])
m['Gnull']=m.S.map(lambda s:ca[int(s)][2])
m['dH']=(m.H-m.Hnull)/m.Hnull_sd
m['dH_raw']=m.H-m.Hnull
m['dGini']=m.gini-m.Gnull

def ols(y,X,cl):
    X=np.column_stack([np.ones(len(y)),X]); y=np.asarray(y,float)
    Xi=np.linalg.pinv(X.T@X); b=Xi@X.T@y; r=y-X@b; M=np.zeros((X.shape[1],)*2)
    for g in pd.unique(cl):
        mm=(cl==g); s=X[mm.values].T@r[mm.values]; M+=np.outer(s,s)
    G=len(pd.unique(cl)); n,k=X.shape; V=Xi@(((G/(G-1))*((n-1)/(n-k)))*M)@Xi
    se=np.sqrt(np.diag(V)); t=b/se
    return b,se,t,2*(1-stats.t.cdf(abs(t),G-1)),1-(r**2).sum()/((y-y.mean())**2).sum()

print('n genomes with >=2 subfamilies:',len(m))
print('median copies/genome %d   median subfamilies %d'%(m.N.median(),m.S.median()))
print('\n=== ABUNDANCE form of Prediction 1 ===')
print('Does dominance rise with total copy number N?')
for resp in ['gini','dom1','J']:
    rho=stats.spearmanr(m[resp],m.N)[0]
    print('  Spearman(%s, N) = %+.3f'%(resp,rho))

print('\n=== dominance vs N, controlling richness S ===')
b,se,t,p,r2=ols(m.gini,np.column_stack([m.logN,m.logS]),m.genus)
print('  gini ~ logN + logS:  logN b=%+.3f t=%+.1f p=%.3g | logS b=%+.3f t=%+.1f'%(b[1],t[1],p[1],b[2],t[2]))

print('\n=== RICHNESS form (recomputed from real copy vectors, matched-richness null) ===')
b,se,t,p,r2=ols(m.dH_raw,m[['logS']].values,m.genus)
print('  dH_raw ~ logS:  b=%+.3f t=%+.1f p=%.3g'%(b[1],t[1],p[1]))
print('  mean dH_raw %+.3f nats | %.0f%% below null | median eff. div lost %.0f%%'%(
    m.dH_raw.mean(),100*(m.dH_raw<0).mean(),100*(1-np.exp(m.dH_raw)).median()))

print('\n=== + genome-size control ===')
b,se,t,p,r2=ols(m.dH_raw,np.column_stack([m.logS,np.log(m.asm)]),m.genus)
print('  dH_raw ~ logS + log(asm):  logS b=%+.3f t=%+.1f | log(asm) b=%+.3f t=%+.1f'%(b[1],t[1],b[2],t[2]))

# rank-abundance slope per genome (log-log tail) as a dominance shape measure
print('\n=== rank-abundance: does the distribution get steeper with N? ===')
def rank_slope(cnt):
    x=np.sort(cnt)[::-1]; r=np.arange(1,len(x)+1)
    m_=(x>0)&(r>0); 
    if m_.sum()<5: return np.nan
    return np.polyfit(np.log(r[m_]),np.log(x[m_]),1)[0]
m['ra_slope']=[rank_slope(vec(c.loc[sp,'subfamily_counts'])) for sp in m.index]
print('  Spearman(rank-abundance slope, N) = %+.3f'%stats.spearmanr(m.ra_slope,m.N,nan_policy='omit')[0])
print('  (more negative slope = steeper = more dominated)')
m.to_csv('p1_counts.tsv',sep='\t')
print('\nsaved')
