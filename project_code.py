# Simulation Methods - Projet 2026
# Samy Daghastani & Alexis Hanna Gerguis
# ESILV A4 IF2, prof: Jiang Pu, TD: Vincent Lambert
#
#only np.random.uniform and np.random.randint allowed
# no randn, no normal, nothing else

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)


# NORMAL GENERATOR (exo 1)
#used everywhere in the project

def generate_normal(n):
    #ratio of uniforms + random sign for N(0,1)
    #Z = U2/U1 follows half-normal (cf exo 1)
    samples = []
    while len(samples) < n:
        u1 = np.random.uniform(0,1)
        u2 = np.random.uniform(0,1)
        
        if u1 > 0 and u1 <= np.exp(-0.25*(u2/u1)**2):
            z = u2/u1  #z >= 0
            #random sign
            sign = 2*np.random.randint(0,2) - 1
            samples.append(sign * z)
    return np.array(samples[:n])


# EXERCICE 1 : Ratio of Uniforms
print("="*60)
print("EXERCICE 1")
print("="*60)

def ratio_of_uniforms_raw(n):
    #without the sign
    samples = []
    total = 0
    while len(samples) < n:
        u1 = np.random.uniform(0,1)
        u2 = np.random.uniform(0,1)
        total += 1
        if u1 > 0 and u1 <= np.exp(-0.25*(u2/u1)**2):
            samples.append(u2/u1)
    return np.array(samples), n/total

np.random.seed(42)
Z_raw, acc = ratio_of_uniforms_raw(100000)

print(f"taux acceptation: {acc:.4f}")
print(f"moyenne Z: {np.mean(Z_raw):.4f} (theorique sqrt(2/pi) = {np.sqrt(2/np.pi):.4f})")
print(f"std Z: {np.std(Z_raw):.4f}")
print("=> Z suit |N(0,1)| (demi-normale)")

#check symmetrized version
np.random.seed(42)
Z_sym = generate_normal(100000)
print(f"\navec symetrisation: moy={np.mean(Z_sym):.4f}, std={np.std(Z_sym):.4f}")

#taux theorique
p_th = np.sqrt(np.pi/8)
print(f"\ntaux theorique: {p_th:.6f}")
print(f"taux empirique: {acc:.6f}")
print(f"nb moyen iterations: {1/p_th:.4f} (th), {1/acc:.4f} (emp)")


fig, axes = plt.subplots(2,2, figsize=(12,10))

ax = axes[0,0]
ax.hist(Z_raw, bins=80, density=True, alpha=0.6, color='steelblue', edgecolor='white', linewidth=0.3)
xp = np.linspace(0, 5, 200)
ax.plot(xp, np.sqrt(2/np.pi)*np.exp(-xp**2/2), 'r-', lw=2, label='demi-normale')
ax.set_xlabel('z'); ax.set_ylabel('densite')
ax.set_title('Z brut (avant symetrisation)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[0,1]
ax.hist(Z_sym, bins=80, density=True, alpha=0.6, color='steelblue', edgecolor='white', linewidth=0.3)
xr = np.linspace(-4,4,200)
ax.plot(xr, norm.pdf(xr), 'r-', lw=2, label='N(0,1)')
ax.set_xlabel('z'); ax.set_ylabel('densite')
ax.set_title('Z symetrise')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1,0]
sz = np.sort(Z_sym); nq = len(sz)
tq = norm.ppf((np.arange(1,nq+1)-0.5)/nq)
st = max(1, nq//1500)
ax.scatter(tq[::st], sz[::st], s=2, alpha=0.4, color='steelblue')
ax.plot([-4,4],[-4,4],'r--',lw=1.5)
ax.set_xlabel('quantiles th N(0,1)'); ax.set_ylabel('quantiles emp')
ax.set_title('QQ plot')
ax.grid(True, alpha=0.3)

ax = axes[1,1]
u1g = np.linspace(0.001,1,300); u2g = np.linspace(0,1,300)
U1g, U2g = np.meshgrid(u1g, u2g)
acc_reg = (U1g <= np.exp(-0.25*(U2g/U1g)**2))
ax.contourf(U1g, U2g, acc_reg.astype(float), levels=[0.5,1.5], colors=['lightblue'], alpha=0.5)
ax.contour(U1g, U2g, acc_reg.astype(float), levels=[0.5], colors=['navy'])
ax.set_xlabel('$U_1$'); ax.set_ylabel('$U_2$')
ax.set_title("region d'acceptation")
ax.text(0.5, 0.9, f'aire = {p_th:.4f}', transform=ax.transAxes, ha='center',
        fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.grid(True, alpha=0.3)

plt.suptitle('Exercice 1: Ratio of Uniforms', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ex1_ratio_uniforms.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n=> figure sauvegardee")


# EXERCICE 2 : Importance Sampling
print("\n" + "="*60)
print("EXERCICE 2")
print("="*60)

#BS params
S0 = 53.6
r_rate = 0.015   #1.5%/an
sigma = 0.235    #23.5%/sqrt(an)
T = 1.0
K = 650000.0     # oui c'est enorme

#d1 d2 closed form
d1 = (np.log(S0/K) + (r_rate + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
A = (np.log(K/S0) - (r_rate - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

print(f"\nparametres: S0={S0}, r={r_rate}, sigma={sigma}, K={K}")
print(f"d1={d1:.4f}, d2={d2:.4f}")
print(f"Phi(d1)={norm.cdf(d1):.2e}, Phi(d2)={norm.cdf(d2):.2e}")
print(f"A = {A:.4f}")
print("=> formule fermee donne 0/0, ca marche pas numeriquement")
# print("debug d1 d2")  # remove later

def importance_sampling_ratio(n_sim, mu_shift):
    #IS: shift N(0,1) to N(mu,1) then weight by LR
    Z_std = generate_normal(n_sim)
    Z_sh = Z_std + mu_shift  # N(mu,1)
    
    ST = S0 * np.exp((r_rate - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z_sh)
    
    #log likelihood ratio
    log_LR = -mu_shift*Z_sh + 0.5*mu_shift**2
    
    call_pay = np.maximum(ST - K, 0)
    digital_pay = (ST > K).astype(float)
    
    #trick: C = -max(log_LR) to avoid overflow
    #doesnt change anything mathematically
    C = -np.max(log_LR)
    w = np.exp(log_LR + C)
    
    num = np.mean(w * call_pay)
    den = np.mean(w * digital_pay)
    return num/den if den > 0 else np.nan

#run it
print("\n--- IS avec shift mu = A ---")
np.random.seed(42)
ratio_is_big = importance_sampling_ratio(500000, A)
print(f"ratio (500k): {ratio_is_big:.2f}")

#multiple runs for CI
np.random.seed(100)
ratios_is = []
for _ in range(30):
    rv = importance_sampling_ratio(50000, A)
    if not np.isnan(rv):
        ratios_is.append(rv)
ratios_is = np.array(ratios_is)

mean_is = np.mean(ratios_is)
std_is = np.std(ratios_is)
se_is = std_is/np.sqrt(len(ratios_is))
print(f"30 runs de 50k: {mean_is:.2f} +/- {std_is:.2f}")
print(f"IC 95%: [{mean_is - 1.96*se_is:.2f}, {mean_is + 1.96*se_is:.2f}]")

#realiste ?
print(f"\nratio ~ {mean_is:.0f} => faudrait 3800 digitales pour 1 call")
print(f"K={K} >> S0={S0}, option mega OTM, pas realiste du tt")


fig, axes = plt.subplots(1,3, figsize=(15,4.5))

ax = axes[0]
xr = np.linspace(A-5, A+5, 200)
ax.plot(xr, norm.pdf(xr, A, 1), 'orange', lw=2, label=f'N({A:.0f}, 1)')
ax.plot(xr, norm.pdf(xr, 0, 1), 'blue', lw=2, label='N(0,1)')  # invisible a cette echelle lol
ax.axvline(x=A, color='red', ls='--', lw=1.5, label=f'A={A:.1f}')
ax.set_xlabel('z'); ax.set_ylabel('densite')
ax.set_title('changement de mesure')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

#convergence plot
ax = axes[1]
np.random.seed(42)
nlist = [1000,2000,5000,10000,20000,50000,100000,200000]
mc = []; sc = []
for nc in nlist:
    rc = [importance_sampling_ratio(nc, A) for _ in range(15)]
    rc = [x for x in rc if not np.isnan(x)]
    mc.append(np.mean(rc) if rc else np.nan)
    sc.append(np.std(rc) if rc else np.nan)
ax.errorbar(nlist, mc, yerr=[1.96*s for s in sc], fmt='o-', capsize=3, color='steelblue')
ax.set_xscale('log'); ax.set_xlabel('n'); ax.set_ylabel('ratio')
ax.set_title('convergence IS'); ax.grid(True, alpha=0.3)

#sensitivity to shift
ax = axes[2]
np.random.seed(42)
mus = np.linspace(A-5, A+5, 12)
r_mu = [importance_sampling_ratio(50000, m) for m in mus]
ax.plot(mus, r_mu, 'o-', color='steelblue')
ax.axvline(x=A, color='red', ls='--', label=f'mu=A')
ax.set_xlabel('shift mu'); ax.set_ylabel('ratio')
ax.set_title('sensibilite au shift'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle('Exercice 2: Importance Sampling', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ex2_importance_sampling.png', dpi=150, bbox_inches='tight')
plt.close()
print("=> figure sauvegardee")


# EXERCICE 3 : Acceptance-Rejection
print("\n" + "="*60)
print("EXERCICE 3")
print("="*60)

# ratio = E[Phi(X) | X >= A]
print(f"\nA = {A:.4f}")
print("ratio = E[Phi(X) | X >= A]")

#Q2-Q5: see report for derivations

#Q6-7: optimal lambda
#lam^2 - A*lam - 1 = 0
lam_opt = (A + np.sqrt(A**2 + 4)) / 2
log_M = lam_opt**2/2 - lam_opt*A - np.log(lam_opt)
print(f"\nlambda* = {lam_opt:.4f}")
print(f"log M = {log_M:.2f} (donc M est minuscle, presque tout est accepte)")

# implementation
def gen_cond_normal_AR(n, A_val, lam):
    #AR for X | X > A
    logM = lam**2/2 - lam*A_val - np.log(lam)
    samples = []
    total = 0
    while len(samples) < n:
        u1 = np.random.uniform(0,1)
        Y = A_val - np.log(u1)/lam   # Y = A + Exp(lam)
        
        u2 = np.random.uniform(0,1)
        #log scale to avoid overflow
        log_acc = -Y**2/2 + lam*(Y-A_val) - np.log(lam) - logM
        total += 1
        if np.log(u2) <= log_acc:
            samples.append(Y)
    return np.array(samples), n/total

np.random.seed(42)
X_cond, acc_ar = gen_cond_normal_AR(200000, A, lam_opt)

#conditional payoff
Phi_vals = S0*np.exp((r_rate - 0.5*sigma**2)*T + sigma*np.sqrt(T)*X_cond) - K
ratio_ar = np.mean(Phi_vals)
se_ar = np.std(Phi_vals)/np.sqrt(len(Phi_vals))

print(f"\ntaux acceptation: {acc_ar:.4f}")
print(f"ratio AR: {ratio_ar:.2f} +/- {se_ar:.2f}")


fig, axes = plt.subplots(1,3, figsize=(15,4.5))

ax = axes[0]
ax.hist(X_cond[:50000], bins=80, density=True, alpha=0.6, color='coral', edgecolor='white', linewidth=0.3)
xr = np.linspace(A, A+1.5, 200)
g_vals = lam_opt*np.exp(-lam_opt*(xr-A))
ax.plot(xr, g_vals, 'b--', lw=1.5, label=f'proposal g')
#normalized for plotting
f_v = np.exp(-xr**2/2)
f_v = f_v / (np.sum(f_v)*(xr[1]-xr[0]))
ax.plot(xr, f_v, 'r-', lw=2, label='target f')
ax.set_xlabel('x'); ax.set_ylabel('densite')
ax.set_title('echantillons X|X>A')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(Phi_vals[:50000], bins=80, density=True, alpha=0.6, color='coral', edgecolor='white', linewidth=0.3)
ax.axvline(x=ratio_ar, color='red', ls='--', lw=2, label=f'moy={ratio_ar:.0f}')
ax.set_xlabel('Phi(X)'); ax.set_ylabel('densite')
ax.set_title('payoff conditionnel')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[2]
rm = np.cumsum(Phi_vals)/np.arange(1, len(Phi_vals)+1)
step = max(1,len(rm)//2000)
ax.plot(np.arange(1,len(rm)+1,step), rm[::step], color='coral', lw=1, alpha=0.7)
ax.axhline(y=ratio_ar, color='red', ls='--', lw=1.5, label=f'{ratio_ar:.0f}')
ax.set_xlabel('n'); ax.set_ylabel('moyenne cumulee')
ax.set_title('convergence'); ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('Exercice 3: Acceptance-Rejection', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ex3_acceptance_rejection.png', dpi=150, bbox_inches='tight')
plt.close()
print("=> figure sauvegardee")


# EXERCICE 4 : Quasi-Monte Carlo
print("\n" + "="*60)
print("EXERCICE 4")
print("="*60)

#Q1: b-ary increment (little endian)
def incr_b_ary(digits, b):
    #+1 in base b, carry propagation
    new = digits.copy()
    carry = 1
    i = 0
    while carry > 0 and i < len(new):
        new[i] += carry
        if new[i] >= b:
            new[i] = 0
            carry = 1
        else:
            carry = 0
        i += 1
    if carry > 0:
        new.append(1)
    return new

#quick test
print("\n--- Q1: test incrementation ---")
print(f"base 10: (9,9,2,1) -> {incr_b_ary([9,9,2,1], 10)}")  # doit donner [0,0,3,1]
print(f"base 2: (1,1,1) -> {incr_b_ary([1,1,1], 2)}")  # doit donner [0,0,0,1]


#Q2: VdC + Horner
def vdc_horner(k_max, b):
    #generate first k_max terms using Horner
    seq = np.zeros(k_max)
    digits = [0]
    for k in range(1, k_max):
        digits = incr_b_ary(digits, b)
        #horner eval
        val = 0.0
        for d in reversed(digits):
            val = (val + d)/b
        seq[k] = val
    return seq

print("\n--- Q2: Van der Corput ---")
vdc2 = vdc_horner(10, 2)
print(f"base 2: {[f'{v:.4f}' for v in vdc2]}")
#0, 1/2, 1/4, 3/4, 1/8, 5/8...


#Q3: QMC+IS
def qmc_is_ratio(n_pts, b, mu):
    #VdC + IS combined
    u = vdc_horner(n_pts, b)
    u = np.clip(u, 1e-10, 1-1e-10)
    z = norm.ppf(u)
    z_sh = z + mu
    
    ST = S0*np.exp((r_rate - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z_sh)
    log_LR = -mu*z_sh + 0.5*mu**2
    
    call_pay = np.maximum(ST - K, 0)
    dig_pay = (ST > K).astype(float)
    
    C = -np.max(log_LR)
    w = np.exp(log_LR + C)
    
    num = np.mean(w*call_pay)
    den = np.mean(w*dig_pay)
    return num/den if den > 0 else np.nan

print("\n--- Q3: QMC + IS ---")
for b in [2,3,5]:
    for n in [10000, 50000, 100000]:
        rv = qmc_is_ratio(n, b, A)
        print(f"  base {b}, n={n:6d}: ratio = {rv:.2f}" if not np.isnan(rv) else f"  base {b}, n={n:6d}: NaN")


#Q4: RQMC
def rqmc_is(n_pts, b, mu, n_rep=30):
    #cranley-patterson: shift + mod 1
    vdc = vdc_horner(n_pts, b)
    res = []
    for _ in range(n_rep):
        shift = np.random.uniform(0,1)
        u = (vdc + shift) % 1.0
        u = np.clip(u, 1e-10, 1-1e-10)
        z = norm.ppf(u)
        z_sh = z + mu
        
        ST = S0*np.exp((r_rate-0.5*sigma**2)*T + sigma*np.sqrt(T)*z_sh)
        log_LR = -mu*z_sh + 0.5*mu**2
        C = -np.max(log_LR)
        w = np.exp(log_LR + C)
        
        num = np.mean(w * np.maximum(ST-K, 0))
        den = np.mean(w * (ST > K).astype(float))
        if den > 0:
            res.append(num/den)
    
    if len(res) > 2:
        return np.mean(res), np.std(res)/np.sqrt(len(res))
    return np.nan, np.nan

print("\n--- Q4: RQMC ---")
np.random.seed(42)
for b in [2,3,5]:
    m,s = rqmc_is(100000, b, A, 30)
    print(f"  base {b}: {m:.2f} +/- {s:.2f}" if not np.isnan(m) else f"  base {b}: NaN")



fig, axes = plt.subplots(2,2, figsize=(12,10))

ax = axes[0,0]
for b, col in [(2,'blue'),(3,'green'),(5,'orange')]:
    v = vdc_horner(100, b)
    ax.plot(range(100), v, 'o-', ms=2, lw=0.8, color=col, alpha=0.7, label=f'base {b}')
ax.set_xlabel('k'); ax.set_ylabel('phi_k')
ax.set_title('sequences Van der Corput'); ax.legend(); ax.grid(True, alpha=0.3)

#MC vs QMC on simple 1D problem
ax = axes[0,1]
nv = [100, 500, 1000, 2000, 5000, 10000, 50000]
se_mc = []; se_qmc = []
for n in nv:
    mc_e = [np.mean(np.exp(-np.random.uniform(0,1,n)**2/2)) for _ in range(30)]
    se_mc.append(np.std(mc_e))
    vdc = vdc_horner(n, 2)
    qe = [np.mean(np.exp(-((vdc+np.random.uniform(0,1))%1.0)**2/2)) for _ in range(30)]
    se_qmc.append(np.std(qe))
ax.loglog(nv, se_mc, 'o-', color='steelblue', label='MC')
ax.loglog(nv, se_qmc, 's-', color='coral', label='QMC base 2')
ref = se_mc[0]*np.sqrt(nv[0])/np.sqrt(np.array(nv))
ax.loglog(nv, ref, 'k--', alpha=0.4, label='O(1/sqrt(n))')
ax.set_xlabel('n'); ax.set_ylabel('std empirique')
ax.set_title('MC vs QMC (1D)'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

#halton 2D scatter
ax = axes[1,0]
ns = 500
vx = vdc_horner(ns, 2); vy = vdc_horner(ns, 3)
ax.scatter(vx, vy, s=5, alpha=0.6, color='steelblue')
ax.set_xlabel('VdC base 2'); ax.set_ylabel('VdC base 3')
ax.set_title(f'Halton (2,3) - {ns} pts')
ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05)
ax.grid(True, alpha=0.3); ax.set_aspect('equal')

#random for comparison
ax = axes[1,1]
ax.scatter(np.random.uniform(0,1,ns), np.random.uniform(0,1,ns), s=5, alpha=0.6, color='gray')
ax.set_xlabel('U1'); ax.set_ylabel('U2')
ax.set_title(f'MC random - {ns} pts')
ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05)
ax.grid(True, alpha=0.3); ax.set_aspect('equal')

plt.suptitle('Exercice 4: Quasi-Monte Carlo', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ex4_quasi_monte_carlo.png', dpi=150, bbox_inches='tight')
plt.close()
print("=> figure sauvegardee")


# COMPARAISON DES 3 METHODES (exo 2-3-4)
print("\n" + "="*60)
print("COMPARAISON")
print("="*60)

np.random.seed(42)

#IS
is_runs = [importance_sampling_ratio(100000, A) for _ in range(50)]
is_runs = np.array([x for x in is_runs if not np.isnan(x)])

#AR
X_ar, _ = gen_cond_normal_AR(200000, A, lam_opt)
Phi_f = S0*np.exp((r_rate-0.5*sigma**2)*T + sigma*np.sqrt(T)*X_ar) - K
r_ar = np.mean(Phi_f); se_ar2 = np.std(Phi_f)/np.sqrt(len(Phi_f))

#QMC
m_q, se_q = rqmc_is(100000, 2, A, 50)

print(f"IS:     {np.mean(is_runs):.2f} +/- {np.std(is_runs)/np.sqrt(len(is_runs)):.2f}")
print(f"AR:     {r_ar:.2f} +/- {se_ar2:.2f}")
print(f"QMC+IS: {m_q:.2f} +/- {se_q:.2f}")

#comparison figure
fig, ax = plt.subplots(figsize=(8,5))
methods = ['IS', 'AR', 'QMC+IS']
means = [np.mean(is_runs), r_ar, m_q]
sems = [np.std(is_runs)/np.sqrt(len(is_runs)), se_ar2, se_q]
cols = ['steelblue', 'coral', 'green']
bars = ax.bar(methods, means, yerr=[1.96*s for s in sems], capsize=5, color=cols, alpha=0.7, edgecolor='black')
ax.set_ylabel('ratio estime'); ax.set_title('comparaison des 3 methodes')
ax.grid(True, alpha=0.3, axis='y')
for bar, m, s in zip(bars, means, sems):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.96*s+10, f'{m:.0f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('comparison_methods.png', dpi=150, bbox_inches='tight')
plt.close()
print("=> figure sauvegardee")


# EXERCICE 5 : Analyse de donnees
print("\n" + "="*60)
print("EXERCICE 5")
print("="*60)

#load data
data = np.loadtxt('data_simulation_methods.csv', delimiter=',')
p1 = data[:,0]; p2 = data[:,1]
n_data = len(p1)

#log returns
lr1 = np.log(p1[1:]/p1[:-1])
lr2 = np.log(p2[1:]/p2[:-1])

#Q1 stats
drift1 = np.mean(lr1)*252; drift2 = np.mean(lr2)*252
vol1 = np.std(lr1)*np.sqrt(252); vol2 = np.std(lr2)*np.sqrt(252)
corr_lr = np.corrcoef(lr1, lr2)[0,1]

print(f"\n--- Q1: stats ---")
print(f"actif 1: drift={drift1:.4f}, vol={vol1:.4f}")
print(f"actif 2: drift={drift2:.4f}, vol={vol2:.4f}")
print(f"corr log-returns: {corr_lr:.4f}")

#Q2 normality check
sk1 = np.mean(((lr1-np.mean(lr1))/np.std(lr1))**3)
ku1 = np.mean(((lr1-np.mean(lr1))/np.std(lr1))**4)
sk2 = np.mean(((lr2-np.mean(lr2))/np.std(lr2))**3)
ku2 = np.mean(((lr2-np.mean(lr2))/np.std(lr2))**4)
print(f"\n--- Q2: normalite ---")
print(f"actif 1: skew={sk1:.4f}, kurtosis={ku1:.4f}")
print(f"actif 2: skew={sk2:.4f}, kurtosis={ku2:.4f}")
#gaussian would be 0 and 3

#Q4 kendall
def kendall_tau(x, y):
    #O(n^2) but only 999 pts so its fine
    n = len(x); conc = 0; disc = 0
    for i in range(n):
        for j in range(i+1, n):
            prod = (x[i]-x[j])*(y[i]-y[j])
            if prod > 0: conc += 1
            elif prod < 0: disc += 1
    return (conc - disc)/(n*(n-1)/2)

print("\n--- Q4: Kendall + Clayton ---")
print("calcul du Kendall tau (c'est lent dsl)...")
tau_k = kendall_tau(lr1, lr2)
theta_cl = 2*tau_k/(1-tau_k) if tau_k > 0 else 0.01
print(f"Kendall tau = {tau_k:.4f}")
print(f"Clayton theta = {theta_cl:.4f}")

#Q5 simulate with Clayton
def sample_clayton(n, theta):
    #conditional inverse (from the CM)
    u1 = np.random.uniform(0,1,n)
    t = np.random.uniform(0,1,n)
    u2 = np.power(
        np.power(u1,-theta)*(np.power(t,-theta/(theta+1))-1)+1,
        -1.0/theta
    )
    return u1, u2

np.random.seed(42)
u1_cl, u2_cl = sample_clayton(1000, theta_cl)
z1s = norm.ppf(np.clip(u1_cl, 1e-10, 1-1e-10))
z2s = norm.ppf(np.clip(u2_cl, 1e-10, 1-1e-10))

#simulated log returns
lr1s = np.mean(lr1) + np.std(lr1)*z1s
lr2s = np.mean(lr2) + np.std(lr2)*z2s

#prix simules
p1s = np.zeros(1001); p2s = np.zeros(1001)
p1s[0] = p1[-1]; p2s[0] = p2[-1]
for i in range(1000):
    p1s[i+1] = p1s[i]*np.exp(lr1s[i])
    p2s[i+1] = p2s[i]*np.exp(lr2s[i])

#simulated stats
d1s = np.mean(lr1s)*252; d2s = np.mean(lr2s)*252
v1s = np.std(lr1s)*np.sqrt(252); v2s = np.std(lr2s)*np.sqrt(252)
cs = np.corrcoef(lr1s, lr2s)[0,1]

print(f"\n--- Q5: simulation ---")
print(f"{'':15s} {'original':>10s} {'simule':>10s}")
print(f"{'drift 1':15s} {drift1:10.4f} {d1s:10.4f}")
print(f"{'drift 2':15s} {drift2:10.4f} {d2s:10.4f}")
print(f"{'vol 1':15s} {vol1:10.4f} {v1s:10.4f}")
print(f"{'vol 2':15s} {vol2:10.4f} {v2s:10.4f}")
print(f"{'corr':15s} {corr_lr:10.4f} {cs:10.4f}")

#Q6 bootstrap
def spearman_rho(x, y):
    n = len(x)
    rx = np.zeros(n); ry = np.zeros(n)
    for i, idx in enumerate(np.argsort(x)): rx[idx] = i+1
    for i, idx in enumerate(np.argsort(y)): ry[idx] = i+1
    return np.corrcoef(rx, ry)[0,1]

print("\n--- Q6: bootstrap ---")
print("calcul Kendall sur simule...")
tau_sim = kendall_tau(lr1s, lr2s)
rho_orig = spearman_rho(lr1, lr2)
rho_sim = spearman_rho(lr1s, lr2s)

print(f"Kendall: orig={tau_k:.4f}, sim={tau_sim:.4f}")
print(f"Spearman: orig={rho_orig:.4f}, sim={rho_sim:.4f}")

#500 replications
n_boot = 500
def boot(x, y, nb):
    n = len(x); taus = []; rhos = []
    for _ in range(nb):
        idx = np.random.randint(0,n,n)
        rhos.append(spearman_rho(x[idx], y[idx]))
        #kendall too slow on 999, subsample to 200
        if n > 200:
            sub = np.random.choice(n, 200, replace=False)
            taus.append(kendall_tau(x[idx][sub], y[idx][sub]))
        else:
            taus.append(kendall_tau(x[idx], y[idx]))
    return np.array(taus), np.array(rhos)

np.random.seed(42)
print("bootstrap original...")
t_bo, r_bo = boot(lr1, lr2, n_boot)
print("bootstrap simule...")
t_bs, r_bs = boot(lr1s, lr2s, n_boot)

print(f"\n{'':25s} {'mean':>8s} {'2.5%':>8s} {'97.5%':>8s}")
print(f"{'Kendall (orig)':25s} {np.mean(t_bo):8.4f} {np.percentile(t_bo,2.5):8.4f} {np.percentile(t_bo,97.5):8.4f}")
print(f"{'Kendall (sim)':25s} {np.mean(t_bs):8.4f} {np.percentile(t_bs,2.5):8.4f} {np.percentile(t_bs,97.5):8.4f}")
print(f"{'Spearman (orig)':25s} {np.mean(r_bo):8.4f} {np.percentile(r_bo,2.5):8.4f} {np.percentile(r_bo,97.5):8.4f}")
print(f"{'Spearman (sim)':25s} {np.mean(r_bs):8.4f} {np.percentile(r_bs,2.5):8.4f} {np.percentile(r_bs,97.5):8.4f}")

#CIs overlap so the copula works


# FIGURES EXERCICE 5

#data overview
fig, axes = plt.subplots(2,2, figsize=(12,10))
ax=axes[0,0]; ax.plot(p1, color='steelblue', lw=0.8, label='actif 1')
ax.plot(p2, color='coral', lw=0.8, label='actif 2')
ax.set_xlabel('jours'); ax.set_ylabel('prix'); ax.set_title('prix'); ax.legend(); ax.grid(True, alpha=0.3)

ax=axes[0,1]; ax.plot(lr1, color='steelblue', lw=0.5, alpha=0.7, label='actif 1')
ax.plot(lr2, color='coral', lw=0.5, alpha=0.5, label='actif 2')
ax.set_xlabel('jours'); ax.set_ylabel('log-ret'); ax.set_title('log-returns'); ax.legend(); ax.grid(True, alpha=0.3)

ax=axes[1,0]
ax.hist(lr1, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='white', linewidth=0.3, label='actif 1')
xh = np.linspace(min(lr1), max(lr1), 200)
ax.plot(xh, norm.pdf(xh, np.mean(lr1), np.std(lr1)), 'b-', lw=2)
ax.hist(lr2, bins=50, density=True, alpha=0.5, color='coral', edgecolor='white', linewidth=0.3, label='actif 2')
xh2 = np.linspace(min(lr2), max(lr2), 200)
ax.plot(xh2, norm.pdf(xh2, np.mean(lr2), np.std(lr2)), 'r-', lw=2)
ax.set_xlabel('log-return'); ax.set_ylabel('densite'); ax.set_title('histogrammes'); ax.legend(); ax.grid(True, alpha=0.3)

ax=axes[1,1]; ax.scatter(lr1, lr2, s=5, alpha=0.4, color='steelblue')
ax.set_xlabel('LR actif 1'); ax.set_ylabel('LR actif 2')
ax.set_title(f'scatter (corr={corr_lr:.3f})'); ax.grid(True, alpha=0.3)

plt.suptitle('Exercice 5: donnees', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ex5_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()

#QQ plots
fig, axes = plt.subplots(1,2, figsize=(10,4.5))
for idx, (lr, name) in enumerate([(lr1,'actif 1'),(lr2,'actif 2')]):
    ax = axes[idx]; sl = np.sort(lr); nq = len(sl)
    tq = norm.ppf((np.arange(1,nq+1)-0.5)/nq)
    ln = (sl - np.mean(lr))/np.std(lr)
    ax.scatter(tq, ln, s=5, alpha=0.5, color='steelblue')
    ax.plot([-4,4],[-4,4],'r--',lw=1.5)
    ax.set_xlabel('th N(0,1)'); ax.set_ylabel('emp')
    ax.set_title(f'QQ {name}'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ex5_qq_plots.png', dpi=150, bbox_inches='tight')
plt.close()

#simulation plots
fig, axes = plt.subplots(2,2, figsize=(12,10))
ax=axes[0,0]; ax.plot(range(n_data), p1, 'steelblue', lw=0.8, label='original')
ax.plot(range(n_data, n_data+1001), p1s, 'orange', lw=0.8, label='simule')
ax.axvline(x=n_data, color='red', ls='--', alpha=0.5)
ax.set_xlabel('jours'); ax.set_ylabel('prix'); ax.set_title('actif 1'); ax.legend(); ax.grid(True, alpha=0.3)

ax=axes[0,1]; ax.plot(range(n_data), p2, 'coral', lw=0.8, label='original')
ax.plot(range(n_data, n_data+1001), p2s, 'orange', lw=0.8, label='simule')
ax.axvline(x=n_data, color='red', ls='--', alpha=0.5)
ax.set_xlabel('jours'); ax.set_ylabel('prix'); ax.set_title('actif 2'); ax.legend(); ax.grid(True, alpha=0.3)

ax=axes[1,0]
ax.hist(lr1s, bins=50, density=True, alpha=0.5, color='orange', edgecolor='white', linewidth=0.3, label='sim')
ax.hist(lr1, bins=50, density=True, alpha=0.5, color='steelblue', edgecolor='white', linewidth=0.3, label='orig')
ax.set_xlabel('log-ret'); ax.set_ylabel('densite'); ax.set_title('actif 1: LR'); ax.legend(); ax.grid(True, alpha=0.3)

ax=axes[1,1]
ax.scatter(lr1s, lr2s, s=5, alpha=0.3, color='orange', label=f'sim (r={cs:.3f})')
ax.scatter(lr1, lr2, s=5, alpha=0.3, color='steelblue', label=f'orig (r={corr_lr:.3f})')
ax.set_xlabel('LR 1'); ax.set_ylabel('LR 2'); ax.set_title('scatter'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.suptitle('Exercice 5: simulation', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ex5_simulation.png', dpi=150, bbox_inches='tight')
plt.close()

#bootstrap plots
fig, axes = plt.subplots(1,2, figsize=(10,4.5))
ax=axes[0]; ax.hist(t_bo, bins=30, density=True, alpha=0.5, color='steelblue', label='orig')
ax.hist(t_bs, bins=30, density=True, alpha=0.5, color='orange', label='sim')
ax.axvline(x=tau_k, color='blue', ls='--', lw=1.5); ax.axvline(x=tau_sim, color='orange', ls='--', lw=1.5)
ax.set_xlabel('Kendall tau'); ax.set_title('bootstrap: Kendall'); ax.legend(); ax.grid(True, alpha=0.3)

ax=axes[1]; ax.hist(r_bo, bins=30, density=True, alpha=0.5, color='steelblue', label='orig')
ax.hist(r_bs, bins=30, density=True, alpha=0.5, color='orange', label='sim')
ax.axvline(x=rho_orig, color='blue', ls='--', lw=1.5); ax.axvline(x=rho_sim, color='orange', ls='--', lw=1.5)
ax.set_xlabel('Spearman rho'); ax.set_title('bootstrap: Spearman'); ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('Exercice 5: bootstrap', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ex5_bootstrap.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n=> toutes les figures sauvegardees")
print("\n" + "="*60)
print("PROJET FINI !")
print("="*60)
