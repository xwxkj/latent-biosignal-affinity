from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path('/mnt/data/results_independence_controlled_unzipped/results_independence_controlled')
OUT = Path('/mnt/data/LBA_reanalysis_summary_assets')
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9.5,
    'axes.titlesize': 11.5,
    'axes.labelsize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.2,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

COL = {'WESAD':'#3D8B67','CASE':'#D9705A','PTB-XL':'#4C78A8'}

summary = pd.read_csv(ROOT/'three_dataset_independence_controlled_summary.csv')
summary.to_csv(OUT/'reanalysis_primary_results.csv', index=False)

# ---------------- Figure 2: independent-unit distributions ----------------
fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.55), constrained_layout=True)
items = [
    ('WESAD', ROOT/'wesad/primary_cross_subject/independent_unit_deltas.csv', 'Participant-level Δ', (-0.05, 0.85)),
    ('CASE', ROOT/'case/primary_valence_arousal_quadrant/independent_unit_deltas.csv', 'Participant-level Δ', (-0.06, 0.08)),
    ('PTB-XL', ROOT/'ptbxl/primary_patient_independent/independent_unit_deltas.csv', 'Patient-level Δ', (-0.52, 0.60)),
]
rng = np.random.default_rng(20260730)
for idx,(name,path,ylabel,ylim) in enumerate(items):
    ax = axes[idx]
    df = pd.read_csv(path)
    vals = df['delta'].to_numpy(float)
    # violin
    parts = ax.violinplot(vals, positions=[1], widths=0.68, showmeans=False, showmedians=False, showextrema=False)
    for body in parts['bodies']:
        body.set_facecolor(COL[name]); body.set_edgecolor(COL[name]); body.set_alpha(0.22); body.set_linewidth(0.8)
    # points: all for WESAD/CASE, deterministic subsample for PTB
    if len(vals) > 900:
        display = rng.choice(vals, size=900, replace=False)
    else:
        display = vals
    jitter = rng.normal(0, 0.055, size=len(display))
    ax.scatter(np.full(len(display),1.0)+jitter, display, s=12 if len(display)<100 else 5,
               alpha=0.72 if len(display)<100 else 0.30, color=COL[name], edgecolors='none', rasterized=len(display)>100)
    row = summary.loc[summary['dataset']==name].iloc[0]
    mean = row['mean_unit_delta']; lo=row['bootstrap_ci_95_low']; hi=row['bootstrap_ci_95_high']
    ax.errorbar(1.28, mean, yerr=[[mean-lo],[hi-mean]], fmt='o', ms=5.6, capsize=3.5,
                color='black', ecolor='black', lw=1.2, zorder=5)
    ax.axhline(0, color='#666666', lw=0.8, ls='--', zorder=0)
    ax.set_xlim(0.48,1.55); ax.set_ylim(*ylim)
    ax.set_xticks([1]); ax.set_xticklabels([name])
    ax.set_ylabel(ylabel)
    n=int(row['n_independent_units'])
    ax.set_title(f"{chr(97+idx)}  {name}  (independent n = {n:,})", loc='left', fontweight='bold')
    ax.text(0.02,0.98, f"mean Δ = {mean:.4f}\n95% CI [{lo:.4f}, {hi:.4f}]\npermutation P = {row['permutation_p_two_sided']:.4f}",
            transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor='#BBBBBB', linewidth=0.7, alpha=0.94))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.suptitle('Independent-unit physiological similarity contrasts', fontsize=13.2, fontweight='bold')
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'Figure_independent_unit_deltas.{ext}', dpi=400 if ext=='png' else None, bbox_inches='tight')
plt.close(fig)

# ---------------- Figure 3: robustness and stimulus sensitivity ----------------
fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.75), constrained_layout=True)

# a WESAD latent dimensions
ax=axes[0]
w = pd.read_csv(ROOT/'wesad/sensitivity_latent_dimension_and_metric.csv')
for metric,marker,ls in [('cosine','o','-'),('correlation','s','--')]:
    d=w[w.metric==metric].sort_values('latent_dim')
    ax.plot(d.latent_dim,d.mean_unit_delta,marker=marker,ls=ls,lw=1.5,ms=5,label=metric.capitalize())
    ax.fill_between(d.latent_dim,d.bootstrap_ci_95_low,d.bootstrap_ci_95_high,alpha=0.12)
ax.axhline(0,color='#666666',ls='--',lw=0.8)
ax.set_xticks([4,8,16]); ax.set_xlabel('Latent dimension'); ax.set_ylabel('Mean participant-level Δ')
ax.set_title('a  WESAD embedding robustness',loc='left',fontweight='bold'); ax.legend(frameon=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# b PTB-XL dimensions + adjusted
ax=axes[1]
p = pd.read_csv(ROOT/'ptbxl/sensitivity_latent_dimension_and_metric.csv')
for metric,marker,ls in [('cosine','o','-'),('correlation','s','--')]:
    d=p[p.metric==metric].sort_values('latent_dim')
    ax.plot(d.latent_dim,d.mean_unit_delta,marker=marker,ls=ls,lw=1.5,ms=5,label=metric.capitalize())
    ax.fill_between(d.latent_dim,d.bootstrap_ci_95_low,d.bootstrap_ci_95_high,alpha=0.12)
adj=pd.read_csv(ROOT/'ptbxl/sensitivity_age_sex_adjusted/summary.csv').iloc[0]
ax.errorbar([8.7],[adj.mean_unit_delta],yerr=[[adj.mean_unit_delta-adj.bootstrap_ci_95_low],[adj.bootstrap_ci_95_high-adj.mean_unit_delta]],
            fmt='D',color='#9C755F',capsize=3,ms=5,label='Age/sex adjusted')
ax.axhline(0,color='#666666',ls='--',lw=0.8)
ax.set_xticks([4,8,16]); ax.set_xlabel('Latent dimension'); ax.set_ylabel('Mean patient-level Δ')
ax.set_title('b  PTB-XL robustness',loc='left',fontweight='bold'); ax.legend(frameon=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# c CASE context dependence
ax=axes[2]
case_primary=summary.loc[summary.dataset=='CASE'].iloc[0]
video=pd.read_csv(ROOT/'case/sensitivity_video_centered/summary.csv').iloc[0]
loo=pd.read_csv(ROOT/'case/leave_one_video_out.csv')
labels=['Primary\ncross-video','Video-centred','Leave-one-video-out\nrange']
x=np.arange(3)
ax.errorbar(x[0],case_primary.mean_unit_delta,
            yerr=[[case_primary.mean_unit_delta-case_primary.bootstrap_ci_95_low],[case_primary.bootstrap_ci_95_high-case_primary.mean_unit_delta]],
            fmt='o',color=COL['CASE'],ms=6,capsize=4,lw=1.4)
ax.errorbar(x[1],video.mean_unit_delta,
            yerr=[[video.mean_unit_delta-video.bootstrap_ci_95_low],[video.bootstrap_ci_95_high-video.mean_unit_delta]],
            fmt='s',color='#777777',ms=6,capsize=4,lw=1.4)
ax.vlines(x[2],loo.mean_unit_delta.min(),loo.mean_unit_delta.max(),color='#B279A2',lw=5,alpha=0.65)
ax.scatter(np.full(len(loo),x[2])+rng.normal(0,0.03,len(loo)),loo.mean_unit_delta,s=17,color='#B279A2',zorder=3)
ax.axhline(0,color='#666666',ls='--',lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('Mean participant-level Δ')
ax.set_title('c  CASE stimulus sensitivity',loc='left',fontweight='bold')
ax.text(0.02,0.98,'Video-centred estimate\nP = 0.1698',transform=ax.transAxes,ha='left',va='top',fontsize=8.4,
        bbox=dict(boxstyle='round,pad=0.25',facecolor='white',edgecolor='#BBBBBB',linewidth=0.7))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.suptitle('Robustness and context dependence of latent biosignal similarity', fontsize=13.2, fontweight='bold')
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'Figure_robustness_and_stimulus_sensitivity.{ext}', dpi=400 if ext=='png' else None, bbox_inches='tight')
plt.close(fig)

print(OUT)
