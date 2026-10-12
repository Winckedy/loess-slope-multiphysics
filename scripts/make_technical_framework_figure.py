import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# ---------- 辅助函数 ----------
def rbox(x, y, w, h, fc, ec, lw=1.5, radius=0.18, alpha=1.0, zorder=2):
    box = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        alpha=alpha, zorder=zorder)
    ax.add_patch(box)

def arrow(x1, y1, x2, y2, color='#555555', lw=1.8,
          style='-|>', ls='-', zorder=1):
    a = FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=18,
        color=color, linewidth=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(a)

# ============ 标题 ============
ax.text(10.0, 13.5,
        'Five-Layer Method Ladder for Rainfall–Deformation Causal Inference',
        ha='center', va='center', fontsize=13.5, fontweight='bold', color='#1A252F')
ax.text(10.0, 13.0,
        'each layer closes a gap the previous leaves open  |  every method first validated on synthetic ground truth',
        ha='center', va='center', fontsize=9, style='italic', color='#555555')

# ============ 数据层 ============
rbox(0.2, 7.5, 2.4, 4.5, '#D6EAF8', '#2471A3', lw=2)
ax.text(1.4, 11.55, 'MONITORING\nDATA', ha='center', va='center',
        fontsize=10.5, fontweight='bold', color='#1A5276')
ax.text(1.4, 10.25, '3 sites:\nXiji · Qingyang\n· Tianshui',
        ha='center', va='center', fontsize=8, color='#1A5276')
ax.text(1.4, 8.95, 'Multi-physics:\nrain, tilt, pwp,\nmoisture, water\nlevel, laser',
        ha='center', va='center', fontsize=7.5, color='#1A5276')
ax.text(1.4, 7.95, 'Daily resampling\n1,681 site-days',
        ha='center', va='center', fontsize=8, color='#1A5276')

# ============ 五层方法阶梯 ============
methods = [
    dict(name='WGCNA-style\nNetwork',
         discipline='Bioinformatics',
         io='All sensors →\nSensor modules',
         question='Is the deformation\nmodule coupled\nto rainfall?',
         finding='Module−rain r:\n−0.22 to −0.17',
         fc='#AED6F1', ec='#2471A3'),
    dict(name='Convergent\nCross Mapping',
         discipline='Nonlinear Dynamics',
         io='Δm, rainfall →\nCross-map skill',
         question='Is rainfall encoded\nin the tilt\nstate space?',
         finding='Skill 0.37 (3 d)\n0.35 (7 d)',
         fc='#D7BDE2', ec='#7D3C98'),
    dict(name='Transfer\nEntropy',
         discipline='Information Theory',
         io='Wet-day, Δm sign →\nDirected info flow',
         question="Does rainfall\ninform tomorrow's\ntilt increment?",
         finding='z = 11.1 (1 d)\nz = 6.7 (1 d)',
         fc='#FAD7A0', ec='#B9770E'),
    dict(name='Wavelet\nCoherence',
         discipline='Geophysics',
         io='Rain, tilt mag. →\nTime-freq. coherence',
         question='Is the coupling\noscillatory\nor episodic?',
         finding='No band above\nred-noise null',
         fc='#ABEBC6', ec='#1E8449'),
    dict(name='HMM + ML',
         discipline='Signal Proc. + ML',
         io='Δm, √rain, events →\nStates, LOSO AUC',
         question='Is coupling a state\nprocess? Does it\ntransfer across sites?',
         finding='Occurrence: 0.75–0.89\nMagnitude: no transfer',
         fc='#F5B7B1', ec='#A93226'),
]

x_start = 2.9
w = 2.6
h = 4.5
y0 = 7.5
gap = 0.3
mid_y = y0 + h / 2

for i, m in enumerate(methods):
    x = x_start + i * (w + gap)
    rbox(x, y0, w, h, m['fc'], m['ec'], lw=2)
    cx = x + w / 2

    # 层号徽章
    badge = plt.Circle((x + 0.38, y0 + h - 0.38), 0.24,
                       color=m['ec'], zorder=4)
    ax.add_patch(badge)
    ax.text(x + 0.38, y0 + h - 0.38, str(i + 1), ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=5)

    ax.text(cx, y0 + h - 0.48, m['name'], ha='center', va='center',
            fontsize=10, fontweight='bold', color=m['ec'])
    ax.text(cx, y0 + h - 1.08, m['discipline'], ha='center', va='center',
            fontsize=7.8, style='italic', color=m['ec'], alpha=0.85)
    ax.plot([x + 0.25, x + w - 0.25],
            [y0 + h - 1.35, y0 + h - 1.35],
            color=m['ec'], lw=0.8, alpha=0.4)
    ax.text(cx, y0 + h - 1.75, m['io'], ha='center', va='center',
            fontsize=7.5, color='#333333')
    ax.text(cx, y0 + 1.55, m['question'], ha='center', va='center',
            fontsize=7.5, style='italic', color='#555555')
    ax.text(cx, y0 + 0.5, m['finding'], ha='center', va='center',
            fontsize=7.8, fontweight='bold', color=m['ec'])

# 箭头：数据 → 方法链 → 输出
arrow(2.6, mid_y, x_start, mid_y)
for i in range(4):
    x1 = x_start + i * (w + gap) + w
    x2 = x1 + gap
    arrow(x1, mid_y, x2, mid_y)
x_last = x_start + 4 * (w + gap) + w
arrow(x_last, mid_y, 17.4, mid_y)

# ============ 输出 ============
rbox(17.4, 7.5, 2.4, 4.5, '#FADBD8', '#922B21', lw=2)
ax.text(18.6, 11.55, 'OUTPUT', ha='center', va='center',
        fontsize=10.5, fontweight='bold', color='#922B21')
ax.text(18.6, 10.5, 'Three coupling\nregimes:',
        ha='center', va='center', fontsize=8.5, color='#922B21')
ax.text(18.6, 9.6, '● Strong dynamic\n  (Xiji)',
        ha='center', va='center', fontsize=8, color='#2471A3')
ax.text(18.6, 8.85, '● Weak localised\n  (Qingyang)',
        ha='center', va='center', fontsize=8, color='#7D3C98')
ax.text(18.6, 8.1, '● Slow integrated\n  (Tianshui)',
        ha='center', va='center', fontsize=8, color='#1E8449')

# ============ 六方法审计 ============
rbox(2.9, 2.3, 14.5, 4.2, '#EAECEE', '#7F8C8D', lw=2)
ax.text(10.15, 6.15, 'INDEPENDENT SIX-METHOD AUDIT',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#2C3E50')
ax.text(10.15, 5.72,
        'each method run end-to-end with machine-checkable output',
        ha='center', va='center', fontsize=8, style='italic', color='#666666')

audit_methods = [
    'Change-point\ndetection\n(PELT)',
    'Event\nstatistics',
    'Intensity–\nduration\nthresholds',
    'Shapley\nattribution',
    'Tree-ensemble\nbenchmarks\n(RF, XGBoost)',
    'Symbolic\nregression*',
]
aw = 2.15
agap = 0.15
ax_start = 3.15
for i, am in enumerate(audit_methods):
    x = ax_start + i * (aw + agap)
    rbox(x, 3.0, aw, 2.1, '#FFFFFF', '#95A5A6', lw=1.2, radius=0.12)
    ax.text(x + aw / 2, 4.05, am, ha='center', va='center',
            fontsize=7.5, color='#2C3E50')
ax.text(10.15, 2.55,
        '*not attempted: 162 events < 200-sample library minimum',
        ha='center', va='center', fontsize=7, style='italic', color='#999999')

# 虚线箭头：方法 → 审计
for i in range(5):
    x = x_start + i * (w + gap) + w / 2
    arrow(x, y0, x, 6.5, color='#AAB7B8', lw=1.0, ls='--', style='-|>')

# ============ 合成验证条 ============
rbox(0.2, 0.3, 19.6, 1.55, '#FEF9E7', '#B7950B', lw=2)
ax.text(10.0, 1.55,
        'SYNTHETIC VALIDATION — every method verified on known-answer systems before touching field data',
        ha='center', va='center', fontsize=9.5, fontweight='bold', color='#7D6608')

val_items = [
    'CCM: 0.02 → 0.97\n(true direction)',
    'TE: peak z = 782\nat true lag',
    'WTC: 38% vs 7.7%\n(planted box)',
    'HMM: Δmatrix < 0.007\nViterbi 0.99',
]
for i, vi in enumerate(val_items):
    x = 2.8 + i * 4.6
    ax.text(x, 0.82, vi, ha='center', va='center',
            fontsize=7.5, color='#7D6608')

# ============ 保存 ============
import os as _os; _out = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'results', 'figures'); _os.makedirs(_out, exist_ok=True)
plt.savefig(_os.path.join(_out, 'technical_framework.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('saved technical_framework.png')
