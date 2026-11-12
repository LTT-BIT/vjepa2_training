"""Render an actual frozen B0 replay against the effective human Subtask labels."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
EID = '3e645bf4e137d0f836c9d7ff'
OUT = ROOT / 'docs/validation/tomato_annotation_prediction_20261112'
MANIFEST = ROOT / 'runs/tomato/complete100_20261111/dataset_manifest.json'
PROBS = ROOT / f'runs/tomato/transition_time_20261109/seed42/{EID}.npz'
TRACE = ROOT / f'runs/tomato/frozen_chain_20261112_release/seed42/{EID}_timeline.jsonl'


def main():
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'font.size': 12, 'axes.unicode_minus': False})
    episode = next(x for x in json.loads(MANIFEST.read_text())['episodes'] if x['episode_id'] == EID)
    segments = episode['annotation']['segments']
    duration = episode['length'] / episode['fps']
    cached = np.load(PROBS)
    times = cached['times']
    trace = [json.loads(s) for s in TRACE.read_text().splitlines()]
    assert np.array_equal(times, [x['observation_timestamp'] for x in trace])
    assert np.allclose(cached['current_probabilities'], [x['current_probabilities'] for x in trace])
    labels = ['pickup', 'harvest', 'return', 'adjust']
    phase_classes = np.array([0, 1, 2, 3, 0, 1, 2])
    colors = ['#3989D7', '#EAA041', '#43A58A', '#9B78C4']
    names = ['拿起', '摘取', '放回', '调整']
    human = [(s['start_frame']/30, s['end_frame']/30, labels.index(s['label'])) for s in segments]
    def blocks(sample_times, values):
        return [(float(t), min(float(t)+0.5, duration), int(v)) for t, v in zip(sample_times, values) if t < duration]
    current = blocks(times, cached['current_probabilities'].argmax(1))
    # Shift output to its predicted target time, not its observation time.
    future = blocks(times + 1.5, cached['future_probabilities'].argmax(1))
    confirmed = blocks(times, phase_classes[[x['decision']['phase'] for x in trace]])
    rows = [human, current, future, confirmed]
    rownames = ['你的人工标注', 'Agent：当前预测', 'Agent：未来预测¹', 'Agent：最终阶段确认']
    fig = plt.figure(figsize=(15.5, 8.5), facecolor='#F6F8FC')
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], left=.17, right=.97, top=.79, bottom=.19, hspace=.55)
    axes = [fig.add_subplot(gs[i]) for i in range(2)]
    def timeline(ax, limits):
        ax.set_facecolor('white')
        for y, spans in zip([3, 2, 1, 0], rows):
            for start, end, label in spans:
                ax.broken_barh([(start, end-start)], (y-.28, .56), facecolors=colors[label], edgecolors='none')
        ax.set(xlim=limits, ylim=(-.55, 3.55), yticks=[3, 2, 1, 0], yticklabels=rownames)
        ax.set_xlabel('录像时间（秒）', labelpad=8)
        ax.tick_params(axis='y', length=0, pad=12)
        ax.spines[['top','right','left']].set_visible(False)
        ax.grid(axis='x', alpha=.13)
        ax.set_axisbelow(True)
    timeline(axes[0], (0, duration))
    axes[0].set_title('整段对照：同一时刻，颜色相同表示阶段一致', loc='left', fontsize=14, pad=12)
    for i, s in enumerate(segments):
        x = (s['start_frame']+s['end_frame'])/60
        axes[0].text(x, 3, str(i+1), ha='center', va='center', color='white', weight='bold')
    for s in segments[1:]:
        axes[0].axvline(s['start_frame']/30, color='#25364A', lw=.8, ls=':', alpha=.4)
    axes[0].axvspan(68, 78, color='#E04E4E', alpha=.08)
    timeline(axes[1], (68, 78))
    axes[1].set_xticks(np.arange(68, 79, 1))
    axes[1].set_title('局部放大：调整 → 第二次拿起', loc='left', fontsize=14, pad=22)
    boundary = segments[4]['start_frame']/30
    change = next(x['observation_timestamp'] for x in trace if x['decision']['changed'] and x['decision']['phase']==4)
    axes[1].axvline(boundary, color='#27364B', lw=1.5, ls='--')
    axes[1].text(boundary+.08, 3.72, f'人工边界 {boundary:.2f}秒', fontsize=11, color='#27364B')
    axes[1].axvline(change, color='#B63B3B', lw=1.1, ls=':')
    axes[1].text(change-.08, 3.72, f'确认 {change:.2f}秒（早 {boundary-change:.2f}秒）', fontsize=11, color='#B63B3B', ha='right')
    fig.text(.055, .94, '你的标注 vs 我们自己的 Agent', fontsize=24, weight='bold', color='#203047')
    fig.text(.055, .893, '真实验证录像 · B0 独立双头 seed42 · 2 Hz · 未重新训练', fontsize=13, color='#526176')
    fig.legend(handles=[Patch(color=c, label=n) for c,n in zip(colors,names)], loc='upper right', bbox_to_anchor=(.965,.95), ncol=4, frameon=False)
    fig.text(.055, .845, 'A/1103_620_642_normal_lerobot/episode_000000   |   人工行编号 1～7 表示七个有序阶段', fontsize=11, color='#526176')
    fig.text(.055, .065, '¹ Future 在 t 秒预测 t＋1.5 秒的类别；图中已向右移动 1.5 秒，与它所预测的时刻比较。开头空白因此没有预测。', fontsize=11, color='#526176')
    fig.text(.055, .033, '人工行采用当前冻结验证清单中的 Subtask 边界（含已有复核版本）；这是一个具体错误案例，不代表全部录像的平均效果。', fontsize=10, color='#526176')
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/'comparison.png', dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    receipt = {'episode_id': EID, 'model':'B0 independent h1.5 seed42', 'future_display_shift_seconds':1.5, 'human_boundary_seconds':boundary, 'confirmed_boundary_seconds':change, 'confirmation_early_seconds':boundary-change, 'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [MANIFEST,PROBS,TRACE]}}
    (OUT/'provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(OUT/'comparison.png')

if __name__ == '__main__':
    main()
