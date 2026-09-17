#!/usr/bin/env python3
"""Generate component ablation pruning plans from unit_scores.json"""
import json, random, sys
from pathlib import Path

scores_path = sys.argv[1] if len(sys.argv) > 1 else 'unit_scores.json'
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('ablation_plans')

random.seed(42)
scores = json.load(open(scores_path))['scores']
heads = [s for s in scores if s['component'] == 'head']
channels = [s for s in scores if s['component'] == 'channel']

both_sorted = sorted(scores, key=lambda x: x['score'])
heads_sorted = sorted(heads, key=lambda x: x['score'])
chans_sorted = sorted(channels, key=lambda x: x['score'])

both_global = both_sorted[:512]
head_only = heads_sorted[:min(512, len(heads_sorted))]
chan_only = chans_sorted[:min(512, len(chans_sorted))]

bhc = sum(1 for s in both_global if s['component'] == 'head')
bcc = len(both_global) - bhc
random.shuffle(heads); random.shuffle(channels)
random_matched = heads[:bhc] + channels[:bcc]

plans = {'both_global': both_global, 'head_only': head_only, 'channel_only': chan_only, 'random_matched': random_matched}

for name, units in plans.items():
    plan = {
        'ablation': name, 'pruned_total': len(units),
        'pruned_heads': sum(1 for s in units if s['component'] == 'head'),
        'pruned_channels': sum(1 for s in units if s['component'] == 'channel'),
        'to_prune': [{'component': s['component'], 'layer': s['layer'], 'index': s['index'],
            'clean_grad_mean': s['clean_grad_mean'], 'proxy_grad_mean': s['proxy_grad_mean'],
            'cosine': s['cosine'], 'score': s['score']} for s in units],
    }
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    json.dump(plan, open(d / 'pruning_plan.json', 'w'), indent=2, ensure_ascii=False)
    print(f'{name}: pruned={len(units)} heads={plan["pruned_heads"]} chans={plan["pruned_channels"]}')
print(f'Done: {out_dir}')
