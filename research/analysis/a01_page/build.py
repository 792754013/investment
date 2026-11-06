"""把 data/research/a01 的结果注入页面模板，生成交互页面（已发布为私有 Artifact：https://claude.ai/artifact/Fn47Z3Eyvx3pAVEpAX5EBX）。
用法：python3 -I research/analysis/a01_page/build.py <仓库根目录> <输出目录>（输出目录里需要有 template.html，或改 sp 指向本目录）"""
import json, sys
repo, sp = sys.argv[1], sys.argv[2]
res = json.load(open(f'{repo}/data/research/a01/results.json'))
r, st = res['main'], res['starts']
s = json.load(open(f'{repo}/data/research/a01/series.json'))
keep = ['cagr', 'mdd', 'mdd_peak', 'mdd_trough', 'longest_underwater_months', 'longest_underwater', 'worst_year']
metrics = {}
for k, m in r.items():
    d = {x: m[x] for x in keep}
    d['cagr'] = round(d['cagr'], 5); d['mdd'] = round(d['mdd'], 5)
    d['worst_year'] = [d['worst_year'][0], round(d['worst_year'][1], 5)]
    d['rolling'] = {h: {'loss_prob': round(v['loss_prob'], 4), 'worst_ann': round(v['worst_ann'], 5)} for h, v in m['rolling'].items()}
    d['end_value'] = s[k][-1]
    d['cagr08'] = round(st['2008'][k]['cagr'], 5)
    d['med10'] = round(m['rolling']['10y']['median_ann'], 5)
    metrics[k] = d
data = {'dates': [x[:7] for x in s['dates']], 'series': {k: [round(v, 4) for v in s[k]] for k in r}, 'metrics': metrics}
t = open(f'{sp}/template.html').read()
out = t.replace('/*DATA*/null', json.dumps(data, ensure_ascii=False, separators=(',', ':')))
open(f'{sp}/alloc.html', 'w').write(out)
open(f'{sp}/preview.html', 'w').write('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' + out)
print(len(out), 'bytes')
