"""Single-run cache variants, separate from repeated-measurement tile sweeps.

Metadata pins the measured source, binary and run before attaching geometry to
TSV rows. It describes software buffers, not measured cache residency. This is
an evidence viewer: no simulator, median, interpolation or causal estimator.
"""
import html
import json
import re

CODEBOOK = ['MHA_QKV', 'Projection', 'FF1', 'FF2']


def require(condition, message):
    if not condition:
        raise ValueError('Cache comparison: ' + message)


def load_variants(path, frame):
    metadata = json.loads(path.read_text())
    require(isinstance(metadata, dict), 'metadata must be an object')
    require(metadata.get('schema_version') == 1, 'unsupported schema')
    entries = metadata.get('variants', [])
    require(isinstance(entries, list) and entries, 'variants must be a nonempty list')
    require(isinstance(metadata.get('configuration'), dict) and metadata['configuration'],
            'missing configuration contract')
    ids, selected = set(), []
    for entry in entries:
        require(isinstance(entry, dict), 'invalid variant')
        eid = entry.get('exp_id')
        require(isinstance(eid, str) and re.fullmatch(r'E\d+', eid), 'invalid experiment ID')
        require(eid not in ids, 'duplicate variant ' + eid)
        ids.add(eid)
        for field, size in [('source_sha', 40), ('binary_sha256', 64)]:
            require(isinstance(entry.get(field), str) and
                    re.fullmatch('[0-9a-f]{%d}' % size, entry[field]), eid + ': invalid ' + field)
        for field in ['label', 'study', 'run', 'representation', 'weight_stride_bytes']:
            require(isinstance(entry.get(field), str) and entry[field], eid + ': missing ' + field)
        for level in ['l1', 'l2']:
            tile = entry.get(level + '_tile')
            require(tile is None or (isinstance(tile, list) and len(tile) == 3 and
                    all(type(n) is int and n > 0 for n in tile)), eid + ': invalid tile')
            stride = entry.get(level + '_x_stride')
            require((tile is None and stride is None) or
                    (tile is not None and type(stride) is int and stride >= tile[2]),
                    eid + ': invalid X stride')
        require(entry.get('l2_tile') is None or entry.get('l1_tile') is not None,
                eid + ': L2 requires L1')
        require(type(entry.get('arena_bytes')) is int and entry['arena_bytes'] >= 0,
                eid + ': invalid arena size')
        rows = frame[frame.exp_id == eid]
        if rows.empty:  # --experiments may select a subset of this publication.
            continue
        expected = dict(metadata['configuration'], study=entry['study'],
                        gem5_timestamp=entry['run'], binary_sha256=entry['binary_sha256'])
        for field, value in expected.items():
            require(field in rows and (rows[field] == value).all(), eid + ': mismatch in ' + field)
        require('repo_commit' in rows and rows.repo_commit.map(lambda sha:
                isinstance(sha, str) and bool(re.fullmatch(r'[0-9a-f]{12,40}', sha)) and
                entry['source_sha'].startswith(sha)).all(), eid + ': source SHA mismatch')
        total = rows[rows.row_kind == 'final_total']
        phases = rows[rows.row_kind == 'interval_delta']
        require(len(total) == 1 and total.phase_timing_valid.all(), eid + ': invalid phase accounting')
        require(not phases.interval.duplicated().any() and set(CODEBOOK) <= set(phases.interval),
                eid + ': missing or duplicate codebook phases')
        selected.append((entry, total.iloc[0], phases.set_index('interval')))
    pairs = metadata.get('comparisons', [])
    require(isinstance(pairs, list), 'invalid comparisons')
    for pair in pairs:
        require(isinstance(pair, dict) and pair.get('baseline') in ids and pair.get('candidate') in ids
                and pair['baseline'] != pair['candidate'] and isinstance(pair.get('label'), str),
                'invalid comparison pair')
    return metadata, selected


def comparison_charts(path, frame, layout, switch_menu):
    metadata, selected = load_variants(path, frame)
    if not selected:
        return []
    labels = [html.escape(e['label']) for e, _, _ in selected]
    totals = [float(row.sim_seconds) for _, row, _ in selected]
    codebook = [float(phases.loc[CODEBOOK, 'sim_seconds'].sum()) for _, _, phases in selected]
    details = [e['source_sha'] + '<br>' + html.escape(e['run']) for e, _, _ in selected]
    figures = []
    title = 'Cache variants · one formal run each · simulated ROI'
    traces = []
    for name, values, color in [('Whole model', totals, '#2762ad'), ('Codebook phases', codebook, '#3d9db5')]:
        traces.append(dict(type='bar', name=name, x=labels, y=[v * 1000 for v in values],
                           marker=dict(color=color), customdata=details,
                           hovertemplate='%{x}<br>%{y:.3f} ms<br>%{customdata}<extra>%{fullData.name}</extra>'))
    lay = layout(title, 470, barmode='group')
    lay['yaxis'].update(title=dict(text='Simulated time (ms)'), rangemode='tozero')
    figures.append(dict(data=traces, layout=lay))

    by_id = {e['exp_id']: (total, cb) for (e, _, _), total, cb in zip(selected, totals, codebook)}
    pairs = [p for p in metadata['comparisons'] if p['baseline'] in by_id and p['candidate'] in by_id]
    if pairs:
        traces = []
        for i, name in enumerate(['Whole model', 'Codebook phases']):
            values = [by_id[p['baseline']][i] / by_id[p['candidate']][i]
                      if by_id[p['candidate']][i] > 0 else None for p in pairs]
            traces.append(dict(type='bar', name=name, x=[html.escape(p['label']).replace(': ', '<br>', 1) for p in pairs], y=values,
                               marker=dict(color=['#2762ad', '#3d9db5'][i]),
                               hovertemplate='%{x}<br>%{y:.6f}x baseline / candidate<extra>%{fullData.name}</extra>'))
        lay = layout('Controlled comparisons · above 1x is faster', 500, barmode='group')
        lay['yaxis'].update(title=dict(text='Baseline time / candidate time'), rangemode='tozero')
        lay['shapes'] = [dict(type='line', xref='paper', x0=0, x1=1, y0=1, y1=1,
                              line=dict(color='#68778b', dash='dash'))]
        figures.append(dict(data=traces, layout=lay))

    # Keep all phases visible: unchanged QK/SV/non-GEMM effects must not be
    # silently credited to the four modified codebook regions.
    intervals = selected[0][2].index.tolist()
    require(all(set(p.index) == set(intervals) for _, _, p in selected), 'mixed phase schemas')
    metrics = [('sim_seconds', 'Simulated time (ms)', 1000), ('instructions', 'Instructions', 1),
               ('dcache_demand_accesses', 'L1D demand accesses', 1), ('dcache_demand_misses', 'L1D demand misses', 1),
               ('l1d_miss_pct', 'L1D demand miss rate (%)', 1), ('l2_demand_accesses', 'L2 demand accesses', 1),
               ('l2_demand_misses', 'L2 demand misses', 1), ('l2_miss_pct', 'L2 demand miss rate (%)', 1)]
    traces = [dict(type='heatmap', x=labels, y=intervals,
                   z=[[float(phases.loc[phase, field]) * scale for _, _, phases in selected] for phase in intervals],
                   visible=i == 0, colorscale=[[0, '#f1f6fb'], [1, '#21599c']], xgap=1, ygap=1, hoverongaps=False,
                   colorbar=dict(title=dict(text=name), thickness=14),
                   hovertemplate='%{x}<br>%{y}<br>' + name + ': %{z:.4f}<extra></extra>')
              for i, (field, name, scale) in enumerate(metrics)]
    buttons = [dict(label=name, method='update', args=[dict(visible=[i == j for i in range(len(metrics))])])
               for j, (_, name, _) in enumerate(metrics)]
    lay = layout('Per-phase evidence · codebook = QKV + Projection + FF1 + FF2', 600,
                 updatemenus=switch_menu(buttons))
    lay['yaxis'].update(autorange='reversed')
    figures.append(dict(data=traces, layout=lay))

    def tile(value):
        return 'None' if value is None else ' × '.join(str(n) for n in value)

    columns = [labels,
               [html.escape(e['representation']) for e, _, _ in selected],
               [tile(e['l1_tile']) for e, _, _ in selected],
               [tile(e['l2_tile']) for e, _, _ in selected],
               [' / '.join(str(e[level + '_x_stride']) if e[level + '_x_stride'] is not None else '—'
                           for level in ['l1', 'l2']) for e, _, _ in selected],
               [html.escape(e['weight_stride_bytes']) for e, _, _ in selected],
               ['{:g}'.format(e['arena_bytes'] / 1024) for e, _, _ in selected]]
    table = dict(type='table', columnwidth=[95, 115, 115, 130, 85, 110, 85],
                 header=dict(values=['Variant', 'Stored weights', 'L1 (S,O,K)', 'L2 (S,O,K)',
                                     'X1/X2 stride B', 'I/W1 / I/W2<br>row stride B', 'Arena KiB'],
                             fill=dict(color='#234e75'), font=dict(color='white'), height=40, align='left'),
                 cells=dict(values=columns, fill=dict(color='#f0f5fa'), align='left', height=42))
    config = metadata['configuration']
    lay = layout('Measured geometry · {} L1D / {} L2 · SVE{}'.format(
        config.get('l1d', '?'), config.get('l2', '?'), config.get('sve_bits', '?')), 420)
    lay['margin'] = dict(l=20, r=20, t=65, b=95)
    lay['annotations'] = [dict(text='Fixed-geometry algorithm comparisons, not a tile-size sweep. No medians.<br>'
                              'Arena = allocated software buffers, not cache occupancy or transferred bytes.',
                              x=0, y=-0.14, xref='paper', yref='paper', xanchor='left', showarrow=False)]
    figures.append(dict(data=[table], layout=lay))
    return figures
