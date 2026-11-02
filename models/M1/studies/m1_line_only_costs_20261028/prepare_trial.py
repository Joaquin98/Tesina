"""Guarded adaptations of the previously verified experiment; never edit it."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import ast, json, hashlib
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
OLD = ROOT / 'data/m1_line_costs_20261028'

def replace(s, a, b, count=None):
    n = s.count(a)
    assert n > 0 and (count is None or n == count), (a, n, count)
    return s.replace(a, b)
sources = {}

def source(name):
    p = OLD / name
    sources[str(p)] = hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()
    return _resolve_model_artifact(p).read_text(encoding='utf-8')

def write(name, s):
    dest = HERE / name
    if _resolve_model_artifact(dest).exists():
        assert _resolve_model_artifact(dest).read_text(encoding='utf-8') == s, ('Refusing overwrite', name)
    else:
        dest.write_text(s, encoding='utf-8')
    if name.endswith('.py'):
        ast.parse(s)
s = source('run.py')
s = replace(s, 'COSTS=(4,8)', 'COSTS=(2,4,8)', 1)
s = replace(s, '_scrum2', '_scrum1')
s = replace(s, 'scrum_cost=2', 'scrum_cost=1')
s = replace(s, 'new_costs=[[4,1,2],[8,1,2]]', 'new_costs=[[2,1,1],[4,1,1],[8,1,1]]', 1)
s = replace(s, 'times[line_cost,1,2]', 'times[line_cost,1,1]', 1)
s = replace(s, "model='M1-H2-D2 line-only cost sweep'", "model='M1-H2-D2 extra cost only for line'", 1)
s = replace(s, 'both final epoch5 checkpoints fixed', 'all three final epoch5 checkpoints fixed', 1)
write('run.py', s)
s = source('evaluate.py')
s = replace(s, '_scrum2', '_scrum1')
s = replace(s, 'scrum_cost=2', 'scrum_cost=1')
s = replace(s, 'def main():', "def main(costs=None,training_record=None,completion_file='evaluation_complete.json'):", 1)
s = replace(s, " trained=read(HERE/'training_complete.json');assert trained['status']=='complete'", " requested=COSTS if costs is None else tuple(costs)\n assert requested and set(requested)<=set(COSTS)\n trained=read(HERE/'training_complete.json') if training_record is None else training_record\n assert trained['status']=='complete' and set(requested)<={r['line_cost'] for r in trained['models']}", 1)
s = replace(s, '  for cost in COSTS:', '  for cost in requested:', 1)
s = replace(s, "save(HERE/'evaluation_complete.json',dict(status='complete',records=records,", "save(HERE/completion_file,dict(status='complete',partial=costs is not None,line_costs=list(requested),records=records,", 1)
write('evaluate.py', s)
s = source('publish.py')
s = replace(s, 'm1_line_costs_20261028', 'm1_line_only_costs_20261028')
s = replace(s, "len(evaluated['records'])==6", "len(evaluated['records'])==9", 1)
line = next((x for x in s.splitlines() if x.startswith('  variants=')))
variants = "  variants=[dict(key='original',title='Original Â· completo',rid=job['original_id']),dict(key='x2',title='M1 Ã—2 Â· line2 / scrum2',rid=job['baseline_id']),dict(key='old_line4',title='Anterior Â· line4 / scrum2',rid=f'20261028_argentina_allblacks_{date}_m1_line4_scrum2_stride1',line_cost=4),dict(key='old_line8',title='Anterior Â· line8 / scrum2',rid=f'20261028_argentina_allblacks_{date}_m1_line8_scrum2_stride1',line_cost=8)]+[dict(key=f'line{cost}',title=f'Nuevo Â· line{cost} / scrum1',cost=cost,rid=f'20261028_argentina_allblacks_{date}_m1_line{cost}_scrum1_stride1') for cost in (2,4,8)]"
s = replace(s, line, variants, 1)
s = replace(s, "out=HERE/f'evaluation/{date}/line{cost}_scrum2'", "out=HERE/f'evaluation/{date}/line{cost}_scrum1'", 1)
s = replace(s, "source_configuration=f'M1-H2-D2-S42-LINE{cost}-OTHER1-SCRUM2'", "source_configuration=f'M1-H2-D2-S42-LINE{cost}-OTHER1-SCRUM1'", 1)
s = replace(s, 'training_class_costs=dict(line=cost,other=1,scrum=2)', 'training_class_costs=dict(line=cost,other=1,scrum=1)', 1)
s = replace(s, "line_cost=cost if cost else 2 if v['key']=='x2' else None", "line_cost=cost if cost else v.get('line_cost',2 if v['key']=='x2' else None)", 1)
s = replace(s, "scrum_cost=2 if v['key']!='original' else None", "scrum_cost=1 if cost else 2 if v['key']!='original' else None", 1)
s = replace(s, "_m1_line_costs_full'", "_m1_line_only_costs_full'", 1)
s = replace(s, 'configurations=12', 'configurations=21', 1)
s = replace(s, 'Published12variants', 'Published21variants', 1)
s = replace(s, 'ComparaciÃ³n completa: original, M1Ã—2(line2/otro1/scrum2), line4/otro1/scrum2 y line8/otro1/scrum2.', 'ComparaciÃ³n completa: original y anteriores line2/4/8 con scrum2, frente a nuevos line2/4/8 con otro1 y scrum1. Se elimina sÃ³lo el multiplicador extra de scrum; conserva pÃ©rdida normal y balanceo de clases.', 1)
s = replace(s, 'M1: mayor peso a los errores de line', 'M1: peso adicional sÃ³lo en la fila de line', 1)
needle = " result=dict(status='complete',variants=summaries"
assert needle in s
peer = '  peer_changes=[]\n  for cost in (2,4,8):\n   key=f\'line{cost}\';peer=\'x2\' if cost==2 else f\'old_line{cost}\';new=next(v for v in local if v[\'key\']==key);previous=next(v for v in local if v[\'key\']==peer)\n   peer_changes.append(dict(variant=key,peer=peer,line_recovered=sorted((matches[key]-matches[peer])&line_ids),line_lost=sorted((matches[peer]-matches[key])&line_ids),scrum_lost=sorted((matches[peer]-matches[key])&scrum_ids),review_delta_s=new[\'review\'][\'unique_flagged_s\']-previous[\'review\'][\'unique_flagged_s\']))\n  for dest in (HERE,WEB):save(dest/f\'peer_changes_{date}.json\',peer_changes)\n  peer_rows=\'\'\n  for c in peer_changes:\n   delta=c[\'review_delta_s\'];text=(\'+\' if delta>=0 else \'âˆ’\')+p.tm(abs(delta))\n   peer_rows+=f\'<tr><th>{c["variant"]}:scrum1 frente a scrum2</th><td>{len(c["line_recovered"])}</td><td>{len(c["line_lost"])}</td><td>{len(c["scrum_lost"])}</td><td>{text}</td></tr>\'\n  sections.append(f\'<h3>Efecto de quitar el costo extra de scrum Â·{date_label}</h3><div class="scroll"><table data-peers="{date}"><thead><tr><th>ComparaciÃ³n emparejada</th><th>Lines recuperados</th><th>Lines perdidos</th><th>Scrums perdidos</th><th>Cambio de revisiÃ³n</th></tr></thead><tbody>{peer_rows}</tbody></table></div>\')\n'
s = s.replace(needle, peer + needle, 1)
s = replace(s, "result=dict(status='complete',variants=summaries,comparisons=comparisons,protocol=protocol,training=training,", 'result=dict(status=\'complete\',variants=summaries,comparisons=comparisons,protocol=protocol,training=training,peer_changes={j[\'date\']:read(HERE/f\'peer_changes_{j["date"]}.json\') for j in JOBS},', 1)
s = replace(s, 'training=training,peer_changes=', "training=training,evaluation_timing_amendment=read(HERE/'evaluation_amendment.json') if (HERE/'evaluation_amendment.json').exists() else None,peer_changes=", 1)
write('publish.py', s)
write('pipeline.py', source('pipeline.py'))
s = source('verify_web.cjs')
s = replace(s, 'm1_line_costs_20261028', 'm1_line_only_costs_20261028')
s = replace(s, 'r.variants.length!==12', 'r.variants.length!==21', 1)
s = replace(s, 'count()!==12', 'count()!==21', 1)
s = replace(s, 'count()!==15', 'count()!==24', 1)
s = replace(s, 'count()!==6', 'count()!==9', 1)
s = replace(s, 'count()!==5', 'count()!==8', 1)
s = replace(s, 'variants:12', 'variants:21', 1)
paired_test = ' if(await page.locator(\'table[data-peers] tbody tr\').count()!==9)throw Error(\'Missing paired scrum-cost comparisons\');\n for(const [date,changes] of Object.entries(r.peer_changes)){\n  const rows=page.locator(`table[data-peers="${date}"] tbody tr`);\n  for(let i=0;i<changes.length;i++){\n   const got=(await rows.nth(i).locator(\'td\').allTextContents()).slice(0,3).map(Number),c=changes[i],want=[c.line_recovered.length,c.line_lost.length,c.scrum_lost.length];\n   if(JSON.stringify(got)!==JSON.stringify(want))throw Error(\'Wrong paired scrum-cost summary \'+date);\n  }\n }\n'
s = replace(s, ' for(const [date,changes] of Object.entries(r.changes)){', paired_test + ' for(const [date,changes] of Object.entries(r.changes)){', 1)
write('verify_web.cjs', s)
s = source('watch.py')
s = replace(s, 'm1_line_costs_20261028', 'm1_line_only_costs_20261028')
s = replace(s, 'line4/otro1/scrum2 y line8/otro1/scrum2', 'line2/otro1/scrum1,line4/otro1/scrum1 y line8/otro1/scrum1', 1)
s = replace(s, 'original y M1Ã—2 sobre los tres partidos completos', 'original,M1Ã—2 y anteriores line4/8 con scrum2 sobre los tres partidos completos', 1)
write('watch.py', s)
s = source('verify_progress.cjs')
s = replace(s, 'm1_line_costs_20261028', 'm1_line_only_costs_20261028')
s = replace(s, '[4,8].includes', '[2,4,8].includes', 1)
s = replace(s, "'[[4,1,2],[8,1,2]]'", "'[[2,1,1],[4,1,1],[8,1,1]]'", 1)
write('verify_progress.cjs', s)
(HERE / 'adaptation_sources.json').write_text(json.dumps(dict(source_hashes=sources, changes='New extra costs[line,other,scrum]=[2,1,1],[4,1,1],[8,1,1]; preserve all old sources and results; add paired scrum1-vs2 comparisons'), indent=2), encoding='utf-8')
print('Prepared isolated3-model experiment; all adapted Python syntax validated')
