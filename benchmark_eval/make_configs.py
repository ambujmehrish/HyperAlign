import os
"""Generate zero-shot eval configs for the trained end-to-end 4-model checkpoint.
Reads GRAM/local configs, writes NEW eval configs here.

ONE config = ONE (benchmark, mode) with a single task string, matching how GRAM's own
eval configs are structured (the Gramian-volume path scores whatever modalities the task
loads, so tva vs tvas must be separate runs). ret_bidirection_evaluation forced True so
both T2V (forward) and V2T (backward) are produced (paper Table 8 + 9).

Paper-exact scope:
  MSR-VTT, VATEX        modes: tv, tva, tvas
  DiDeMo, ActivityNet   modes: tv, tva
  AudioCaps             mode : ta       (text->audio retrieval; no video available)
  VGGSound              classification  (handled by a separate script, not here)
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
E2E  = os.path.dirname(HERE)                        # the HyperAlign folder (self-locating)
sys.path.insert(0, HERE)
from pick_ckpt import pick_ckpt                      # prefer best-val, fallback latest model_step
OUT  = f'{HERE}/configs'
os.makedirs(OUT, exist_ok=True)
# our pretrained 4-model (best-val checkpoint, GRAM-style save_best)
CKPT = os.environ.get('GRAM_CKPT') or pick_ckpt(f'{E2E}/workdir_v2full/4model/ckpt', f'{E2E}/workdir_v2full/4model/ckpt/PRETRAIN_FIRST.pt')
print(f'  zero-shot init: {os.path.basename(CKPT)}'
      + ('' if os.path.exists(CKPT) else '  ⚠️ run pretrain first!'))

# model_cfg MUST match the trained checkpoint architecture (stage B + hypergraph).
train_cfg = json.load(open(f'{E2E}/config/gram/pretrain_cfg/hyperalign.json'))
model_cfg = dict(train_cfg['model_cfg'])
model_cfg['default'] = f'{E2E}/config/gram/default_model_cfg.json'
model_cfg['ret_bidirection_evaluation'] = True     # need V2T too (Table 9)

# benchmark -> (source finetune cfg to borrow val block, [modes])
BENCH = {
    'msrvtt':      ('retrieval-msrvtt',      ['tv', 'tva', 'tvas']),
    'didemo':      ('retrieval-didemo',      ['tv', 'tva']),
    'activitynet': ('retrieval-activitynet', ['tv', 'tva']),
    'vatex':       ('retrieval-vatex',       ['tv', 'tva', 'tvas']),
    'audiocaps':   ('retrieval-audiocaps',   ['tva']),   # T-V-A (videos downloaded from YouTube)
}
# 5th finetuning: MSR-VTT WITH depth (5-modal, tvasd). Only added when DEPTH_EVAL=1 so the normal
# zero-shot loop never evals a depth config on the 4-modal pretrained checkpoint (that would be meaningless).
if os.environ.get('DEPTH_EVAL'):
    BENCH['msrvtt_depth'] = ('retrieval-msrvtt_depth', ['tvasd'])

def run_cfg():
    return {'default': f'{E2E}/config/gram/default_run_cfg.json',
            'mode': 'testing', 'checkpoint': CKPT}

made = []
DS = '/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset'
for name, (srccfg, modes) in BENCH.items():
    src = json.load(open(f'{E2E}/config/gram/finetune_cfg/{srccfg}.json'))
    base_val = dict(src['data_cfg']['val'][0])
    # per-benchmark model_cfg: merge GRAM's overrides (e.g. max_caption_len:70 for didemo/activitynet
    # long paragraph captions) onto our hypergraph model_cfg — GRAM zero-shot uses the full
    # retrieval-*.json in --mode testing, so these overrides apply to zero-shot too (else captions
    # truncate at the 40 default and didemo/anet score unfairly low)
    bench_mc = dict(model_cfg)
    for _k, _v in src.get('model_cfg', {}).items():
        if _k != 'default':
            bench_mc[_k] = _v
    if name == 'audiocaps':               # T-V-A (paper Table 5: AudioCaps is Threemodal T-V-A, 8 frames)
        # our own AudioCaps directory — the one retrieval-audiocaps.json uses. The old
        # gram_data/audiocaps copy is stale (699 files, and its ids match only 165 of the
        # annotation), which silently shrank the zero-shot gallery to a quarter of the paper's.
        acdir = f'{DS}/AudioCaps'
        base_val['audio'] = f'{acdir}/audios'
        base_val['vision'] = f'{acdir}/videos'                    # videos downloaded from YouTube
        base_val['vision_format'] = 'video_rawvideo'
        base_val['vision_transforms'] = 'crop_flip'
        base_val['vision_sample_num'] = 8
        # filter annotation to clips that have BOTH audio(.wav) AND video(.mp4) on disk
        _ann = json.load(open(f'{E2E}/{base_val["txt"]}'))
        _keep = [a for a in _ann
                 if os.path.exists(f'{acdir}/videos/{a["video_id"]}.mp4')
                 and os.path.exists(f'{acdir}/audios/{a["video_id"]}.wav')]
        _fann = f'{HERE}/audiocaps_tva_annotation.json'
        json.dump(_keep, open(_fann, 'w'))
        base_val['txt'] = _fann
        print(f'  audiocaps T-V-A: {len(_keep)}/{len(_ann)} clips have both audio+video')
    for mode in modes:
        val = dict(base_val)
        val['task'] = f'ret%{mode}'
        val['batch_size'] = 32
        val['n_workers'] = 8
        # train block = copy of val: parser (compute_max_vision_sample_num) needs data_cfg.train;
        # testing mode never builds a train loader, so it only feeds the position-embedding sizing.
        cfg = {'run_cfg': run_cfg(), 'model_cfg': bench_mc,
               'data_cfg': {'train': [dict(val)], 'val': [val]}}
        fn = f'zs_{name}_{mode}.json'
        json.dump(cfg, open(f'{OUT}/{fn}', 'w'), indent=1)
        made.append(fn); print(f'  {fn:28} task=ret%{mode}')

# ---- VGGSound 5K : T-AV classification (Acc@1/@10) ----------------------------------------
# build annotation [{video_id, desc=class_name}] from vgg5k.json; name='vgg_ret' triggers the
# classification path in evaluation_classification.evaluate_ret.
VGG = '/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset/vggsound_5k'
vgg = json.load(open(f'{VGG}/vgg5k.json'))
ann = [{'video_id': c['file'], 'desc': c['cls']} for c in vgg['clips']]
ann_path = f'{HERE}/vgg5k_annotation_5000.json'
json.dump(ann, open(ann_path, 'w'))
n_cls = len({a['desc'] for a in ann})
print(f'\n  vgg annotation: {len(ann)} clips, {n_cls} unique classes -> {ann_path}')

vgg_val = {
    'type': 'annoindexed', 'training': False, 'name': 'vgg_ret',
    'txt': ann_path,
    'vision_transforms': 'crop_flip', 'vision_format': 'video_rawvideo',
    'vision': f'{VGG}/videos', 'audio': f'{VGG}/audios',
    'vision_sample_num': 8, 'audio_sample_num': 1,
    'task': 'ret%tva',                 # T-AV = text + video + audio (volume_computation3)
    'n_workers': 8, 'batch_size': 32,
}
json.dump({'run_cfg': run_cfg(), 'model_cfg': model_cfg,
           'data_cfg': {'train': [dict(vgg_val)], 'val': [vgg_val]}},
          open(f'{OUT}/zs_vggsound_tav.json', 'w'), indent=1)
made.append('zs_vggsound_tav.json'); print(f'  zs_vggsound_tav.json         task=ret%tva (T-AV classification)')

print(f'\n{len(made)} configs -> {OUT}')
