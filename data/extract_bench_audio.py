import os, sys, subprocess
from multiprocessing import Pool
from collections import Counter
DS='/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset'
# benchmark -> (video_dir, audio_out_dir)
BENCHES = {
  'msrvtt':      (f'{DS}/MSRVTT_full/MSRVTT/videos/all',   f'{DS}/MSRVTT_full/audios'),
  'didemo':      (f'{DS}/DiDeMo/videos_flat',              f'{DS}/DiDeMo/audios'),
  'vatex':       (f'{DS}/VATEX/videos_raw',                f'{DS}/VATEX/audios'),
  'activitynet': (f'{DS}/ActivityNet/videos/Activity_Videos', f'{DS}/ActivityNet/audios'),
}

def extract(a):
    vpath, out = a
    if os.path.exists(out) and os.path.getsize(out) > 1000: return 'skip'
    # GRAM offline spec: -f wav -vn -ac 1 -ab 16k -ar 22050
    subprocess.call(f'ffmpeg -y -nostdin -i "{vpath}" -f wav -vn -ac 1 -ab 16k -ar 22050 '
                    f'-loglevel error "{out}"', shell=True)
    if os.path.exists(out) and os.path.getsize(out) > 1000: return 'ok'
    if os.path.exists(out): os.remove(out)
    return 'no_audio'

if __name__ == '__main__':
    shard, nshard = int(sys.argv[1]), int(sys.argv[2])
    tasks = []
    for bench, (vdir, adir) in BENCHES.items():
        if not os.path.isdir(vdir):
            print(f'  WARN: {bench} video dir missing {vdir}', flush=True); continue
        os.makedirs(adir, exist_ok=True)
        vids = sorted(f for f in os.listdir(vdir) if f.endswith('.mp4'))
        for f in vids:
            tasks.append((f'{vdir}/{f}', f'{adir}/{f[:-4]}.wav'))
    tasks = tasks[shard::nshard]
    print(f'shard {shard}/{nshard}: {len(tasks)} clips across {len(BENCHES)} benchmarks', flush=True)
    res = Counter(); done = 0
    with Pool(32) as p:
        for r in p.imap_unordered(extract, tasks, chunksize=8):
            res[r] += 1; done += 1
            if done % 10000 == 0: print(f'  [{done}/{len(tasks)}] {dict(res)}', flush=True)
    print(f'shard {shard} DONE: {dict(res)}', flush=True)
