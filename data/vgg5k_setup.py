import os, json, shutil, subprocess, sys
from multiprocessing import Pool
from collections import Counter
OUT_ROOT='/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset/vggsound_5k'
a=json.load(open(f'{OUT_ROOT}/vgg5k.json'))
SRC=a['video_dir']
OUT='/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset/vggsound_5k'
DST_V, DST_A = OUT+'/videos', OUT+'/audios'
os.makedirs(DST_V, exist_ok=True); os.makedirs(DST_A, exist_ok=True)

def process(clip):
    f = clip['file']
    src = f'{SRC}/{f}.mp4'; dv = f'{DST_V}/{f}.mp4'; da = f'{DST_A}/{f}.wav'
    if not os.path.exists(src): return 'no_video'
    if not (os.path.exists(dv) and os.path.getsize(dv) > 1000):
        try: shutil.copy(src, dv)
        except Exception: return 'copy_fail'
    if not (os.path.exists(da) and os.path.getsize(da) > 1000):
        subprocess.call(f'ffmpeg -y -nostdin -i "{src}" -f wav -vn -ac 1 -ab 16k -ar 22050 '
                        f'-loglevel error "{da}"', shell=True)
    return 'ok' if (os.path.exists(da) and os.path.getsize(da) > 1000) else 'no_audio'

if __name__ == '__main__':
    clips = a['clips']
    print(f'VGGSound-5K: {len(clips)} clips, src={SRC}', flush=True)
    res = Counter()
    with Pool(32) as p:
        for r in p.imap_unordered(process, clips, chunksize=8):
            res[r] += 1
    print(f'DONE: {dict(res)}', flush=True)
    print(f'  videos: {len(os.listdir(DST_V))}, audios: {len(os.listdir(DST_A))}', flush=True)
    # save the vgg5k annotation into our folder for eval convenience
    json.dump(a, open(OUT+'/vgg5k.json','w'))
