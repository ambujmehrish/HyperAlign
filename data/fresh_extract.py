import os, sys, subprocess
from multiprocessing import Pool
from collections import Counter
DS='/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset/vast27m_150k'
CL, OUT = DS+'/clips', DS+'/audios_wav'
os.makedirs(OUT, exist_ok=True)

def extract(fname):
    cid=fname[:-4]; out=f'{OUT}/{cid}.wav'
    # GRAM offline_process_data.py EXACT spec: -f wav -vn -ac 1 -ab 16k -ar 22050 ; -y = fresh overwrite
    r=subprocess.call(f'ffmpeg -y -nostdin -i "{CL}/{fname}" -f wav -vn -ac 1 -ab 16k -ar 22050 '
                      f'-loglevel error "{out}"', shell=True)
    if os.path.exists(out) and os.path.getsize(out)>1000: return 'ok'
    # no audio stream (the ~20 truly-silent clips) -> ffmpeg writes tiny/empty; remove it
    if os.path.exists(out): os.remove(out)
    return 'no_audio'

if __name__=='__main__':
    shard, nshard = int(sys.argv[1]), int(sys.argv[2])
    files=sorted(f for f in os.listdir(CL) if f.endswith('.mp4'))[shard::nshard]
    print(f'shard {shard}/{nshard}: {len(files)} clips', flush=True)
    res=Counter(); done=0
    with Pool(32) as p:
        for r in p.imap_unordered(extract, files, chunksize=8):
            res[r]+=1; done+=1
            if done%20000==0: print(f'  [{done}/{len(files)}] {dict(res)}', flush=True)
    print(f'shard {shard} DONE: {dict(res)}', flush=True)
