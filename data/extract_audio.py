import os, sys, subprocess
from multiprocessing import Pool
DS = '/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset/vast27m_150k'
CL, OUT = DS+'/clips', DS+'/audios_wav'
os.makedirs(OUT, exist_ok=True)
def extract(cid):
    out = f'{OUT}/{cid}.wav'
    if os.path.exists(out): return
    # GRAM offline_process_data.py exact cmd: -f wav -vn -ac 1 -ab 16k -ar 22050
    subprocess.call(f'ffmpeg -y -nostdin -i "{CL}/{cid}.mp4" -f wav -vn -ac 1 -ab 16k -ar 22050 -loglevel error "{out}"',
                    shell=True)
if __name__ == '__main__':
    shard, nshard = int(sys.argv[1]), int(sys.argv[2])
    ids = open(DS+'/clip_ids.txt').read().split()
    ids = ids[shard::nshard]
    print(f'shard {shard}/{nshard}: {len(ids)} clips', flush=True)
    with Pool(32) as p:
        p.map(extract, ids)
    print(f'shard {shard} done', flush=True)
