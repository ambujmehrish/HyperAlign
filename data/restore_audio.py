import os, sys, subprocess
from multiprocessing import Pool
from collections import Counter
DS='/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset/vast27m_150k'
CL, WV = DS+'/clips', DS+'/audios_wav'

def has_audio(p):
    r=subprocess.run(f'ffprobe -v error -select_streams a -show_entries stream=codec_type -of csv=p=0 "{p}"',
                     shell=True, capture_output=True, text=True)
    return 'audio' in r.stdout

def restore(fname):
    cid=fname[:-4]; clip=f'{CL}/{fname}'; wav=f'{WV}/{cid}.wav'
    if has_audio(clip):        return 'has_audio'     # original clip, already has audio -> leave as-is
    if not os.path.exists(wav): return 'no_wav'       # truly silent (the ~20) -> nothing to restore
    tmp=f'{CL}/.mux_{fname}'
    # keep H.264 video byte-for-byte (copy); add the wav as an AAC audio track
    r=subprocess.run(f'ffmpeg -y -nostdin -i "{clip}" -i "{wav}" -c:v copy -c:a aac -b:a 128k '
                     f'-map 0:v:0 -map 1:a:0 -shortest "{tmp}" -loglevel error', shell=True)
    if r.returncode!=0 or not os.path.exists(tmp) or os.path.getsize(tmp)<1000:
        if os.path.exists(tmp): os.remove(tmp)
        return 'mux_fail'
    if not has_audio(tmp):
        os.remove(tmp); return 'verify_fail'
    os.replace(tmp, clip)
    return 'restored'

if __name__=='__main__':
    shard, nshard = int(sys.argv[1]), int(sys.argv[2])
    files=sorted(f for f in os.listdir(CL) if f.endswith('.mp4'))[shard::nshard]
    print(f'shard {shard}/{nshard}: {len(files)} clips to scan', flush=True)
    res=Counter(); done=0
    with Pool(32) as p:
        for r in p.imap_unordered(restore, files, chunksize=8):
            res[r]+=1; done+=1
            if done%10000==0: print(f'  [{done}/{len(files)}] {dict(res)}', flush=True)
    print(f'shard {shard} DONE: {dict(res)}', flush=True)
