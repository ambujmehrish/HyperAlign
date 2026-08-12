import os, sys, subprocess
from multiprocessing import Pool
from collections import Counter
import cv2
cv2.setNumThreads(0)
CL='/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset/vast27m_150k/clips'

def get_codec(p):
    r=subprocess.run(f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "{p}"',
                     shell=True, capture_output=True, text=True)
    return r.stdout.strip()

def cv2_ok(p):
    cap=cv2.VideoCapture(p); ret,frame=cap.read(); cap.release()
    return ret and frame is not None

def process(fname):
    p=f'{CL}/{fname}'
    try:
        if get_codec(p)!='av1':
            return 'skip_nonav1'
    except Exception:
        return 'probe_err'
    tmp=f'{CL}/.tmp_{fname}'
    r=subprocess.run(f'ffmpeg -y -nostdin -i "{p}" -c:v libx264 -preset fast -crf 20 -an "{tmp}" -loglevel error',
                     shell=True)
    if r.returncode!=0 or not os.path.exists(tmp) or os.path.getsize(tmp)<1000:
        if os.path.exists(tmp): os.remove(tmp)
        return 'ffmpeg_fail'
    if not cv2_ok(tmp):            # verify cv2 can now read it BEFORE replacing
        os.remove(tmp); return 'verify_fail'
    os.replace(tmp, p)            # atomic in-place replace
    return 'converted'

if __name__=='__main__':
    shard, nshard = int(sys.argv[1]), int(sys.argv[2])
    files=[f for f in os.listdir(CL) if f.endswith('.mp4')]
    files=sorted(files)[shard::nshard]
    print(f'shard {shard}/{nshard}: {len(files)} clips to scan', flush=True)
    res=Counter(); done=0
    with Pool(32) as pool:
        for r in pool.imap_unordered(process, files, chunksize=8):
            res[r]+=1; done+=1
            if done % 5000 == 0:
                print(f'  [{done}/{len(files)}] {dict(res)}', flush=True)
    print(f'shard {shard} DONE: {dict(res)}', flush=True)
