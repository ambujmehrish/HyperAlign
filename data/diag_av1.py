import os, subprocess
from collections import Counter
import cv2
cv2.setNumThreads(0)
CL='/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset/vast27m_150k/clips'

def codec(p):
    r=subprocess.run(f'ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "{p}"',
                     shell=True, capture_output=True, text=True)
    return r.stdout.strip()

def cv2_read(p):
    cap=cv2.VideoCapture(p); tf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret,frame=cap.read(); cap.release()
    return (ret and frame is not None), tf

# sample 300 clips -> codec distribution + find one AV1
files=sorted(os.listdir(CL))[:300]
codecs=Counter(); av1=None; h264ok=None
for f in files:
    c=codec(f'{CL}/{f}'); codecs[c]+=1
    if c=='av1' and av1 is None: av1=f
    if c=='h264' and h264ok is None: h264ok=f
print('codec dist (300 sample):', dict(codecs))

# sanity: cv2 reads an h264 clip
if h264ok:
    ok,tf=cv2_read(f'{CL}/{h264ok}'); print(f'cv2 read H264 clip {h264ok}: ok={ok} frames={tf}')

# the AV1 test
if av1:
    print(f'\nAV1 clip: {av1}')
    ok,tf=cv2_read(f'{CL}/{av1}')
    print(f'  [1] cv2 read ORIGINAL AV1: ok={ok} frames={tf}  (expect ok=False)')
    out='/tmp/reenc_h264.mp4'
    r=subprocess.run(f'ffmpeg -y -nostdin -i "{CL}/{av1}" -c:v libx264 -preset fast -crf 20 -an "{out}" -loglevel error',
                     shell=True)
    sz=os.path.getsize(out) if os.path.exists(out) else 0
    print(f'  [2] ffmpeg AV1->H264 re-encode: rc={r.returncode} size={sz}')
    ok2,tf2=cv2_read(out)
    print(f'  [3] cv2 read RE-ENCODED H264: ok={ok2} frames={tf2}  (expect ok=True)')
    print(f'\n  VERDICT: {"SUCCESS - re-encoding makes AV1 cv2-readable" if ok2 and not ok else "needs review"}')
else:
    print('no AV1 found in first 300 (sample too small)')
