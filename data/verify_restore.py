import os, subprocess, random
import cv2
cv2.setNumThreads(0)
DS='/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset/vast27m_150k'
CL, WV = DS+'/clips', DS+'/audios_wav'

def probe(p, stream):  # stream 'a' or 'v'
    r=subprocess.run(f'ffprobe -v error -select_streams {stream}:0 -show_entries stream=codec_name -of csv=p=0 "{p}"',
                     shell=True, capture_output=True, text=True)
    return r.stdout.strip()
def cv2_read(p):
    cap=cv2.VideoCapture(p); tf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); ret,frame=cap.read(); cap.release()
    return (ret and frame is not None), tf

# 1) the known re-encoded clip
print("=== known restored clip --GUsNS6Z38.57 ===")
cid='--GUsNS6Z38.57'; p=f'{CL}/{cid}.mp4'
vok,tf=cv2_read(p)
print(f"  video codec={probe(p,'v')} audio codec={probe(p,'a')} | cv2 reads video={vok} frames={tf}")

# 2) sample 300 clips: how many now have audio + video intact
print("\n=== sample 300 clips ===")
files=sorted(os.listdir(CL))[:300]
have_a=have_v=cv2ok=0
for f in files:
    if probe(f'{CL}/{f}','a'): have_a+=1
    vc=probe(f'{CL}/{f}','v')
    if vc=='h264': have_v+=1
    ok,_=cv2_read(f'{CL}/{f}')
    if ok: cv2ok+=1
print(f"  have audio stream: {have_a}/300")
print(f"  video=h264:        {have_v}/300")
print(f"  cv2 reads video:   {cv2ok}/300")

# 3) full count: clips with audio stream across ALL (sampled every 200th for speed) + a restored-specific set
print("\n=== restored clips deep check (10 that were re-encoded = had no audio before) ===")
# find clips that are h264 AND now have audio via aac (restored ones use aac; originals often aac too, so just verify audio present + video ok)
allf=sorted(os.listdir(CL))
checked=0
for f in allf:
    p=f'{CL}/{f}'
    ac=probe(p,'a')
    if ac=='aac':  # restored ones are aac; check a handful
        vok,tf=cv2_read(p)
        aud_wav=os.path.exists(f'{WV}/{f[:-4]}.wav')
        print(f"  {f[:-4]}: audio={ac} video_ok={vok} frames={tf} wav_exists={aud_wav}")
        checked+=1
        if checked>=8: break
