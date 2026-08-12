import os, subprocess, random
import soundfile as sf
DS='/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset/vast27m_150k'
CL, WV = DS+'/clips', DS+'/audios_wav'

clips=set(f[:-4] for f in os.listdir(CL) if f.endswith('.mp4'))
wavs=set(f[:-4] for f in os.listdir(WV) if f.endswith('.wav'))
print(f"clips: {len(clips)}  wavs: {len(wavs)}  clips-without-wav: {len(clips-wavs)}")

# 1) validity of fresh wavs (22050 mono, readable)
print("\n-- fresh wav validity (12 random) --")
sr_bad=0
for f in random.Random(0).sample(list(wavs), 12):
    try:
        data, sr = sf.read(f'{WV}/{f}.wav')
        ch = 'mono' if data.ndim==1 else f'{data.shape[1]}ch'
        if sr!=22050 or ch!='mono': sr_bad+=1
        print(f"  {f}: sr={sr} {ch} dur={len(data)/sr:.1f}s")
    except Exception as e:
        print(f"  {f}: ERROR {e}"); sr_bad+=1
print(f"  -> {12-sr_bad}/12 correct (22050 mono)")

# 2) the clips WITHOUT wav: do they have audio? how long? recoverable?
missing=sorted(clips-wavs)
print(f"\n-- {len(missing)} clips WITHOUT wav: do they have audio? --")
def astream(p):
    r=subprocess.run(f'ffprobe -v error -select_streams a -show_entries stream=codec_name,duration -of csv=p=0 "{p}"',shell=True,capture_output=True,text=True)
    return r.stdout.strip()
recoverable=0
for cid in missing:
    info=astream(f'{CL}/{cid}.mp4')
    # try extracting to check actual size
    tmp='/tmp/chk.wav'
    subprocess.call(f'ffmpeg -y -nostdin -i "{CL}/{cid}.mp4" -f wav -vn -ac 1 -ar 22050 -loglevel error {tmp}',shell=True)
    sz=os.path.getsize(tmp) if os.path.exists(tmp) else 0
    has = 'AUDIO' if info else 'NO-AUDIO-STREAM'
    if info and sz>44: recoverable+=1
    print(f"  {cid}: {has} probe='{info}' extracted_size={sz}")
print(f"\n  recoverable (has some audio): {recoverable}/{len(missing)}")
