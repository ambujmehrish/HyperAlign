import os, subprocess
DS='/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset/vast27m_150k'
CL, WV = DS+'/clips', DS+'/audios_wav'
clips=['DYY8KovKO3M.52','J9dUhX5qC-4.16','PTnWkQoD4FE.495','eqAxonbXylc.56',
       'f28ru88Ri0k.34','rr5gcGAGFlw.46','uFCptSipOAs.53']
methods={
  'pcm_explicit': '-vn -acodec pcm_s16le -ac 1 -ar 22050',
  'map_audio'   : '-map 0:a:0 -vn -acodec pcm_s16le -ac 1 -ar 22050',
  'genpts'      : '-fflags +genpts -vn -acodec pcm_s16le -ac 1 -ar 22050',
  'async'       : '-vn -af aresample=async=1 -acodec pcm_s16le -ac 1 -ar 22050',
}
best={}
for cid in clips:
    print(f'\n{cid}:')
    bestsz=0; bestm=None
    for name,opts in methods.items():
        out=f'/tmp/rec_{name}.wav'
        if os.path.exists(out): os.remove(out)
        subprocess.call(f'ffmpeg -y -nostdin -i "{CL}/{cid}.mp4" {opts} "{out}" -loglevel error',shell=True)
        sz=os.path.getsize(out) if os.path.exists(out) else 0
        print(f'  {name}: size={sz}')
        if sz>bestsz: bestsz=sz; bestm=name
    best[cid]=(bestm,bestsz)
    # if a method recovered real audio (>1000 bytes), write it to the real wav dir
    if bestsz>1000:
        opts=methods[bestm]
        subprocess.call(f'ffmpeg -y -nostdin -i "{CL}/{cid}.mp4" {opts} "{WV}/{cid}.wav" -loglevel error',shell=True)
        print(f'  -> RECOVERED via {bestm} ({bestsz} bytes) -> wrote {cid}.wav')
print('\n=== SUMMARY ===')
rec=sum(1 for m,s in best.values() if s>1000)
print(f'  recovered: {rec}/{len(clips)}')
for cid,(m,s) in best.items(): print(f'  {cid}: best={m} size={s} {"OK" if s>1000 else "still empty"}')
