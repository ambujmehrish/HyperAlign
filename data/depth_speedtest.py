import time, os, sys, numpy as np, torch, cv2
from PIL import Image
cv2.setNumThreads(0)
sys.path.insert(0, os.path.dirname(__file__))
from data.chronodepth_pipeline import ChronoDepthPipeline

MODEL = '/leonardo_work/IscrC_GMEG/anag0000/HyperAlign/pretrained_weights/chronodepth'
CL = '/leonardo_scratch/large/userexternal/anag0000/Multimodal_HyperGraph_Dataset/vast27m_150k/clips'
DENOISE, NFRAMES = 10, 8   # GRAM: denoise_steps=10; NFRAMES = frames/clip for depth

print("loading ChronoDepth (local, fp16)...", flush=True)
t0 = time.time()
pipe = ChronoDepthPipeline.from_pretrained(MODEL, torch_dtype=torch.float16).to('cuda')
try: pipe.enable_xformers_memory_efficient_attention()
except Exception: pass
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

def read_frames(path, n):
    cap = cv2.VideoCapture(path); tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if tf < 1: cap.release(); return None
    idxs = np.linspace(0, tf-1, n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ret, f = cap.read()
        if ret and f is not None: frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames) if frames else None

clips = sorted(f for f in os.listdir(CL) if f.endswith('.mp4'))[:6]
times = []
for cid in clips:
    rgb = read_frames(f'{CL}/{cid}', NFRAMES)
    if rgb is None: print(f"  {cid}: no frames"); continue
    imgs = [Image.fromarray(rgb[i]) for i in range(rgb.shape[0])]
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        out = pipe(imgs, num_frames=len(imgs), num_inference_steps=DENOISE,
                   decode_chunk_size=1, motion_bucket_id=127, fps=7, noise_aug_strength=0.0)
    torch.cuda.synchronize(); dt = time.time()-t
    times.append(dt)
    print(f"  {cid}: {len(imgs)} frames -> {dt:.1f}s/clip", flush=True)

if times:
    avg = sum(times[1:])/max(len(times)-1,1)  # skip 1st (warmup)
    print(f"\n=== avg {avg:.1f}s/clip ({NFRAMES} frames, {DENOISE} steps) ===")
    for nclip, tag in [(136694,'136k pretrain'),(1000,'MSR-VTT test')]:
        h = nclip*avg/3600
        print(f"  {tag} ({nclip}): 1 GPU ~{h:.1f}h | 4 GPU ~{h/4:.1f}h | 16 GPU ~{h/16:.1f}h")
