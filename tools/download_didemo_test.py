"""Re-fetch the DiDeMo retrieval-test videos from the YFCC100M AWS mirror.

Why this exists
---------------
GRAM's own released checkpoint scores 3.6 R@1 below GRAM's published table in this harness. Every
non-video explanation has been eliminated: the test annotations are md5-identical to upstream's,
default_model_cfg.json is identical, utils/volume.py is upstream plus additions with no deletions,
the eval-path feature extraction is unchanged, and GRAM's own unmodified evaluation_mm.py
reproduces our numbers to within 0.2 (GRAM_UPSTREAM_EVAL=1). What is left is the video files.

The residual gap orders by how much the videos must be reconstructed from third-party sources:
msrvtt (fixed archive) -1.0/-2.3, didemo -3.4, activitynet -2.7/-3.7, vatex -6.7/-7.2. DiDeMo is
the cheap test of that hypothesis, because its videos have a STABLE mirror: the upstream repo
(github.com/LisaAnne/LocalizingMoments) says to prefer AWS "as many videos have been deleted off
of Flickr since the dataset was collected". So a Flickr-sourced copy can differ from the canonical
one, and re-fetching is a controlled experiment rather than a guess.

If re-fetching DiDeMo moves GRAM's checkpoint from 50.8 toward 54.2, provenance is confirmed and
VATEX (biggest gap, and also mirrored -- CVDF hosts Kinetics-600 as pre-cut ~10s clips) is worth
the transfer. If it does not move, the hypothesis is wrong and nothing further needs downloading.

Inputs
------
Upstream's id->hash table and split files, from a shallow clone of the DiDeMo repo:

    git clone --depth 1 https://github.com/LisaAnne/LocalizingMoments.git /path/to/didemo_repo

Usage
-----
    python3 tools/download_didemo_test.py \
        --didemo-repo /path/to/didemo_repo \
        --out $HA_WORK/datasets_fresh/DiDeMo/videos_flat

    # inspect what would happen without transferring anything
    python3 tools/download_didemo_test.py --didemo-repo ... --out ... --dry-run

Only the clips named by datasets/annotations/didemo/descs_ret_test.json are fetched (~1003), not
the full ~10k dataset. Output files are named "<video_id>.mp4" using OUR annotation's video_id, so
the existing data_cfg vision path can be pointed straight at --out.
"""
import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# The YFCC100M AWS mirror. Hash h is sharded as h[:3]/h[3:6]/h.mp4 -- taken verbatim from
# upstream's download_videos_AWS.py (which is Python 2 and cannot be run as-is: it uses urllib2
# and leaves an unconditional pdb.set_trace() inside the download loop).
S3 = 'https://multimedia-commons.s3-us-west-2.amazonaws.com/data/videos/mp4/{}/{}/{}.mp4'


def s3_url(h):
    return S3.format(h[:3], h[3:6], h)


def load_hash(repo):
    """flickr id -> yfcc100m hash"""
    path = os.path.join(repo, 'data', 'yfcc100m_hash.txt')
    if not os.path.exists(path):
        sys.exit(f"missing {path}\n  clone it: git clone --depth 1 "
                 f"https://github.com/LisaAnne/LocalizingMoments.git {repo}")
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def load_upstream_names(repo):
    """Every video filename upstream knows about, e.g. '26292851@N04_4253489686_265c3c8051.m4v'.

    Needed because our annotation's video_id may carry no extension while the flickr id -- the
    only thing that maps into the hash table -- is the middle underscore-separated field.
    """
    names = set()
    for split in ('test', 'val', 'train'):
        p = os.path.join(repo, 'data', f'{split}_data.json')
        if os.path.exists(p):
            names.update(r['video'] for r in json.load(open(p)))
    return names


def flickr_id(name):
    """'26292851@N04_4253489686_265c3c8051.m4v' -> '4253489686' (upstream's video.split('_')[1])."""
    parts = os.path.splitext(name)[0].split('_')
    return parts[1] if len(parts) > 1 else None


def wanted_ids(ann_path):
    ann = json.load(open(ann_path))
    # keep annotation order stable but unique
    seen, out = set(), []
    for a in ann:
        v = a['video_id']
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def fetch(url, dest, timeout=120, retries=3):
    for attempt in range(retries):
        try:
            tmp = dest + '.part'
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, 'wb') as w:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    w.write(chunk)
            if os.path.getsize(tmp) == 0:
                os.remove(tmp)
                return 'empty'
            os.replace(tmp, dest)
            return 'ok'
        except Exception as e:
            if attempt == retries - 1:
                return f'{type(e).__name__}: {e}'
    return 'unreachable'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--didemo-repo', required=True,
                    help='shallow clone of github.com/LisaAnne/LocalizingMoments')
    ap.add_argument('--out', required=True, help='destination directory for <video_id>.mp4')
    ap.add_argument('--ann', default=os.path.join(REPO, 'datasets/annotations/didemo/descs_ret_test.json'))
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--dry-run', action='store_true', help='resolve URLs, transfer nothing')
    args = ap.parse_args()

    ids = wanted_ids(args.ann)
    hashes = load_hash(args.didemo_repo)
    upstream = load_upstream_names(args.didemo_repo)
    by_stem = {os.path.splitext(n)[0]: n for n in upstream}
    by_flickr = {}
    for n in upstream:
        fid = flickr_id(n)
        if fid:
            by_flickr.setdefault(fid, n)

    print(f"  annotation      : {args.ann}")
    print(f"  want            : {len(ids)} clips")
    print(f"  upstream names  : {len(upstream)}   hash table: {len(hashes)}")

    jobs, unresolved = [], []
    for vid in ids:
        # our video_id may be the full upstream filename, its stem, or just the flickr id
        name = vid if vid in upstream else by_stem.get(os.path.splitext(vid)[0])
        fid = flickr_id(name or vid) or (vid if vid.isdigit() else None)
        h = hashes.get(fid) if fid else None
        if not h:
            unresolved.append(vid)
            continue
        jobs.append((vid, s3_url(h)))

    print(f"  resolved        : {len(jobs)}/{len(ids)}"
          + (f"   UNRESOLVED {len(unresolved)}" if unresolved else ""))
    if unresolved:
        print(f"    first few: {unresolved[:5]}")
        print("    ^ our video_id does not map into upstream's hash table. Fix the id mapping "
              "before trusting a partial download -- a short gallery is not comparable.")

    if args.dry_run:
        for vid, url in jobs[:5]:
            print(f"    {vid} -> {url}")
        print(f"  dry run: would fetch {len(jobs)} files into {args.out}")
        return 0

    os.makedirs(args.out, exist_ok=True)
    todo = [(v, u) for v, u in jobs if not os.path.exists(os.path.join(args.out, f'{v}.mp4'))]
    print(f"  already present : {len(jobs) - len(todo)}\n  to fetch        : {len(todo)}\n")

    results = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(fetch, u, os.path.join(args.out, f'{v}.mp4')): v for v, u in todo}
        for fut in as_completed(futs):
            v = futs[fut]
            results[v] = fut.result()
            done += 1
            if done % 25 == 0 or done == len(todo):
                ok = sum(1 for r in results.values() if r == 'ok')
                sys.stdout.write(f"\r  {done}/{len(todo)}  ok={ok}")
                sys.stdout.flush()
    print()

    failed = {v: r for v, r in results.items() if r != 'ok'}
    have = sum(1 for v, _ in jobs if os.path.exists(os.path.join(args.out, f'{v}.mp4')))
    print(f"\n  on disk now     : {have}/{len(ids)} of the annotated test clips")
    if failed:
        print(f"  failed          : {len(failed)}")
        for v, r in list(failed.items())[:8]:
            print(f"    {v}: {r}")
        with open(os.path.join(args.out, 'FAILED.txt'), 'w') as f:
            for v, r in failed.items():
                f.write(f'{v}\t{r}\n')
        print(f"  -> {os.path.join(args.out, 'FAILED.txt')}")
    if have < len(ids):
        print("\n  NOTE: a gallery smaller than the annotation is NOT comparable to the paper -- "
              "fewer distractors inflates recall. Either recover the rest (upstream hosts 13 clips "
              "that were never on AWS) or filter the annotation and footnote the size.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
