import json, os, copy

H = "/leonardo_work/IscrC_GMEG/anag0000/HyperAlign"
FT = os.path.join(H, "config/gram/finetune_cfg")
ANN = os.path.join(H, "benchmark_eval/smoke_annos")
os.makedirs(ANN, exist_ok=True)
N_VAL = 100

def truncate(path, n, tag):
    obj = json.load(open(path))
    sub = obj[:n] if isinstance(obj, list) else dict(list(obj.items())[:n])
    outp = os.path.join(ANN, tag)
    json.dump(sub, open(outp, "w"))
    return outp, len(sub)

for ds in ["msrvtt", "didemo", "activitynet", "vatex"]:
    ftp = os.path.join(FT, f"retrieval-{ds}_ftsmoke.json")
    ft = json.load(open(ftp))
    base = json.load(open(os.path.join(FT, f"retrieval-{ds}.json")))
    val = copy.deepcopy(base["data_cfg"]["val"][0])
    # resolve base val txt (may be relative to H)
    vtxt = val["txt"]
    if not os.path.isabs(vtxt):
        vtxt = os.path.join(H, vtxt)
    vsmoke, nv = truncate(vtxt, N_VAL, f"ft_{ds}_test{N_VAL}.json")
    val["txt"] = vsmoke
    ft["data_cfg"]["val"] = [val]
    # enable validation (mirror gram_implement_2) — keep new_lr for hgnn
    rc = ft["run_cfg"]
    rc["valid_freq"] = 6
    rc["save_steps"] = 6
    rc["save_best"] = True
    rc["first_eval"] = True
    json.dump(ft, open(ftp, "w"), indent=1)
    print(f"{ds}: val {nv} (task {val.get('task')}) added | num_train_steps={rc.get('num_train_steps')} valid_freq=6 save_best=True")

print("DONE: HyperAlign ftsmoke configs now have validation")
