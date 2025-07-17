# Prepare the sportsMOT dataset (just the football clips) in the style of the MSMT17 dataset.

import os, sys, argparse, random, json
from pathlib import Path
from collections import defaultdict
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mot_root',   type=Path, required=True,
                   help='Path to SportsMOT root (contains dataset/, splits_txt/)')
    p.add_argument('--out_root',   type=Path, required=True,
                   help='Output dir for crops + lists (e.g. datasets/SportsMOT_Football)')
    p.add_argument('--holdout',    type=float, default=0.1,
                   help='Fraction of PIDs to hold out as validation')
    args = p.parse_args()
    return args

def read_football_clips(mot_root):
    # splits_txt/football.txt lists clip names like v_HdiyOtliFiw_c010
    txt = mot_root/'splits_txt'/'football.txt'
    football_clips = [line.strip() for line in txt.read_text().splitlines() if line.strip()]
    print("Number of football clips is ",len(football_clips))
    # Remove the ones in the test split
    test_txt = mot_root/'splits_txt'/'test.txt'
    test_clips = [line.strip() for line in test_txt.read_text().splitlines() if line.strip()]
    football_clips = [c for c in football_clips if c not in test_clips]
    print("Number of football clips after removing test is",len(football_clips))
    return football_clips

def load_tracks(mot_root, clips):
    """
    Returns:
      tracks: dict[(clip_name,pid)] -> list of (img_path, (x,y,w,h))
      match_map: dict[(clip_name,pid)] -> match_id_str
    """
    tracks = defaultdict(list)
    match_map = {}
    for clip in clips:
        print(clip)
        # detect train vs val folder
        for split in ('train','val'):
            base = mot_root/'dataset'/split/clip
            gt = base/'gt'/'gt.txt'
            img_dir = base/'img1'
            if not gt.exists():
                continue
            match_id = clip.split('_c')[0]  # v_<matchID>
            for line in gt.read_text().splitlines():
                f,pid,x,y,w,h,mark,cls,_ = map(str.strip,line.split(','))
                if mark!='1' or cls!='1': continue
                pid = int(pid)
                fnum = int(f)
                img_path = img_dir/f"{int(f):06d}.jpg"
                tracks[(clip,pid)].append((img_path, (int(x),int(y),int(w),int(h))))
                match_map[(clip,pid)] = match_id
    return tracks, match_map

def subsample_frames(frames):
    L = len(frames)
    B = min(8, max(4, round(L/100))) # Every 4 seconds on average
    # split into B bins
    bins = []
    step = L / B
    for i in range(B):
        idx = int(i*step)
        bins.append(frames[idx: idx+int(step)] or frames[idx:idx+1])
    # pick one random from each
    sel = [random.choice(b) for b in bins]
    return sel

def main():
    args = parse_args()
    random.seed(0)

    clips     = read_football_clips(args.mot_root)
    tracks, match_map = load_tracks(args.mot_root, clips)
    # assign global PIDs
    all_keys = sorted(tracks.keys())
    gid_map = {k:i for i,k in enumerate(all_keys)}
    # print("gid_map",gid_map)
    # split train vs val gids
    all_gids = list(gid_map.values())
    nval = max(1, int(len(all_gids)*args.holdout))
    # print(nval)
    # print(len(all_gids))
    val_gids = set(random.sample(all_gids, nval))
    print("Number of desired val gids is",nval)
    train_gids = set(all_gids) - val_gids
    gid_map_train = {k: i for i, k in enumerate([k for k in gid_map if gid_map[k] in train_gids])}
    gid_map_val = {k: i for i, k in enumerate([k for k in gid_map if gid_map[k] in val_gids])}
    print("Total train gids",len(gid_map_train.keys()))
    print("Total validation gids",len(gid_map_val.keys())) # These will be made query and gallery sets later.
    assert len(gid_map_train) == len(train_gids)
    assert len(gid_map_val) == len(val_gids)
    # prepare output dirs
    out = args.out_root
    (out/'train').mkdir(parents=True, exist_ok=True)
    (out/'val').mkdir(parents=True, exist_ok=True)

    recs = {'train':[], 'val': defaultdict(list)}  # list of (relpath, pid, camid)
    # camid=0 for train. For val we'll split query/gallery later.
    for (clip,pid), items in tracks.items():
        gid = gid_map[(clip,pid)]
        split = 'val' if gid in val_gids else 'train'
        if split == 'train':
            remapped_gid = gid_map_train[(clip,pid)]
        elif split == 'val':
            remapped_gid = gid_map_val[(clip,pid)]
        sub = out/split/str(remapped_gid)
        sub.mkdir(parents=True, exist_ok=True)

        # get frame-level crops
        sel = subsample_frames(items)
        print("Sampled",len(sel),"identities for",remapped_gid)
        for idx, (img_path, (x,y,w,h)) in enumerate(sel):
            im = Image.open(img_path).crop((x,y, x+w, y+h))
            fname = f"img_{idx:06d}.jpg"
            savep = sub/fname
            # print("Theoritically saving")
            im.save(savep)
            rel = f"{split}/{remapped_gid}/{fname}"
            camid = 0 if split=='train' else 0  # temp 0 for val
            if split == 'train':
                recs['train'].append((rel, remapped_gid, 0))
            else:  # split == 'val'
                recs['val'][remapped_gid].append(rel)   # only store rel; camid will be assigned later

    # write list_train.txt
    with open(out/'list_train.txt','w') as f:
        for rel, gid, camid in recs['train']:
            f.write(f"{rel} {gid} {camid}\n")

    # for val: pick one query per pid, rest gallery
    with open(out/'list_query_val.txt','w') as fq, open(out/'list_gallery_val.txt','w') as fg:
        for gid, rels in recs['val'].items():
            # pick one random query
            q = random.choice(rels)
            fq.write(f"{q} {gid} 1\n")
            # rest are gallery
            for g in rels:
                if g == q: 
                    continue
                fg.write(f"{g} {gid} 0\n")


    # dump mapping (for advanced sampler if i want to use it)
    # 5. Dump pid→match mapping correctly
#    build inverse of your gid_map {(clip,pid):gid}
    # inv_gid_map = {gid:key for key,gid in gid_map.items()}
    # with open(out/'pid2match.json','w') as f:
    #     json.dump(
    #         { str(gid): match_map[inv_gid_map[gid]] 
    #         for gid in inv_gid_map },
    #         f, indent=2
    #     )
    info = {}
    for (clip, pid), gid in gid_map_train.items():
        info[str(gid)] = {
            'match': match_map[(clip, pid)],
            'clip': clip
        }
    with open(out/'gid2info.json', 'w') as f:
        json.dump(info, f, indent=2)
    # The above basically gives the match id and clip id strings, given the gid of train.

    print("Generated crops + list_train.txt + list_query_val.txt + list_gallery_val.txt")

if __name__=='__main__':
    main()

