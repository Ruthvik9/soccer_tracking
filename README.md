# Football Player Re-Identification & Tracking Pipeline

This repository provides the implementation to try and track soccer players in video clips using a **SportsMOT fine-tuned CLIP-ReID model** and **BoT-SORT**.
---
Brief outline - 1) Finetune CLIP-ReID on the SportsMOT dataset (after converting it to a ReID format) and incorporate that and your YOLOv11 into BoT-SORT

![Input video](15sec_input_720p.gif)

![Output](15sec_output_720p_tight.gif)

## Setup

All checkpoints are available at https://drive.google.com/drive/folders/1Htbxdq-VQULw2s2fUWWKfcTuhcYoUeOX?usp=sharing

### Clone the repo and install the environments

```bash
git clone https://github.com/Ruthvik9/soccer_tracking.git
conda env create -f clipreid.yml # For finetuning ClipReID
conda env create -f track.yml # for tracking inference
```

### Download and Organize SportsMOT Dataset (if you want to finetune)

Ensure the dataset is structured as follows (anywhere, we'll later give this path to the ReID style setup code):

```text
/path/to/SportsMOT/
  splits_txt/football.txt
  dataset/
    train/<clip>/gt/gt.txt/img1/*.jpg
    val/<clip>/gt/...
```



### MSMT17 Pretrained Checkpoint

Download and place the MSMT17 checkpoint at:

```text
clip_reid/checkpoints/MSMT17_clipreid_ViT-B-16_60.pth
```

---

## Data Preparation

Use the `prepare_sportsmot.py` script from clip_reid:

```bash
cd clip_reid
mkdir -p datasets/SportsMOT_Football
python prepare_sportsmot.py \
  --mot_root /path/to/SportsMOT \
  --out_root datasets/SportsMOT_Football \
  --holdout 0.1 # Use 10% of unique ids for query/gallery inference
```

Note: Since SportsMOT has videos/tracks of players and not images, I first converted the dataset into a format suitable for ReID.
For this, I create 'bins' and sample one frame for every bin (no. of bins is a hyperparam). This helps capture the same player under different poses and viewpoints etc.
Some examples are shown below. I then created the query and gallery lists from the 10% held-out val set and remapped all the ids to 0....<num_identities-1> since the ClipReID head ideally expects that 
(in the train set.) 
The log files for this are available in this repo.

<img width="38" height="80" alt="image" src="https://github.com/user-attachments/assets/873d496d-c8dd-4993-bab6-bd0c5db15d26" />
<img width="42" height="128" alt="image" src="https://github.com/user-attachments/assets/2c0565e2-ea89-41ba-9874-b14b2eb6690b" />
<img width="32" height="131" alt="image" src="https://github.com/user-attachments/assets/7e369c80-a694-4e91-8cc8-bd04adf29672" />
<img width="34" height="77" alt="image" src="https://github.com/user-attachments/assets/0a574313-fcc2-4016-9c08-d7daeeff3af3" />
<img width="20" height="60" alt="image" src="https://github.com/user-attachments/assets/efe6da5d-68fe-4175-9653-4695d8b8eaa9" />
<img width="53" height="106" alt="image" src="https://github.com/user-attachments/assets/9f61d01e-4060-45af-9a08-3f5001cdb758" />
<img width="48" height="72" alt="image" src="https://github.com/user-attachments/assets/c9c3371b-a314-4f74-80f9-a77bccc3b4f7" />
<img width="32" height="77" alt="image" src="https://github.com/user-attachments/assets/3719d65b-7500-419c-9963-16aef362c5cc" />
<img width="34" height="72" alt="image" src="https://github.com/user-attachments/assets/38a0d9dc-f76a-4a16-b38c-75808e9039f7" />
<img width="38" height="112" alt="image" src="https://github.com/user-attachments/assets/3a01dc94-4199-4a82-87f3-86d221d2851a" />
<img width="50" height="102" alt="image" src="https://github.com/user-attachments/assets/96afaca7-fa40-4d7d-a723-d4e8957c8382" />
<img width="59" height="112" alt="image" src="https://github.com/user-attachments/assets/44d17553-d057-41d3-8b5c-3c9d671962fd" />
<img width="72" height="103" alt="image" src="https://github.com/user-attachments/assets/b6f12bf8-9a52-4602-8d1b-45991d64e31d" />
<img width="38" height="82" alt="image" src="https://github.com/user-attachments/assets/e4f16b43-2b7c-40d2-8c31-70f6d5090ff9" />
<img width="28" height="74" alt="image" src="https://github.com/user-attachments/assets/a96cb986-3e9f-43d1-9831-b66d4682c027" />
<img width="46" height="65" alt="image" src="https://github.com/user-attachments/assets/f542913f-9ba2-4de2-ae5e-34e30ed3dfa7" />
<img width="48" height="82" alt="image" src="https://github.com/user-attachments/assets/f6d13e33-2139-4415-a599-0bdcbd9458d5" />
<!---
<img width="63" height="82" alt="image" src="https://github.com/user-attachments/assets/4ce8ff4b-4163-411c-a964-a889dfb646ea" />
<img width="22" height="62" alt="image" src="https://github.com/user-attachments/assets/797e7e85-35f2-4af2-affb-fcf40630fc1a" />
<img width="28" height="90" alt="image" src="https://github.com/user-attachments/assets/471c73b7-e271-497b-9def-711dd3d38751" />
<img width="31" height="79" alt="image" src="https://github.com/user-attachments/assets/a9f6fa02-8f62-47b5-877a-351644ecf8e2" />
--->





Output structure after running the above code looks like:

```text
datasets/SportsMOT_Football/ # In clip_reid root
  train/0...Ntrain-1/*.jpg
  val/0...Nval-1/*.jpg
  list_train.txt
  list_query_val.txt
  list_gallery_val.txt
  gid2info.json # maps global id to clip and match in SportsMOT. Useful later for the match-aware sampler.
```

---

## Some notes
### Match-Aware Sampler

The Match-Aware Sampler is a custom extension of the standard “Random Identity Sampler” used in person Re-ID training. Its core functionality is:

i) Batch Construction

Like RandomIdentitySampler, it selects N identities per batch and K examples of each identity, yielding a batch of size N×K.

ii) Match-Aware Constraint (the main part of how it's different from RandomIdentitySampler)

In SportsMOT I noticed that each clip comes from a particular match and for some reason there are multiple clips from the same match in the same split.
This is fine for tracking but can pose problems in ReID. Because player IDs reset in every clip **even for the same real person**.

Without special handling, the default sampler can sample two crops from different clips of the same match and treat them as a “negative” pair, even though they might show the same player. That injects false negatives into the triplet loss.

The Match-Aware Sampler avoids this by:

Allowing multiple PIDs from the same clip (they truly are different players).

Allowing PIDs from different matches (most probably negatives, can't really avoid this unless we globally reannotate the dataset).

Forbidding PIDs from **different clips of the same match (potential false negatives).

---

### Augmentation Pipeline

Added to the default training transforms in `clipreid/dataset.py` some extra steps including:

* Color jittering, gamma adjustment
* Gaussian blur

This enhances model robustness through photometric variations so that the model doesn't overfit to jersey colors etc. (that's the hope atleast :))

---

## Finetuning

To fine-tune the model, run: (evaluation metrics are logged automatically out of the box from ClipReID's original implementation.)

```bash
python train_clipreid.py --config_file configs/person/vit_clipreid.yml # Already modified code to start training from MSMT17 checkpoint and accept the SportsMOT dataset and the sampling logic.
```

---

## Tracking with BoT-SORT

To perform inference with BoT-SORT, simply run the code
```bash
cd botsort; mkdir -p models
python infer_botsort.py # After placing yolov11 and clip_reid_finetuned checkpoints in botsort/models/
```
The above code does the following -

i) Loads YOLOv11 detector from models/yolov11_best.pt (the checkpoint you provided) (not the default used by BoTSORT)<br />
(Note: Like you said, the checkpoint is just a basic one, I see many spurious detections)<br />
ii) CLIP-ReID encoder from models/clip_reid_finetuned.pth, wrapped in a small loader (for feature extraction)<br />
iii) Transforms - Deterministic resize to 256×128 + normalize with my training mean/std (to match the finetuning config when finetuning clipreid)<br />
iv) All key thresholds (detection conf, IoU gate, appearance gate, new-track conf, buffer length, score fusion) set via a single args = SimpleNamespace(...)<br />
v) Replaces FastReID in the default BoT-SORT with a custom CLIPInterface that crops each box, runs it through CLIP-ReID, and returns embeddings<br />
vi) At detection time requests only “player” and “goalkeeper” classes (drops ball, referee which are also classes in your pretrained model.)<br />
vii) Track → tracker.update(detections, frame) applies Kalman+GMC prediction, runs CLIP embeddings for appearance association.<br />

---

## Known Issues & Resolutions

* **CamID in Val:** camid=1 for query, camid=0 for gallery (since same camid for query and gallery results in them being filtered out)
* **Sampler Logic:** Enforced via custom `SportsMOTSampler`. Created a custom sampler to get 'match-aware' samples in each batch.
* (Not including other minor things like shape mismatches after loading MSMT17 pretrained weights (The pretrained MSMT17 checkpoint’s final classifier was built for MSMT’s 1041 train identities), ID continuity that CLIPReID wants, converting SportsMOT into a ReID format, incorporating your yolov11 into BoTSORT etc.)

## Some design choices

1. Training split – combine “train” + “val”, ignore public “test”.
SportsMOT’s test set is huge but unlabeled, so I merged the 45-clip train and 45-clip val partitions into a single training pool. Around 10 % of the player-IDs are set aside as an internal validation fold; this protects us from over-fitting even though we aren’t chasing the official leaderboard.
<br />

2. Frame subsampling – atmost 8 crops per track.
Each football track is ~500 frames long. Cropping every frame would flood the GPU with near-duplicates. I split each track into temporal bins and pick one random frame per bin. This keeps useful pose/zoom variation while cutting redundant images.
<br />

3. Duplicate IDs across clips – handled in the sampler, not the labels.
SportsMOT assigns new person-IDs in every clip, so the same striker can appear with a different PID in another match segment. Instead of re-labelling the dataset, I introduced a match-aware PK sampler (described above) so the triplet loss doesn't see false negatives (not 100% foolproof tho unless we re-label the entire dataset). Cross-entropy still works because labels are unique within the batch.
<br />

4. Increased the allowed duration after which tracks are considered lost
Because of how the same player can come back into the field of view much later.
<br />

5. Query / Gallery from Val Split
Issue: In retrieval evaluation I need query vs gallery cam-IDs to differ so the same‐camera filter doesn’t drop everything. But I also want val to reflect disjoint identities.
Solution: I took the held-out val GIDs (global ids), generated one random query image per GID (camid=1) and put the rest in gallery (camid=0). So that evaluation works exactly like a test set.

## Future Work and Potential Enhancements

Given more time and resources, some things that come to mind are: <br />

i) Investigate better handling of jersey color reliance, which can lead to shortcut learning. <br />
ii) Develop better motion-based features. <br />
iii) Integrate the deconfuse track idea to stabilize identity assignment. <br />
iv) Explore the use of MixSort (by SportsMOT authors) for inference, as it is trained specifically on SportsMOT for tracking. While I take a custom route, MixSort is a strong baseline to benchmark against. <br />
v) Combine SportsMOT with SoccerNet dataset to improve domain generalization. <br />
vi) Integrate auxiliary cues like OCR (jersey numbers) or head/face features, if reliably detected. <br />
vii) The scope of unique IDs in SportsMOT is not global. Nor is it a dataset that can be used for commercial purposes. So maybe getting some custom data annotated? <br />
viii) Investigate including samples from datasets like Market1501 to regularize the model and improve generalization beyond uniforms (proposed by one of the papers) <br />
ix)Explore more modern tracking methods such as TrackTrack or other motion-aware trackers. <br />
x)Tune tracking hyperparameters (e.g., detection confidence, match thresholds) to suppress spurious detections.
xi) Explore end to end tracking approaches along with TbD. <br />

