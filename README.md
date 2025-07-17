# Football Player Re-Identification & Tracking Pipeline

This repository provides the implementation to try and track soccer players in video clips clips using a **SportsMOT fine-tuned CLIP-ReID model** and **BoT-SORT**.

---

![Input video](15sec_input_720.gif)

![Output](15sec_output_720_tight.gif)

## Setup

All checkpoints are available at https://drive.google.com/drive/folders/1Htbxdq-VQULw2s2fUWWKfcTuhcYoUeOX?usp=sharing

### Clone the repo and install the environments

```bash
git clone https://github.com/Ruthvik9/soccer_tracking.git
cd clip_reid
conda env create -f clipreid.yml # For finetuning ClipReID
cd ../botsort
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
The SportsThe log files for this are available in this repo.

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


## Match-Aware Sampler

The Match-Aware Sampler is an extension of the standard “Random Identity Sampler” used in person Re-ID training. Its core functionality is:

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

## Augmentation Pipeline

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
python infer_botsort.py # Place yolov11 and clip_reid_finetuned checkpoints in botsort/models/
```
The above code does the following -

i) Loads YOLOv11 detector from models/yolov11_best.pt (the checkpoint you provided) (not the default used by BoTSORT)
(Note: Like you said, the checkpoint is just a basic one, I see many spurious detections)
ii) CLIP-ReID encoder from models/clip_reid.pth, wrapped in a small loader (for feature extraction)
iii) Transforms - Deterministic resize to 256×128 + normalize with your training mean/std (to match the finetuning config when finetuning clipreid)
iv) All key thresholds (detection conf, IoU gate, appearance gate, new-track conf, buffer length, score fusion) set via a single args = SimpleNamespace(...)
v) Replaces FastReID in the default BoT-SORT with a custom CLIPInterface that crops each box, runs it through CLIP-ReID, and returns embeddings
vi) At detection time requests only “player” and “goalkeeper” classes (drops ball, referee which are also classes.)
vii) Track → tracker.update(detections, frame) applies Kalman+GMC prediction, runs CLIP embeddings for appearance association.

---

## Known Issues & Resolutions

* **CamID in Val:** camid=1 for query, camid=0 for gallery (since same camid for query and gallery results in them being filtered out)
* **Sampler Logic:** Enforced via custom `SportsMOTSampler`. Created a custom sampler to get 'match-aware' samples in each batch.
* (Not including other minor things like shape mismatches after loading MSMT17 pretrained weights (The pretrained MSMT17 checkpoint’s final classifier was built for MSMT’s 1041 train identities), ID continuity that CLIPReID wants, converting SportsMOT into a ReID format, incorporating your yolov11 into BoTSORT etc.)

## Future Work and Potential Enhancements

Given more time and resources, some things that come to mind are:

i) Investigate better handling of jersey color reliance, which can lead to shortcut learning.
ii) Develop better motion-based features.
iii) Integrate the deconfuse track idea to stabilize identity assignment.
iv) Explore the use of MixSort (by SportsMOT authors) for inference, as it is trained specifically on SportsMOT for tracking. While we take a custom route, MixSort is a strong baseline to benchmark against.
v) Combine SportsMOT with SoccerNet dataset to improve domain generalization.
vi) Integrate auxiliary cues like OCR (jersey numbers) or head/face features, if reliably detected.
vii) The scope of unique IDs in SportsMOT is not global. Nor is it a dataset that can be used for commercial purposes. So maybe getting some custom data annotated?
viii) Investigate including samples from datasets like Market1501 to regularize the model and improve generalization beyond uniforms (proposed by one of the papers)
ix)Explore more modern tracking methods such as TrackTrack or other motion-aware trackers.
x)Tune tracking hyperparameters (e.g., detection confidence, match thresholds) to suppress spurious detections.

