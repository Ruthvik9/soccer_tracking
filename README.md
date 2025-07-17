# Football Player Re-Identification & Tracking Pipeline

This repository provides the implementation to try and track soccer players in video clips clips using a **fine-tuned CLIP-ReID model** and **BoT-SORT**.

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

### Download and Organize SportsMOT Dataset (if you want to fine)

Ensure the dataset is structured as follows:

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

Use the `prepare_sportsmot.py` script from the project root:

```bash
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
datasets/SportsMOT_Football/
  train/0...Ntrain-1/*.jpg
  val/0...Nval-1/*.jpg
  list_train.txt
  list_query_val.txt
  list_gallery_val.txt
  gid2info.json # maps global id to clip and match in SportsMOT. Useful later for the match-aware sampler.
```

---


## Match-Aware Sampler

This sampler allows negative sampling within a clip but not across clips of the same match. 

---

## Augmentation Pipeline

Replaced the default training transforms in `clipreid/dataset.py` with some extra steps including:

* Color jittering, gamma adjustment
* Gaussian blur, normalization

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
python infer_botsort.py
```


---

## Known Issues & Resolutions

* **PID Continuity:** Handled by separate remapping for train and val sets.
* **CamID in Val:** camid=1 for query, camid=0 for gallery.
* **Sampler Logic:** Enforced via custom `SportsMOTSampler`.
* **Position Embedding Mismatch:** Ensure input size matches or resize position embeddings in the load method.
* **Checkpoint Usage in Training:** Confirmed by explicitly loading the weight in `train_clipreid.py`.
