# Football Player Re-Identification & Tracking Pipeline

This repository provides a full pipeline to transform raw SportsMOT football clips into broadcast-quality player tracking results using a fine-tuned CLIP-ReID model and BoT-SORT.

---

## Prerequisites

* Linux environment
* Python 3.8 or higher
* PyTorch 1.12 or higher
* torchvision 0.13 or higher

---

![Input video](15sec_input_720.gif)

![Output](15sec_output_720_tight.gif)

## Setup

### Clone and Install CLIP-ReID

Clone the repository and install dependencies:

```bash
git clone https://github.com/Syliz517/CLIP-ReID.git
cd CLIP-ReID
pip install -r requirements.txt
```

### Download and Organize SportsMOT Dataset

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
checkpoints/MSMT17_clipreid_ViT-B-16_60.pth
```

---

## Data Preparation

Use the `prepare_sportsmot.py` script from the project root:

```bash
mkdir -p datasets/SportsMOT_Football
python prepare_sportsmot.py \
  --mot_root /path/to/SportsMOT \
  --out_root datasets/SportsMOT_Football \
  --holdout 0.1
```

Output structure:

```text
datasets/SportsMOT_Football/
  train/0...Ntrain-1/*.jpg
  val/0...Nval-1/*.jpg
  list_train.txt
  list_query_val.txt
  list_gallery_val.txt
  gid2info.json
```

---

## Dataset Integration

Define a class `SportsMOTFootball` inheriting from `BaseImageDataset` to parse the prepared list files and convert PIDs to a contiguous range. Register this dataset in `clipreid/dataset.py` under the dataset catalog.

---

## Match-Aware Sampler

Implement `SportsMOTSampler` in `clipreid/utils/sampler.py`. This sampler allows negative sampling within a clip but not across clips of the same match. Register it in `sampler_factory.py` and configure via YAML:

```yaml
DATALOADER:
  SAMPLER: 'SportsMOTSampler'
  SAMPLER_ARGS:
    gid2info_path: 'datasets/SportsMOT_Football/gid2info.json'
  NUM_INSTANCE: 4
  PIDS_PER_BATCH: 16
```

---

## Augmentation Pipeline

Replace the default training transforms in `clipreid/dataset.py` with a custom pipeline including:

* Resize, padding, random crop, horizontal flip
* Color jittering, gamma adjustment
* Gaussian blur, normalization
* Random erasing

This enhances model robustness through both geometric and photometric variations.

---

## Checkpoint Loading

Modify `train_clipreid.py` to load pretrained weights using:

```python
model.load_param(cfg.TEST.WEIGHT)
```

Ensure the `cfg.TEST.WEIGHT` is set to the path of the MSMT17 checkpoint in the config or CLI args.

---

## Training and Validation

To fine-tune the model, run:

```bash
python train_clipreid.py \
  --config_file configs/person/vit_clipreid.yml \
  DATASETS.NAMES "('sportsmot_football',)" \
  DATASETS.ROOT_DIR "('datasets',)" \
  DATALOADER.SAMPLER "SportsMOTSampler" \
  DATALOADER.SAMPLER_ARGS '{"gid2info_path":"datasets/SportsMOT_Football/gid2info.json"}' \
  TEST.WEIGHT "checkpoints/MSMT17_clipreid_ViT-B-16_60.pth"
```

Evaluate using:

```bash
python test_clipreid.py \
  --config_file configs/person/vit_clipreid.yml \
  --q datasets/SportsMOT_Football/list_query_val.txt \
  --g datasets/SportsMOT_Football/list_gallery_val.txt
```

---

## Tracking with BoT-SORT

1. Run a detector like YOLOv5 and save per-frame detection `.txt` files.
2. Crop detections and extract 768-d embeddings using the fine-tuned CLIP-ReID model.
3. Use `BoTSORT` initialized with the ReID model and match threshold.

The final tracking output is in the format:

```text
[frame, track_id, x1, y1, x2, y2]
```

Convert it to MOT format or visualize with overlays.

---

## Known Issues & Resolutions

* **PID Continuity:** Handled by separate remapping for train and val sets.
* **CamID in Val:** camid=1 for query, camid=0 for gallery.
* **Sampler Logic:** Enforced via custom `SportsMOTSampler`.
* **Position Embedding Mismatch:** Ensure input size matches or resize position embeddings in the load method.
* **Checkpoint Usage in Training:** Confirmed by explicitly loading the weight in `train_clipreid.py`.
