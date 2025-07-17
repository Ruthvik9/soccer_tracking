import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as T
from make_model_clipreid import make_model
from tracker.bot_sort import BoTSORT
from clip_reid_configs import cfg
import torchvision.transforms.functional as VF
from types import SimpleNamespace


device = 'cuda'
args = SimpleNamespace(
    # association thresholds
    track_high_thresh   = 0.7,      # first‐pass detection threshold
    track_low_thresh    = 0.2,      # second‐pass threshold
    new_track_thresh    = 0.8,      # start new tracks above this conf
    match_thresh        = 0.8,      # gating threshold for Hungarian solver
    # how long to keep “lost” tracks
    track_buffer        = 150,
    # appearance gating (cosine / IoU)
    proximity_thresh    = 0.3,
    appearance_thresh   = 0.15,
    # enable appearance‐based matching
    with_reid           = True,
    # FastReID stubs (we’ll overwrite `.encoder` anyway)
    fast_reid_config    = '',
    fast_reid_weights   = '',
    # device for both Kalman & ReID
    device              = device,   # e.g. 'cuda' or 'cpu'
    # camera‐motion compensation
    cmc_method          = 'sparseOptFlow',
    # other CLI flags
    name                = 'BoTSORT',
    ablation            = False,
    mot20               = False,
)

cfg.merge_from_file('vit_clipreid.yml') # The same config used for finetuning on SportsMOT
cfg.freeze()
reid_model = make_model(cfg,
                   num_class=558, # 558 identities during finetuning, but it won't be used for feat. extraction anyways
                   camera_num=1,
                   view_num=1)
reid_model.load_param('models/clip_reid_finetuned.pth')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
reid_model.to(device).eval()

# 3) Inference transforms (deterministic)
reid_transforms = T.Compose([
    T.Resize([256,128], interpolation=3),           # your SIZE_TEST
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # your PIXEL_MEAN / PIXEL_STD
])

# Init tracker
tracker = BoTSORT(args, frame_rate=30)
# Define the CLIP inference so that the tracker can call it internally.
class CLIPInterface:
    def __init__(self, model, transforms, device):
        self.model      = model
        self.transforms = transforms
        self.device     = device

    def inference(self, img, tlbrs):
        """
        img: full BGR frame (numpy array)
        tlbrs: list/array of [x1,y1,x2,y2] boxes
        returns: list of 1D np.ndarray embeddings
        """
        feats = []
        for (x1,y1,x2,y2) in tlbrs:
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil  = T.ToPILImage()(rgb)
            inp  = reid_transforms(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = reid_model(inp, get_image=True)   # [1×D]
            feats.append(emb.squeeze(0).cpu().numpy())
        return feats

tracker.encoder = CLIPInterface(reid_model, reid_transforms, device)

# Load YOLO detector
detector = YOLO('models/yolov11_best.pt') 



# loading video I/O
cap = cv2.VideoCapture('15sec_input_720p.mp4')
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('15sec_output_720p_tight.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Class-filter index for 'player' and 'goalkeeper' only
names = detector.model.names
print("Classes are",names)
exit
player_cls = [i for i,n in names.items() if n in ('player')][0]
goalkeeper_cls = [i for i,n in names.items() if 'goalkeeper' in n.lower()][0]
allowed        = {player_cls, goalkeeper_cls}


while True:
    ret, frame = cap.read()
    if not ret: break
    res = detector(frame, imgsz=640)[0]
    if res.boxes:
        dets  = res.boxes.xyxy.cpu().numpy()    # Nx4
        confs = res.boxes.conf.cpu().numpy()    # N
        clss  = res.boxes.cls.cpu().numpy().astype(int)  # N
    else:
        dets = np.zeros((0,4)); confs = np.zeros((0,)); clss = np.zeros((0,),int)

    # Filter out ball detections and allow player and goalkeeper
    mask   = np.isin(clss, list(allowed))
    dets, confs, clss = dets[mask], confs[mask], clss[mask]

    output_results = np.concatenate([
    dets,
    confs.reshape(-1,1),
    clss.reshape(-1,1)
], axis=1)  # shape = (N, 6)

    # Extract ReID features
    # feats = []
    # for (x1,y1,x2,y2) in dets.astype(int):
    #     crop = frame[y1:y2, x1:x2]
    #     rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    #     pil  = T.ToPILImage()(rgb)
    #     inp    = reid_transforms(pil).unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         # get_image=True produce the visual embedding
    #         emb = reid_model(inp, get_image=True)             # tensor [1, D]
    #     # 3) Turn into a 1-D numpy vector
    #     feats.append(emb.squeeze(0).cpu().numpy())

    # Tracker update & draw
    # online = tracker.update(dets, confs, clss, feats, frame)
    online_targets = tracker.update(output_results, frame)
    for t in online_targets:
        x1,y1,x2,y2 = map(int, t.tlbr)
        tid = t.track_id
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame, str(tid),(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    out.write(frame)

cap.release()
out.release()
print("Saved as 15sec_output_720p_tight.mp4")
