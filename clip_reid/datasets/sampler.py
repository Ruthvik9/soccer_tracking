from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy, json
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid). # should be (img_path, pid, camid, trackid)
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class SportsMOTSampler(Sampler):
    """
    Like RandomIdentitySampler but:
      - Allows multiple PIDs from the same clip (true negatives)
      - Forbids PIDs from different clips of the same match (as these can be false negatives)
    Expects:
      data_source.dataset → list of (img, pid, camid, tid)
      gid2info.json → { gid: { "match": match_id, "clip": clip_id } }
    """
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source         = data_source
        self.batch_size          = batch_size
        self.num_instances       = num_instances
        self.num_pids_per_batch  = batch_size // num_instances
        gid2info_path = "datasets/SportsMOT_Football/gid2info.json"

        # 1) Build pid → indices
        self.index_dic = defaultdict(list)
        for idx, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(idx)
        self.pids = list(self.index_dic.keys())

        # 2) Load pid → {match,clip}
        with open(gid2info_path, 'r') as f:
            raw = json.load(f)
        # ensure types
        self.pid2match = {int(pid): info['match'] for pid,info in raw.items()}
        self.pid2clip  = {int(pid): info['clip']  for pid,info in raw.items()}

        # 3) Estimate length (same as RandomIdentitySampler)
        length = 0
        for pid in self.pids:
            cnt = len(self.index_dic[pid])
            cnt = cnt if cnt >= num_instances else num_instances # cuz we'll pad if cnt < num_instances
            length += cnt - cnt % num_instances
        self.length = length

    def __iter__(self):
        # 4) Pre‐chunk each pid’s indices into groups of size num_instances
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = list(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = list(np.random.choice(idxs, self.num_instances, replace=True))
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        available_pids = set(batch_idxs_dict.keys())
        final_idxs = []

        # 5) Build batches
        while len(available_pids) >= self.num_pids_per_batch:
            chosen_match_clip = {}
            chosen_pids       = []
            pid_list = list(available_pids)
            random.shuffle(pid_list)

            # select pids with the clip/match constraint
            for pid in pid_list:
                mid = self.pid2match[pid]
                cid = self.pid2clip[pid]
                if mid not in chosen_match_clip:
                    chosen_match_clip[mid] = cid
                    chosen_pids.append(pid)
                elif chosen_match_clip[mid] == cid:
                    chosen_pids.append(pid) # Same match and same clip so append the person
                # else: same match but different clip → skip
                if len(chosen_pids) >= self.num_pids_per_batch:
                    break

            if len(chosen_pids) < self.num_pids_per_batch:
                break  # can’t form a full batch any more

            # pull one chunk per chosen pid
            for pid in chosen_pids:
                grp = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(grp)
                if not batch_idxs_dict[pid]:
                    available_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


