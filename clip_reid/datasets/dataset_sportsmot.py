import os, os.path as osp
from .bases import BaseImageDataset

class SportsMOTFootball(BaseImageDataset):
    """
    SportsMOT Football → Image ReID.
    Expects:
      list_train.txt
      list_query_val.txt
      list_gallery_val.txt
    under dataset_dir.
    """
    dataset_dir = 'SportsMOT_Football'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(SportsMOTFootball, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.list_train = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_query = osp.join(self.dataset_dir, 'list_query_val.txt')
        self.list_gallery = osp.join(self.dataset_dir, 'list_gallery_val.txt')
        self._check_before_run()

        train = self._process_list(self.list_train, pid_begin)
        query = self._process_list(self.list_query, pid_begin)
        gallery = self._process_list(self.list_gallery, pid_begin)

        if verbose:
            print("=> SportsMOTFootball loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train   = train
        self.query   = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    def _check_before_run(self):
        for path in (self.list_train, self.list_query, self.list_gallery):
            if not osp.exists(path):
                raise RuntimeError(f"'{path}' is not available")
    def _process_list(self, list_path, pid_begin):
        dataset = []
        pid_container = set()
        cam_container = set()
        with open(list_path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
        for line in lines:
            img_rel, pid, camid = line.split()
            pid   = int(pid) + pid_begin
            camid = int(camid)
            img_path = osp.join(self.dataset_dir, img_rel)
            dataset.append((img_path, pid, camid, 0))
            pid_container.add(pid)
            cam_container.add(camid)
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset