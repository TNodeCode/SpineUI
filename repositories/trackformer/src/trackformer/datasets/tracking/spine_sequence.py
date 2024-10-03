# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT17 sequence dataset.
"""
import configparser
import csv
import os
import json
from pathlib import Path
import os.path as osp
from argparse import Namespace
from typing import Optional, Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..coco import make_coco_transforms
from ..transforms import Compose


class SequenceHelper:
    @staticmethod
    def get_sequence_names(annotation_filepath: str):
        """
        Read the JSON annotation file and return all sequences defined in there

        Params:
            annotation_filename     The annotation file
        """
        with open(annotation_filepath) as f:
            content = json.load(f)
        return content['sequences']


class SpineSequence(Dataset):
    """SpineSequence, Custom Spine Dataset.
    """
    data_folder = os.getenv('DATASET') if os.getenv('DATASET') else None
    if data_folder is None:
        raise ValueError("Please provide dataset name via DATASET environment variable") 

    def __init__(self, root_dir: str = 'data', seq_name: Optional[str] = None, 
                 vis_threshold: float = 0.0, img_transform: Namespace = None) -> None:
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons
                                   above which they are selected
        """
        super().__init__()
        
        self._seq_name = seq_name
        self._vis_threshold = vis_threshold

        self._data_dir = osp.join(root_dir, self.data_folder)

        self._train_seqs = SequenceHelper.get_sequence_names(f"{self._data_dir}/annotations/train.json")
        self._val_seqs = SequenceHelper.get_sequence_names(f"{self._data_dir}/annotations/val.json")
        self._test_seqs = SequenceHelper.get_sequence_names(f"{self._data_dir}/annotations/test.json")

        self.transforms = Compose(make_coco_transforms('val', img_transform, overflow_boxes=True))
        self.data = []
        self.no_gt = True

        assert (seq_name is not None) and (seq_name in self._train_seqs or self._val_seqs or self._test_seqs), \
                'Sequence not in train, val or test sequences : {}'.format(seq_name)
        if seq_name in self._train_seqs:
            self._split = 'train'
        if seq_name in self._val_seqs:
            self._split = 'val'
        if seq_name in self._test_seqs:
            self._split = 'test'
        else:
            raise Exception(f"Sequence {seq_name} not found neither in train, val or test set")

        self.gt = self.load_gt()
        self.images = self.load_image_metadata()
        self.annotations = self.load_annotations()
        self.img_ids = self.load_img_ids()
        
        self.data = self._sequence()

        self.no_gt = not osp.exists(self.get_gt_file_path())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        width_orig, height_orig = img.size

        img, _ = self.transforms(img)
        width, height = img.size(2), img.size(1)

        sample = {}
        sample['img'] = img
        sample['dets'] = torch.empty((0,4))
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']
        sample['orig_size'] = torch.as_tensor([int(height_orig), int(width_orig)])
        sample['size'] = torch.as_tensor([int(height), int(width)])

        return sample
    
    def __str__(self):
        return f"Sequence: {self._seq_name}, length: {len(self.data)}, images: {len(self.images)}, annotations: {len(self.annotations)}"

    def _sequence(self) -> dict:
        total = []
        img_dir = osp.join(self._data_dir,self._split)

        boxes, visibility = self.get_track_boxes_and_visibility()

        total = [
            {'gt': boxes[frame_id],
             'im_path': osp.join(img_dir, self.img_ids[frame_id]),
             'vis': visibility[frame_id]}
            for frame_id in self.img_ids.keys()]

        return total
    
    def load_gt(self) -> dict:
        gt_file = self.get_gt_file_path()

        if not osp.exists(gt_file):
            print(f'GT file {gt_file} does not exist.')
            self.seq_length = 0
            return None

        # load file
        with open(gt_file, 'r') as f:
            gt = json.loads(f.read())

        return gt

    def load_annotations(self) -> list:
            """ Load annotations from GT file."""
            annotations = [annot for annot in self.gt['annotations'] \
                           if annot['seq'] == self._seq_name]
            return annotations
    
    def load_image_metadata(self) -> list:
        img_metadata = [img_info for img_info in self.gt['images'] \
                        if self._seq_name in img_info['file_name']]
        self.seq_length = int(img_metadata[0]['seq_length']) if len(img_metadata) > 0 else 0

        return img_metadata

    def load_img_ids(self) -> dict:
        img_ids = {}
        for metadata in self.images:
            frame_id = int(metadata['frame_id'])
            img_ids[frame_id] = metadata['file_name']        
        return img_ids

    def get_track_boxes_and_visibility(self) -> Tuple[dict, dict]:
        """ Load ground truth boxes and their visibility."""
        boxes = {}
        visibility = {}
        image_ids = []

        for i in range(self.seq_length):
            boxes[i] = {}
            visibility[i] = {}
        
        for annot in self.annotations:
            image_id = int(annot['image_id'])
            if image_id not in image_ids:
                image_ids.append(image_id)
            xywh = annot['bbox'] 
            frame_id = image_ids.index(image_id)
            track_id = int(annot['track_id'])
            
            boxes[frame_id][track_id] = np.array([xywh[0], xywh[1], xywh[0]+xywh[2]-1, xywh[1]+xywh[3]-1], dtype=np.float32)
            visibility[frame_id][track_id] = float(annot['visibility'] )
        
        return boxes, visibility

    def get_gt_file_path(self) -> str:
        """ Return ground truth file of sequence. """
        return osp.join(self._data_dir, 'annotations', self._split + '.json')
    
    @property
    def results_file_name(self) -> str:
        """ Generate file name of results file. """
        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

        return f"{self._seq_name}.txt"
    
    def write_results(self, results: dict, output_dir: str) -> None:
        """Write the tracks in the format for MOT16/MOT17 sumbission

        results: dictionary with 1 dictionary for every track with
                 {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

        # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        result_file_path = osp.join(output_dir, self.results_file_name)

        with open(result_file_path, "w") as r_file:
            writer = csv.writer(r_file, delimiter=',')

            for i, track in results.items():
                for frame, data in track.items():
                    x1 = data['bbox'][0]
                    y1 = data['bbox'][1]
                    x2 = data['bbox'][2]
                    y2 = data['bbox'][3]

                    writer.writerow([
                        frame + 1,
                        i + 1,
                        x1 + 1,
                        y1 + 1,
                        x2 - x1 + 1,
                        y2 - y1 + 1,
                        -1, -1, -1, -1])

    def load_results(self, results_dir: str) -> dict:
        results = {}
        if results_dir is None:
            return results

        file_path = osp.join(results_dir, self.results_file_name)

        if not os.path.isfile(file_path):
            return results

        with open(file_path, "r") as file:
            csv_reader = csv.reader(file, delimiter=',')

            for row in csv_reader:
                frame_id, track_id = int(row[0]) - 1, int(row[1]) - 1

                if track_id not in results:
                    results[track_id] = {}

                x1 = float(row[2]) - 1
                y1 = float(row[3]) - 1
                x2 = float(row[4]) - 1 + x1
                y2 = float(row[5]) - 1 + y1

                results[track_id][frame_id] = {}
                results[track_id][frame_id]['bbox'] = [x1, y1, x2, y2]
                results[track_id][frame_id]['score'] = 1.0

        return results


class SpineWrapper(Dataset):    
    """A Wrapper for the SpineSequence class to return multiple sequences."""
    def __init__(self, split: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT20Sequence dataset
        """
        train_sequences = SequenceHelper.get_sequence_names(f"{self._data_dir}/annotations/train.json")
        val_sequences = SequenceHelper.get_sequence_names(f"{self._data_dir}/annotations/val.json")
        test_sequences = SequenceHelper.get_sequence_names(f"{self._data_dir}/annotations/test.json")

        if split == "train":
            sequences = train_sequences
        elif split == "val":
            sequences = val_sequences
        elif split == "test":
            sequences = test_sequences
        elif split == "all":
            sequences = train_sequences + val_sequences + test_sequences
            sequences = sorted(sequences)
        else:
            raise NotImplementedError(f"Split {split} not available.")

        self._data = []
        for seq in sequences:
            self._data.append(SpineSequence(split=split, seq_name=seq, **kwargs))  
        
    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]