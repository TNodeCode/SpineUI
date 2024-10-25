import pandas as pd
from PIL import Image
import shutil
import json
import os



root_dir = 'data/MOT17'
out_dir = 'data/DanceTrack'

os.makedirs(out_dir, exist_ok=True)
os.makedirs(f"{out_dir}/train", exist_ok=True)
os.makedirs(f"{out_dir}/val", exist_ok=True)
os.makedirs(f"{out_dir}/test", exist_ok=True)

for split in ['train', 'val', 'test']:
	stacks = os.listdir(f"{root_dir}/{split}")
	for stack in stacks:
		if stack == "seqmaps.txt":
			continue
		stack_dir = f"{root_dir}/{split}/{stack}"
		dest_stack_dir = f"{out_dir}/{split}/{stack}"
		dest_stack_img_dir = f"{dest_stack_dir}/img1"
		dest_stack_gt_dir = f"{dest_stack_dir}/gt"
		os.makedirs(dest_stack_img_dir, exist_ok=True)		
		os.makedirs(dest_stack_gt_dir, exist_ok=True)		
		if not os.path.isdir(stack_dir):
			continue
		df = pd.read_csv(f"{stack_dir}/gt/gt.txt", header=None)
		filenames = df[0].unique()
		for filename in filenames:
			image_filepath = f"{split}/{stack}/img/{str(filename).zfill(6)}.png"
			df_img = df[df[0] == filename]
			# Copy images and save them as JPG files
			Image.open(f"{root_dir}/{image_filepath}").convert('RGB').save(f"{dest_stack_img_dir}/{str(filename).zfill(8)}.jpg")
		# Copy and modify seqinfo.ini file
		with open(f"{stack_dir}/seqinfo.ini", "r") as f:
			content = f.read()
		content = content.replace("imDir=img", "imDir=img1")
		content = content.replace("imExt=.png", "imExt=.jpg")
		with open(f"{dest_stack_dir}/seqinfo.ini", "w+") as f:
			f.write(content)
		# Copy gt
		shutil.copytree(
			src=f"{stack_dir}/gt",
			dst=dest_stack_gt_dir,
			dirs_exist_ok=True
		)