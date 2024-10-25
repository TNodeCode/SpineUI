# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
import numpy as np
import pandas as pd
import click
import torch

import mmengine
import mmcv

from mmdet.apis import inference_mot, init_track_model
from mmdet.structures.track_data_sample import TrackDataSample
from mmdet.registry import VISUALIZERS
from mmengine.structures import InstanceData

from detect import detect as _detect
from eval import evaluate as evaluate


@click.group()
def cli():
    pass


@cli.command()
@click.option('--model-type', type=str, required=True, help='Type of model to use for detection')
@click.option('--model-name', type=str, required=True, help='Name of the model')
@click.option('--weight-file', type=str, required=True, help='Model weight file')
@click.option('--work-dir', type=str, required=True, help='Root path of the checkpoint files')
@click.option('--dataset-dir', type=str, required=True, help='Root path of dataset')
@click.option('--image-files', type=str, required=True, help='Glob path for images')
@click.option('--results-file', type=str, required=True, help='Name of the resulting CSV file')
@click.option('--batch-size', type=int, default=2, help='Batch size for training (greater than 0, default: 2)')
@click.option('--score-threshold', type=float, default=0.5, help='Minimum confidence score for bounding box detection')
@click.option('--device', type=str, default='cuda:0', help='Device to use for detection (default: cuda:0)')
def detect(model_type, model_name, weight_file, work_dir, dataset_dir, image_files, results_file, batch_size, score_threshold, device):
    _detect(
        model_name=model_name,
        model_type=model_type,
        weight_file=weight_file,
        results_file=results_file,
        work_dir=work_dir,
        dataset_dir=dataset_dir,
        image_files=image_files.replace("'", ""),
        batch_size=batch_size,
        score_threshold=score_threshold,
        device=device
    )


@cli.command()
@click.option('--model_type', type=str, required=True, help='Type of model to use for detection')
@click.option('--model_name', type=str, required=True, help='Name of the model')
@click.option('--annotations', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help='Path to the annotations')
@click.option('--epochs', type=int, required=True, help='Number of training epochs (greater than 0)')
@click.option('--csv_file_pattern', type=str, required=True, help='Pattern for the CSV files ($i will be replaced by epoch number)')
@click.option('--results_file', type=str, required=True, help='Name of the resulting CSV file')
@click.option('--score-threshold', type=float, default=0.5, help='Minimum confidence score for bounding box detection')
def eval(
    model_type: str,
    model_name: str,
    annotations: str,
    epochs: int,
    csv_file_pattern: str,
    results_file: str,
    score_threshold: float,
):
    evaluate(
        gt_file_path=annotations,
        model_type=model_type,
        model_name=model_name,
        csv_file_pattern=csv_file_pattern,
        results_file=results_file,
        max_epochs=epochs,
        score_threshold=score_threshold,
    )


@cli.command()
@click.option('--config', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help='config file')
@click.option('--ckp-detector', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help='checkpoint file for detection model')
@click.option('--ckp-reid', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=False, default=None, help='checkpoint file for reid model')
@click.option('--input-dir', type=click.Path(exists=True, file_okay=True, dir_okay=True), required=True, help='input video file or folder')
@click.option('--output-dir', type=click.Path(exists=False, file_okay=True, dir_okay=True), required=True, help='directory where outputs should be saved')
@click.option('--fps', type=int, required=False, default=1, help='FPS of the output video')
@click.option('--score_thr', type=float, required=False, default=0.5, help='The threshold of score to filter bboxes')
@click.option('--backend', type=str, required=False, default='cv2', help='The backend used for generating video frames (cv2 | plt)')
@click.option('--device', type=str, required=False, default='cuda:0', help='device used for inference (cuda:0 | cpu)')
@click.option('--show', type=bool, required=False, default=True, help='whether show the results on the fly')
def track(config: str, ckp_detector: str, ckp_reid: str, input_dir: str, output_dir: str, fps: int, score_thr: float, backend: str, device: str, show: bool):
    mmengine.registry.init_default_scope('mmdet')

    # build the model from a config file and a checkpoint file
    checkpoint = None
    model = init_track_model(
        config,
        checkpoint,
        detector=ckp_detector,
        reid=ckp_reid,
        device=device,
        cfg_options={},
    )

    # build the visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # Get all stack names
    stack_names = os.listdir(input_dir)

    # Perform detection and tracking for each stack
    for stack_index, stack_name in enumerate(stack_names):
        print(f"Perform tracking for stack {stack_index + 1}/{len(stack_names)}")
        stack_img_dir = os.path.join(input_dir, stack_name, 'img')

        # load input images or video
        if osp.isdir(stack_img_dir):
            imgs = sorted(
                filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                    os.listdir(stack_img_dir)),
                key=lambda x: int(x.split('.')[0]))
            IN_VIDEO = False
        else:
            imgs = mmcv.VideoReader(stack_img_dir)
            IN_VIDEO = True
        
        # define output
        OUT_VIDEO = True
        output = "video.mp4"
        out_dir = tempfile.TemporaryDirectory()
        out_path = out_dir.name
        prog_bar = mmengine.ProgressBar(len(imgs))

        detections = []
        trajectories = []

        # test and show/save the images
        for i, img_path in enumerate(imgs):
            if isinstance(img_path, str):
                img_path = osp.join(stack_img_dir, img_path)
            img = mmcv.imread(img_path)
            result: TrackDataSample = inference_mot(model, img, frame_id=i, video_len=len(imgs))

            if result.video_data_samples[0].pred_instances.bboxes.shape[0] > 0: # check if there are any detections
                # Detection results
                if result.video_data_samples[0].pred_instances.bboxes.shape[1] == 4:
                    bboxes = result.video_data_samples[0].pred_instances.bboxes
                    labels = result.video_data_samples[0].pred_instances.labels
                    scores = result.video_data_samples[0].pred_instances.scores          
                else:
                    raise Exception("Invalid bboxes shape")
                
                for j, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
                    if bool(bbox.min().isinf()) or bool(bbox.max().isinf() or int(bbox.min()) < 0 or int(bbox.max()) > img.shape[0]):
                        continue
                    if type(bbox) == np.ndarray and (bool(np.isinf(bbox.min())) or bool(np.isinf(bbox.max())) or int(bbox.min()) < 0 or int(bbox.max()) > img.shape[0]):
                        continue
                    detections.append({
                        'frame': i+1,
                        'x0': int(bbox[0]),
                        'x1': int(bbox[2]),
                        'y0': int(bbox[1]),
                        'y1': int(bbox[3]),
                        'score': float(score)
                    })
            
            if result.video_data_samples[0].pred_track_instances.bboxes.shape[0] > 0: # check if there are any trajectories
                if result.video_data_samples[0].pred_track_instances.bboxes.shape[1] == 4:
                    bboxes = result.video_data_samples[0].pred_track_instances.bboxes
                    labels = result.video_data_samples[0].pred_track_instances.labels
                    instances_ids = result.video_data_samples[0].pred_track_instances.instances_id
                    scores = result.video_data_samples[0].pred_track_instances.scores
                elif result.video_data_samples[0].pred_track_instances.bboxes.shape[1] == 7:
                    bboxes = result.video_data_samples[0].pred_track_instances.bboxes[:, 2:6]
                    # with to x1 = x0 + width
                    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
                    # height to y1 = y0 + height
                    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
                    labels = result.video_data_samples[0].pred_track_instances.bboxes[:, 0]
                    instances_ids = result.video_data_samples[0].pred_track_instances.bboxes[:, 1]
                    scores = result.video_data_samples[0].pred_track_instances.bboxes[:, 6]
                else:
                    raise Exception("Invalid bboxes shape")
                
                for j, (bbox, label, instance_id, score) in enumerate(zip(bboxes, labels, instances_ids, scores)):
                    if type(bbox) == torch.Tensor and (bool(bbox.min().isinf()) or bool(bbox.max().isinf()) or int(bbox.min()) < 0 or int(bbox.max()) > img.shape[0]):
                        continue
                    if type(bbox) == np.ndarray and (bool(np.isinf(bbox.min())) or bool(np.isinf(bbox.max())) or int(bbox.min()) < 0 or int(bbox.max()) > img.shape[0]):
                        continue
                    trajectories.append({
                        'frame': i+1,
                        'object_id': int(instance_id),
                        'x0': int(bbox[0]),
                        'x1': int(bbox[1]),
                        'w': int(bbox[2]) - int(bbox[0]),
                        'h': int(bbox[3]) - int(bbox[1]),
                        'score': float(score)
                    })

            if output is not None:
                if IN_VIDEO or OUT_VIDEO:
                    out_file = osp.join(out_path, f'{i:06d}.jpg')
                else:
                    out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
            else:
                out_file = None
            
            # show the results
            visualizer.add_datasample(
                'mot',
                img[..., ::-1],
                data_sample=result[0],
                show=show,
                draw_gt=False,
                out_file=out_file,
                wait_time=float(1 / int(fps)) if fps else 0,
                pred_score_thr=score_thr,
                step=i)
            prog_bar.update()

        # Save results as CSV file
        df_detections = pd.DataFrame(detections)
        df_trajectories = pd.DataFrame(trajectories)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'seqmaps'), exist_ok=True)
        df_detections.to_csv(os.path.join(output_dir, f"det_{stack_name}.csv"), index=False, header=False)
        df_trajectories.to_csv(os.path.join(os.path.join(output_dir, 'seqmaps'), f"{stack_name}.csv"), index=False, header=False)

        if OUT_VIDEO:
            print(f'making the output video at {output_dir} with a FPS of {fps}')
            mmcv.frames2video(out_path, os.path.join(output_dir, f"{stack_name}.mp4"), fps=fps, fourcc='mp4v')
            out_dir.cleanup()


if __name__ == '__main__':
    cli()
