from yolox.tracker.byte_tracker import BYTETracker
import numpy as np
import pandas as pd
import os


def track(args, df_stack, output_dir, stack_name):
    tracker = BYTETracker(args, frame_rate=30)

    # Store tracking results
    tracking_results = []

    # Group by frame and process detections frame by frame
    for frame_id, frame_data in df_stack.groupby('frame'):
        detections = []

        # Convert each detection into the format ByteTrack expects
        for _, row in frame_data.iterrows():
            x0, y0, x1, y1, score = row[['x0', 'y0', 'x1', 'y1', 'score']]
            detection = [x0, y0, x1, y1, score]  # Convert to (x0, y0, x1, y1, score)
            detections.append(detection)

        # Convert detections into NumPy array
        detections = np.array(detections)

        if len(detections) == 0:
            continue

        # Track objects
        online_targets = tracker.update(detections, img_info=[512, 512], img_size=[512, 512])

        for t in online_targets:
            tlwh = t.tlwh  # Get bounding box (top-left-width-height)
            tid = t.track_id  # Get unique object ID
            score = t.score  # Confidence score

            # Add tracking result: <frame>, <object_id>, <x0>, <y0>, <w>, <h>, <score>
            tracking_results.append([int(os.path.splitext(os.path.basename(frame_id))[0]), tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], score])

    # Save tracking results to CSV
    tracking_results_df = pd.DataFrame(tracking_results, columns=['frame', 'object_id', 'x0', 'y0', 'w', 'h', 'score'])
    tracking_results_df.to_csv(output_dir + '/' + stack_name + '.txt', index=False, header=False)