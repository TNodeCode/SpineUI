export DATASET=spine_all
export SUBDIR=run_4

for MODEL_NAME in MOTA IDF1 BBOX_AP_IoU_0_50
do
    for SPLIT in train val test
    do
        for file in ./data/$DATASET/$SPLIT/*
        do
        if [ -d "$file" ]; then
            STACK_NAME="$(basename -- $file)"
            echo "Processing" $SPLIT $STACK_NAME "..."
            python src/track.py with \
                dataset_name=$STACK_NAME \
                obj_detect_checkpoint_file=checkpoints/$SUBDIR/checkpoint_best_$MODEL_NAME.pth \
                write_images=True \
                generate_attention_maps=False \
                output_dir=detections/$SUBDIR/$MODEL_NAME/$SPLIT
        fi
        done
    done
done