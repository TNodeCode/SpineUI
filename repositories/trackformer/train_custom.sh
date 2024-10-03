export DATASET=spine_all

python src/train.py with \
    mot17 \
    deformable \
    multi_frame \
    tracking \
    device=cuda:0 \
    output_dir=models/custom_dataset_deformable \
    mot_path_train=data/${DATASET} \
    mot_path_val=data/${DATASET} \
    train_split=train \
    val_split=val \
    epochs=20 \
