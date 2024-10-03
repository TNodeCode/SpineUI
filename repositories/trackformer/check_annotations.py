import os
import json

dataset_root = "data/" + os.getenv('DATASET') if os.getenv('DATASET') else "data/spine_dataset"

for split in ['train', 'val', 'test']:
    print(f"Checking {split} dataset")
    with open(f"{dataset_root}/annotations/{split}.json") as f:
        content = json.load(f)
    filenames = list(map(lambda x: x['file_name'], content['images']))
    sequences = content['sequences']
    category_ids = list(map(lambda x: x['id'], content['categories']))
    image_ids = list(map(lambda x: x['id'], content['images']))

    assert min(image_ids) == 0, "First image ID should be zero"
    assert len(image_ids) == max(image_ids) + 1, "Last image ID should be len(image_ids) + 1"
    
    for filename in filenames:
        assert os.path.exists(f"{dataset_root}/{split}/{filename}"), f"Image does not exist in {split} dataset: {filename}"

    for filename in os.listdir(f"{dataset_root}/{split}"):
        assert filename in filenames, f"Found filename {filename} in {split} dataset image folder that does never appear in annotations"

    for image in content["images"]:
        image["first_frame_image_id"] in image_ids

    for annotation in content['annotations']:
        id = annotation['id']
        assert annotation['image_id'] in image_ids, f"Image ID {annotation['image_id']} found in annotation dies not exist"
        assert annotation['category_id'] in category_ids, f"Category ID {annotation['category_id']} found in annotation dies not exist"
        assert annotation['seq'] in sequences, f"Sequence {annotation['seq']} not found"
        expected_area = int(annotation['bbox'][2]) * int(annotation['bbox'][3])
        assert expected_area == int(annotation['area']), f"Annotation {id}:Expected area to be {expected_area}, but was {annotation['area']}"

    for sequence in sequences:
        seq_annotations = list(filter(lambda x: x['seq'] == sequence, content['annotations']))
        assert len(seq_annotations) > 0, f"Sequence {sequence} has no annotations"
        sequence_images = [x for x in content['images'] if sequence in x['file_name']]
        for img in sequence_images:
            assert len(sequence_images) == img["seq_length"]