tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))

img_scales = [(1333, 800), (666, 400), (2000, 1200)]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='Resize', scale=s, keep_ratio=True) for s in img_scales
        ], [
            dict(type='RandomFlip', prob=0.5, direction="horizontal"), 
            dict(type='RandomFlip', prob=0.5, direction="vertical"),
            dict(type='RandomFlip', prob=0.5, direction="diagonal"),
            dict(type='RandomAffine', max_rotate_degree=10.0, max_translate_ratio=0.1, scaling_ratio_range=(0.5, 1.5), max_shear_degree=2.0),
        ], [dict(type='LoadAnnotations', with_bbox=True)],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ]])
]
