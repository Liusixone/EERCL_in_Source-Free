_base_ = [
    '../_base_/cls_datasets/sfda_fer/sfda_fer_source_1gpu_raf.py',
    '../_base_/cls_models/sfda_fer/resnet_50_sfda_fer_source_model.py'
]

models = dict(find_unused_parameters=False)

log_interval = 100
val_interval = 500

control = dict(
    log_interval=log_interval,
    max_iters=20000,
    val_interval=val_interval,
    # cudnn_deterministic=True,
    save_interval=1000,
    max_save_num=4,
    # seed=2,
)

train = dict(
    src_ce_type='weak',
    lambda_label_smooth=0.1,
)

test = dict(
custom_hooks=[
        dict(type='ClsAccuracy', dataset_index=0, pred_key='pred'),
        # dict(type='ClsAccuracy', dataset_index=0, pred_key='target_pred'),
        dict(type='ClsAccuracy', dataset_index=1, pred_key='pred'),
        # dict(type='ClsAccuracy', dataset_index=0, pred_key='target_pred'),
        dict(type='ClsBestAccuracyByVal', patience=100, priority="LOWEST")
    ]
)
