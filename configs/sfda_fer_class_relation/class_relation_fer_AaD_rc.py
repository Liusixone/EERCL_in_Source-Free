_base_ = [
    '../_base_/cls_datasets/sfda_fer/sfda_fer_target_2gpu_rc.py',
    '../_base_/cls_models/sfda_fer/resnet_50_sfda_simplified_fer_target_model.py'
]

# models = dict(find_unused_parameters=False)

log_interval = 100
val_interval = 300

control = dict(
    log_interval=log_interval,
    max_iters=15000,
    val_interval=val_interval,
    cudnn_deterministic=True,
    save_interval=500,
    max_save_num=1,
    seed=2023,
    pretrained_model='/home/wbx/code/CRCo/runs/sfda_fer_source_raf/job_my_exp_exp_10764/best_model.pth',
)

train = dict(
    baseline_type='AaD',
    lambda_nce=1.0,
    lambda_ent=0.0,
    lambda_div=0.0,
    fix_classifier=True,
    pseudo_update_interval=50,
    lambda_fixmatch=1.0,
    prob_threshold=0.95,
    use_cluster_label_for_fixmatch=True,
    lambda_fixmatch_temp=0.07,
)

test = dict(
custom_hooks=[
        dict(type='ClsAccuracy', dataset_index=0, pred_key='pred'),
        # dict(type='ClsAccuracy', dataset_index=0, pred_key='target_pred'),
        dict(type='ClsAccuracy', dataset_index=1, pred_key='pred'),
        # dict(type='ClsAccuracy', dataset_index=0, pred_key='target_pred'),
        dict(type='ClsBestAccuracyByTest', patience=1000, priority="LOWEST")
    ]
)
