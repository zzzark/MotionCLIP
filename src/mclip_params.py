mclip_params = {
    'activation': 'gelu',
    'align_pose_frontview': False,
    'archiname': 'transformer',
    'batch_size': 20,
    'clip_image_losses': ['cosine'],
    'clip_lambda_ce': 1.0,
    'clip_lambda_cosine': 1.0,
    'clip_lambda_mse': 1.0,
    'clip_lambdas': {
        'image': {'cosine': 1.0},
        'text': {'cosine': 1.0}
    },
    'clip_map_images': False,
    'clip_map_text': False,
    'clip_mappers_type': 'no_mapper',
    'clip_text_losses': ['cosine'],
    'cuda': True,
    'datapath': './data/amass_db/amass_30fps_db.pt',
    'dataset': 'amass',
    'debug': False,
    # 'device': 'cuda:0',
    'expname': 'exps',
    'folder': './exps/paper-model',
    'glob': True,
    'glob_rot': [3.141592653589793, 0, 0],
    'jointstype': 'smpl',
    'lambda_kl': 0.0,
    'lambda_rc': 100.0,
    'lambda_rcxyz': 100.0,
    'lambda_vel': 100.0,
    'lambda_velxyz': 1.0,
    'lambdas': {'rc': 100.0, 'rcxyz': 100.0, 'vel': 100.0},
    'latent_dim': 512,
    'losses': ['rc', 'rcxyz', 'vel'],
    'lr': 0.0001,
    'max_len': -1,
    'min_len': -1,
    'modelname': 'cvae_transformer_rc_rcxyz_vel',
    'modeltype': 'cvae',
    'normalize_encoder_output': False,
    'num_epochs': 5000,
    'num_frames': 60,
    'num_layers': 8,
    'num_seq_max': -1,
    'pose_rep': 'rot6d',
    'sampling': 'conseq',
    'sampling_step': 1,
    'snapshot': 10,
    'translation': True,
    'vertstrans': False,
    'num_actions_to_sample': 5,
    'num_samples_per_action': 5,
    'fps': 20,
    'appearance_mode': 'motionclip',
    'force_visu_joints': True,
    'noise_same_action': 'random',
    'noise_diff_action': 'random',
    'duration_mode': 'mean',
    'reconstruction_mode': 'ntf',
    'decoder_test': 'new',
    'fact_latent': 1,
    'images_dir': './action_images',
    'input_file': './assets/rm_texts.txt',
    'zero_global_orient': False,
    'ae_after_generation': False,
    # 'checkpointname': './exps/classes-model/checkpoint_0200.pth.tar',
    'checkpointname': './exps/paper-model/checkpoint_0100.pth.tar',
    'figname': 'fig_{:03d}',
    'num_classes': 1,
    'nfeats': 6,
    'njoints': 25,
    'outputxyz': True
}
