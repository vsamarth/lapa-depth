from laq_model import LAQTrainer
from laq_model import LatentActionQuantization

# rgb_path = '/media/do/data1/philo/lapa/something-something-v2/frames'
# depth_path = '/media/do/data1/philo/lapa/something-something-v2/depth'
# z_rgb_path = '/media/do/data1/philo/lapa/something-something-v2/pred_z_rgb_step1'


rgb_path = '/media/do/data1/philo/lapa/something-something-v2/ssv2-mini-2k-5/frames_train'
depth_path = '/media/do/data1/philo/lapa/something-something-v2/ssv2-mini-2k-5/depth_train'
z_rgb_path = '/media/do/data1/philo/lapa/something-something-v2/ssv2-mini-2k-5/z_rgb_indices_stage2_train'


laq = LatentActionQuantization(
    dim = 1024,
    quant_dim=32,
    codebook_size = 8,
    image_size = 256,
    patch_size = 32,
    spatial_depth = 8, #8
    temporal_depth = 8, #8
    dim_head = 64,
    heads = 16,
    code_seq_len=4,
).cuda()


trainer = LAQTrainer(
    laq,
    folder = rgb_path,
    depth_folder = depth_path,
    z_rgb_folder = z_rgb_path,
    offsets = 30,
    batch_size = 64,
    grad_accum_every = 1,
    train_on_images = False, 
    use_ema = False,          
    num_train_steps = 10000,
    results_folder='results_case4',
    lr=1e-4,
    save_model_every=500,
    save_results_every=200,
    modality = 'both',
)

trainer.train()        

