import glob
from pathlib import Path 

root_path = '/home/barza/DepthContrast'
cfg_dir = 'pointnet_train_all_FOV3000_60'
cc_infos_dir = 'DepthContrast_FOV3000_Infos'
cfg_dir_path = f'{root_path}/configs/{cfg_dir}'
models = ['dc', 'vdc']
tcp_port = 18800
sbatch_file = f'{root_path}/run_sbatch.sh'

with open(sbatch_file, 'w') as f:
    f.write('#FOV3000\n')

for model in models:
    cfg_files = glob.glob(f'{cfg_dir_path}/{model}/*.yaml')
    for cfg_file in cfg_files:
        cfg_name=cfg_file.split('/')[-1].replace('.yaml', '')
        tcp_port +=1
        sbatch_cmd = f'sbatch --time=24:00:00 --array=1-2%1 --job-name={cfg_dir}-{cfg_name} --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh \
        --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/{cc_infos_dir} --tcp_port {tcp_port} --cfg_file configs/{cfg_dir}/{model}/{cfg_name}.yaml\n'

        with open(sbatch_file, 'a') as f:
            f.write(sbatch_cmd)

