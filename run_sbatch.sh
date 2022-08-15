# # 360 deg shortlist
# sbatch --time=24:00:00 --array=1-1%1 --job-name=pointnet_train_all_60-dc --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18851 --cfg_file configs/pointnet_train_all_60/dc/shortlist/dc_360deg.yaml
# sbatch --time=24:00:00 --array=1-1%1 --job-name=pointnet_train_all_60-dc_snow1in2_wet_fog1in2_cube_upsampleF_360deg --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18852 --cfg_file configs/pointnet_train_all_60/dc/shortlist/dc_snow1in2_wet_fog1in2_cube_upsampleF_360deg.yaml
# sbatch --time=24:00:00 --array=1-1%1 --job-name=pointnet_train_all_60-dc_snow1in2_wet_fog1in2_cubeF_upsampleF_360deg --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18853 --cfg_file configs/pointnet_train_all_60/dc/shortlist/dc_snow1in2_wet_fog1in2_cubeF_upsampleF_360deg.yaml
# sbatch --time=24:00:00 --array=1-1%1 --job-name=pointnet_train_all_60-dc_snow1in10_wet_fog1in10_cube_upsampleF_360deg --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18854 --cfg_file configs/pointnet_train_all_60/dc/shortlist/dc_snow1in10_wet_fog1in10_cube_upsampleF_360deg.yaml
# sbatch --time=24:00:00 --array=1-1%1 --job-name=pointnet_train_all_60-dc_snow1in10_wet_fog1in10_cubeF_upsampleF_360deg --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18855 --cfg_file configs/pointnet_train_all_60/dc/shortlist/dc_snow1in10_wet_fog1in10_cubeF_upsampleF_360deg.yaml

# FOV3000

# #DC
# sbatch --time=24:00:00 --array=1-3%1 --job-name=pointnet_train_all_FOV3000_60-dc --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18856 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/shortlist/dc_FOV3000.yaml
# sbatch --time=24:00:00 --array=1-3%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow1in2_wet_fog1in2_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18858 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/shortlist/dc_snow1in2_wet_fog1in2_cubeF_upsampleF_FOV3000.yaml
# sbatch --time=24:00:00 --array=1-3%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow1in10_wet_fog1in10_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18859 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/shortlist/dc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000.yaml
# sbatch --time=24:00:00 --array=1-3%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow1in2_wet_fog1in2_cube_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18857 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/shortlist/dc_snow1in2_wet_fog1in2_cube_upsampleF_FOV3000.yaml

# #VDC (cubeF!)
# sbatch --time=24:00:00 --array=1-4%1 --job-name=pointnet_train_all_FOV3000_60-vdc --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18856 --cfg_file configs/pointnet_train_all_FOV3000_60/vdc/shortlist/vdc_FOV3000.yaml

# sbatch --time=24:00:00 --array=1-4%1 --job-name=pointnet_train_all_FOV3000_60-vdc_snow1in2_wet_fog1in2_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18857 --cfg_file configs/pointnet_train_all_FOV3000_60/vdc/shortlist/vdc_snow1in2_wet_fog1in2_cubeF_upsampleF_FOV3000.yaml
# sbatch --time=24:00:00 --array=1-4%1 --job-name=pointnet_train_all_FOV3000_60-vdc_snow1in10_wet_fog1in10_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18858 --cfg_file configs/pointnet_train_all_FOV3000_60/vdc/shortlist/vdc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000.yaml

# sbatch --time=24:00:00 --array=1-4%1 --job-name=pointnet_train_all_FOV3000_60-vdc_snow1in2_wet_fog1in2_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18859 --cfg_file configs/pointnet_train_all_FOV3000_60/vdc/shortlist/vdc_snow1in2_wet_fog1in2_cubeF_upsample_FOV3000.yaml
# sbatch --time=24:00:00 --array=1-4%1 --job-name=pointnet_train_all_FOV3000_60-vdc_snow1in10_wet_fog1in10_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18860 --cfg_file configs/pointnet_train_all_FOV3000_60/vdc/shortlist/vdc_snow1in10_wet_fog1in10_cubeF_upsample_FOV3000.yaml


# #DC-VDC (cubeF!)
# sbatch --time=24:00:00 --array=1-4%1 --job-name=pointnet_train_all_FOV3000_60-dc-vdc --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18861 --cfg_file configs/pointnet_train_all_FOV3000_60/dc_vdc/shortlist/dc_vdc_FOV3000.yaml

# sbatch --time=24:00:00 --array=1-4%1 --job-name=pointnet_train_all_FOV3000_60-dc-vdc_snow1in2_wet_fog1in2_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18862 --cfg_file configs/pointnet_train_all_FOV3000_60/dc_vdc/shortlist/dc_vdc_snow1in2_wet_fog1in2_cubeF_upsampleF_FOV3000.yaml
# sbatch --time=24:00:00 --array=1-4%1 --job-name=pointnet_train_all_FOV3000_60-dc-vdc_snow1in10_wet_fog1in10_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18863 --cfg_file configs/pointnet_train_all_FOV3000_60/dc_vdc/shortlist/dc_vdc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000.yaml

# sbatch --time=24:00:00 --array=1-4%1 --job-name=pointnet_train_all_FOV3000_60-dc-vdc_snow1in2_wet_fog1in2_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18864 --cfg_file configs/pointnet_train_all_FOV3000_60/dc_vdc/shortlist/dc_vdc_snow1in2_wet_fog1in2_cubeF_upsample_FOV3000.yaml
# sbatch --time=24:00:00 --array=1-4%1 --job-name=pointnet_train_all_FOV3000_60-dc-vdc_snow1in10_wet_fog1in10_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18865 --cfg_file configs/pointnet_train_all_FOV3000_60/dc_vdc/shortlist/dc_vdc_snow1in10_wet_fog1in10_cubeF_upsample_FOV3000.yaml


#VDC and DC-VDC(cubeF!)
sbatch --time=24:00:00 --array=1-1%1 --job-name=pointnet_train_all_FOV3000_60-vdc --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18856 --cfg_file configs/pointnet_train_all_FOV3000_60/vdc/shortlist/vdc_FOV3000.yaml
#sbatch --time=24:00:00 --array=1-4%1 --job-name=pointnet_train_all_FOV3000_60-dc-vdc --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18861 --cfg_file configs/pointnet_train_all_FOV3000_60/dc_vdc/shortlist/dc_vdc_FOV3000.yaml

sbatch --time=24:00:00 --array=1-1%1 --job-name=pointnet_train_all_FOV3000_60-vdc_snow1in2_wet_fog1in2_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18857 --cfg_file configs/pointnet_train_all_FOV3000_60/vdc/shortlist/vdc_snow1in2_wet_fog1in2_cubeF_upsampleF_FOV3000.yaml
#sbatch --time=24:00:00 --array=1-4%1 --job-name=pointnet_train_all_FOV3000_60-dc-vdc_snow1in2_wet_fog1in2_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18862 --cfg_file configs/pointnet_train_all_FOV3000_60/dc_vdc/shortlist/dc_vdc_snow1in2_wet_fog1in2_cubeF_upsampleF_FOV3000.yaml
sbatch --time=24:00:00 --array=1-1%1 --job-name=pointnet_train_all_FOV3000_60-vdc_snow1in10_wet_fog1in10_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18858 --cfg_file configs/pointnet_train_all_FOV3000_60/vdc/shortlist/vdc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000.yaml
#sbatch --time=24:00:00 --array=1-4%1 --job-name=pointnet_train_all_FOV3000_60-dc-vdc_snow1in10_wet_fog1in10_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18863 --cfg_file configs/pointnet_train_all_FOV3000_60/dc_vdc/shortlist/dc_vdc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000.yaml

sbatch --time=24:00:00 --array=1-1%1 --job-name=pointnet_train_all_FOV3000_60-vdc_snow1in2_wet_fog1in2_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18859 --cfg_file configs/pointnet_train_all_FOV3000_60/vdc/shortlist/vdc_snow1in2_wet_fog1in2_cubeF_upsample_FOV3000.yaml
#sbatch --time=24:00:00 --array=1-4%1 --job-name=pointnet_train_all_FOV3000_60-dc-vdc_snow1in2_wet_fog1in2_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18864 --cfg_file configs/pointnet_train_all_FOV3000_60/dc_vdc/shortlist/dc_vdc_snow1in2_wet_fog1in2_cubeF_upsample_FOV3000.yaml
sbatch --time=24:00:00 --array=1-1%1 --job-name=pointnet_train_all_FOV3000_60-vdc_snow1in10_wet_fog1in10_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18860 --cfg_file configs/pointnet_train_all_FOV3000_60/vdc/shortlist/vdc_snow1in10_wet_fog1in10_cubeF_upsample_FOV3000.yaml
#sbatch --time=24:00:00 --array=1-4%1 --job-name=pointnet_train_all_FOV3000_60-dc-vdc_snow1in10_wet_fog1in10_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh   --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18865 --cfg_file configs/pointnet_train_all_FOV3000_60/dc_vdc/shortlist/dc_vdc_snow1in10_wet_fog1in10_cubeF_upsample_FOV3000.yaml


#FOV3000
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_coupled_fogF_cube_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18801 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_coupled_fogF_cube_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snowF_wet_fog_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18802 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snowF_wet_fog_cubeF_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_wet_fogF_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18803 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_wet_fogF_cubeF_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_wetF_fogF_cube_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18804 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_wetF_fogF_cube_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_wet_fog_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18805 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_wet_fog_cubeF_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_coupled_fogF_cube_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18806 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_coupled_fogF_cube_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_wet_fog_cube_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18807 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_wet_fog_cube_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snowF_wet_fog_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18808 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snowF_wet_fog_cubeF_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_coupled_fog_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18809 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_coupled_fog_cubeF_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_wetF_fogF_cube_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18810 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_wetF_fogF_cube_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_wet_fogF_cube_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18811 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_wet_fogF_cube_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_coupled_fogF_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18812 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_coupled_fogF_cubeF_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_coupled_fog_cube_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18813 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_coupled_fog_cube_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_wet_fogF_cube_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18814 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_wet_fogF_cube_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snowF_wetF_fog_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18815 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snowF_wetF_fog_cubeF_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_wetF_fogF_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18816 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_wetF_fogF_cubeF_upsampleF.yaml
# sbatch --time=5:00:00 --array=1-6%1 --job-name=pointnet_train_all_FOV3000_60-dc --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18817 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_coupled_fogF_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18818 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_coupled_fogF_cubeF_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_wet_fogF_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18819 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_wet_fogF_cubeF_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_coupled_fog_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18820 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_coupled_fog_cubeF_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_coupled_fog_cube_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18821 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_coupled_fog_cube_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_wetF_fogF_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18822 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_wetF_fogF_cubeF_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snowF_wetF_fog_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18823 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snowF_wetF_fog_cubeF_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_wet_fog_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18824 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_wet_fog_cubeF_upsampleF.yaml
# sbatch --time=5:00:00 --array=1-6%1 --job-name=pointnet_train_all_FOV3000_60-dc_snow_wet_fog_cube_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_FOV3000_Infos --tcp_port 18825 --cfg_file configs/pointnet_train_all_FOV3000_60/dc/dc_snow_wet_fog_cube_upsampleF.yaml
#360deg
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_coupled_fogF_cube_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18826 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_coupled_fogF_cube_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snowF_wet_fog_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18827 --cfg_file configs/pointnet_train_all_60/dc/dc_snowF_wet_fog_cubeF_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_wet_fogF_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18828 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_wet_fogF_cubeF_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_wetF_fogF_cube_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18829 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_wetF_fogF_cube_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_wet_fog_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18830 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_wet_fog_cubeF_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_coupled_fogF_cube_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18831 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_coupled_fogF_cube_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_wet_fog_cube_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18832 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_wet_fog_cube_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snowF_wet_fog_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18833 --cfg_file configs/pointnet_train_all_60/dc/dc_snowF_wet_fog_cubeF_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_coupled_fog_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18834 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_coupled_fog_cubeF_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_wetF_fogF_cube_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18835 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_wetF_fogF_cube_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_wet_fogF_cube_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18836 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_wet_fogF_cube_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_coupled_fogF_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18837 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_coupled_fogF_cubeF_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_coupled_fog_cube_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18838 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_coupled_fog_cube_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_wet_fogF_cube_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18839 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_wet_fogF_cube_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snowF_wetF_fog_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18840 --cfg_file configs/pointnet_train_all_60/dc/dc_snowF_wetF_fog_cubeF_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_wetF_fogF_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18841 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_wetF_fogF_cubeF_upsampleF.yaml
# sbatch --time=5:00:00 --array=1-6%1 --job-name=pointnet_train_all_60-dc --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18842 --cfg_file configs/pointnet_train_all_60/dc/dc.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_coupled_fogF_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18843 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_coupled_fogF_cubeF_upsampleF.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_wet_fogF_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18844 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_wet_fogF_cubeF_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_coupled_fog_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18845 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_coupled_fog_cubeF_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_coupled_fog_cube_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18846 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_coupled_fog_cube_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_wetF_fogF_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18847 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_wetF_fogF_cubeF_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snowF_wetF_fog_cubeF_upsample --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18848 --cfg_file configs/pointnet_train_all_60/dc/dc_snowF_wetF_fog_cubeF_upsample.yaml
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointnet_train_all_60-dc_snow_wet_fog_cubeF_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18849 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_wet_fog_cubeF_upsampleF.yaml
# sbatch --time=5:00:00 --array=1-6%1 --job-name=pointnet_train_all_60-dc_snow_wet_fog_cube_upsampleF --mail-user=barzanisar93@gmail.com scripts/compute_canada_train_depth_contrast_dense.sh         --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/DepthContrast_Infos --tcp_port 18850 --cfg_file configs/pointnet_train_all_60/dc/dc_snow_wet_fog_cube_upsampleF.yaml
