# sbatch --time=5:00:00 --array=1-1%1 --job-name=precompute_test_split-martin_index0 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_martin.txt --snowfall_rate_index 0
# sbatch --time=5:00:00 --array=1-1%1 --job-name=precompute_test_split-martin_index0 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_martin.txt --snowfall_rate_index 1

sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split0_index0 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_0.txt --snowfall_rate_index 0
sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split1_index0 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_1.txt --snowfall_rate_index 0
sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split2_index0 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_2.txt --snowfall_rate_index 0
sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split3_index0 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_3.txt --snowfall_rate_index 0
sbatch --time=4:00:00 --array=1-1%1 --job-name=precompute_test_split4_index0 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_4.txt --snowfall_rate_index 0

# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split0_index1 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_0.txt --snowfall_rate_index 1
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split1_index1 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_1.txt --snowfall_rate_index 1
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split2_index1 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_2.txt --snowfall_rate_index 1
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split3_index1 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_3.txt --snowfall_rate_index 1
# sbatch --time=4:00:00 --array=1-1%1 --job-name=precompute_test_split4_index1 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_4.txt --snowfall_rate_index 1

# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split0_index2 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_0.txt --snowfall_rate_index 2
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split1_index2 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_1.txt --snowfall_rate_index 2
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split2_index2 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_2.txt --snowfall_rate_index 2
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split3_index2 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_3.txt --snowfall_rate_index 2
# sbatch --time=4:00:00 --array=1-1%1 --job-name=precompute_test_split4_index2 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_4.txt --snowfall_rate_index 2

# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split0_index3 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_0.txt --snowfall_rate_index 3
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split1_index3 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_1.txt --snowfall_rate_index 3
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split2_index3 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_2.txt --snowfall_rate_index 3
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split3_index3 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_3.txt --snowfall_rate_index 3
# sbatch --time=4:00:00 --array=1-1%1 --job-name=precompute_test_split4_index3 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_4.txt --snowfall_rate_index 3

# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split0_index4 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_0.txt --snowfall_rate_index 4
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split1_index4 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_1.txt --snowfall_rate_index 4
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split2_index4 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_2.txt --snowfall_rate_index 4
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split3_index4 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_3.txt --snowfall_rate_index 4
# sbatch --time=4:00:00 --array=1-1%1 --job-name=precompute_test_split4_index4 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_4.txt --snowfall_rate_index 4

# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split0_index5 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_0.txt --snowfall_rate_index 5
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split1_index5 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_1.txt --snowfall_rate_index 5
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split2_index5 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_2.txt --snowfall_rate_index 5
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split3_index5 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_3.txt --snowfall_rate_index 5
# sbatch --time=4:00:00 --array=1-1%1 --job-name=precompute_test_split4_index5 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_4.txt --snowfall_rate_index 5

# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split0_index6 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_0.txt --snowfall_rate_index 6
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split1_index6 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_1.txt --snowfall_rate_index 6
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split2_index6 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_2.txt --snowfall_rate_index 6
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split3_index6 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_3.txt --snowfall_rate_index 6
# sbatch --time=4:00:00 --array=1-1%1 --job-name=precompute_test_split4_index6 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_4.txt --snowfall_rate_index 6

# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split0_index7 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_0.txt --snowfall_rate_index 7
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split1_index7 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_1.txt --snowfall_rate_index 7
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split2_index7 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_2.txt --snowfall_rate_index 7
# sbatch --time=3:00:00 --array=1-1%1 --job-name=precompute_test_split3_index7 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_3.txt --snowfall_rate_index 7
# sbatch --time=4:00:00 --array=1-1%1 --job-name=precompute_test_split4_index7 --mail-user=barzanisar93@gmail.com scripts/compute_canada_lidar_snow_sim.sh --split train_clear_FOV3000_60_4.txt --snowfall_rate_index 7