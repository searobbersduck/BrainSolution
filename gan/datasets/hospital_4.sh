# python gan_utils.py ncct_extract_infos_from_patients_all '../data/gan/hospital_4/0.ori' '../data/gan/hospital_4/0.raw_dcm'
# python gan_utils.py ncct_convert_dcm_to_niigz_multiprocess '../data/gan/hospital_4/0.raw_dcm' '../data/gan/hospital_4/experiment_registration2/1.nii_file'
python gan_utils.py ncct_set_original_point '../data/gan/hospital_4/experiment_registration2/1.nii_file' '../data/gan/hospital_4/experiment_registration2/2.nii_file_ori'
ln -s '/ssd2/zhangwd/data/brain/gan/hospital_4/experiment_registration2/2.nii_file_ori' '/ssd2/zhangwd/data/brain/gan/hospital_4/experiment_registration2/4 Patient_nii_unity'
python gan_utils.py extract_cerebral_parenchyma_multiprocess '../data/gan/hospital_4/experiment_registration2/4 Patient_nii_unity' '../data/gan/hospital_4/experiment_registration2/4 Patient_nii_unity' _NCCT.nii.gz _brain.nii.gz

python gan_ncct_rigid_align.py

python gan_utils.py ncct_generate_cerebral_parenchyma_multiprocess '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.out/cerebral_parenchyma' *_brain.nii.gz 12

python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/hospital_4/experiment_registration2/8.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.out/NCCT' *brain*.nii.gz *NCCT.nii.gz
python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/hospital_4/experiment_registration2/8.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.out/DWI_B0' *brain*.nii.gz *DWI_B0.nii.gz
python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/hospital_4/experiment_registration2/8.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.out/DWI_BXXX' *brain*.nii.gz *DWI_BXXX.nii.gz
python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/hospital_4/experiment_registration2/8.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.out/ADC' *brain*.nii.gz *ADC.nii.gz


# python gan_utils.py ncct_generate_cerebral_parenchyma_middle_layer_multiprocess '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.1.out/cerebral_parenchyma' *_brain.nii.gz 12
# python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/hospital_4/experiment_registration2/8.1.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.1.out/NCCT' *brain*.nii.gz *NCCT.nii.gz
# python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/hospital_4/experiment_registration2/8.1.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.1.out/DWI_B0' *brain*.nii.gz *DWI_B0.nii.gz
# python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/hospital_4/experiment_registration2/8.1.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.1.out/DWI_BXXX' *brain*.nii.gz *DWI_BXXX.nii.gz
# python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/hospital_4/experiment_registration2/8.1.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.1.out/ADC' *brain*.nii.gz *ADC.nii.gz
# python gan_utils.py ncct_genereate_cta2dwi_config_file_with_cerebral_parenchyma ../data/gan/hospital_4/experiment_registration2/8.1.out ../data/gan/hospital_4/experiment_registration2/8.1.out/config
# python cta_to_dwi_dataset.py calculate_crop_range ../data/gan/hospital_4/experiment_registration2/8.1.out/config/ncct_to_dwi_b0_train_config_file.txt


python gan_utils.py ncct_genereate_cta2dwi_config_file_with_cerebral_parenchyma ../data/gan/hospital_4/experiment_registration2/8.out ../data/gan/hospital_4/experiment_registration2/8.out/config

python cta_to_dwi_dataset.py calculate_crop_range ../data/gan/hospital_4/experiment_registration2/8.out/config/ncct_to_dwi_b0_train_config_file.txt