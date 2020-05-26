#python gan_utils.py cta_extract_series_to_patient '../data/gan/hospital_6/ori', '../data/gan/hospital_6/0.raw_dcm', '../data/gan/hospital_6/2 排除约40人后新增32人基于原六院CTA2DWI.xlsx'
# python gan_utils.py ncct_convert_dcm_to_niigz_multiprocess '../data/gan/hospital_6/0.raw_dcm' '../data/gan/hospital_6/experiment_registration2/1.nii_file' 24 False

python gan_utils.py ncct_set_original_point '../data/gan/hospital_6/experiment_registration2/1.nii_file' '../data/gan/hospital_6/experiment_registration2/2.nii_file_ori'

ln -s '/ssd2/zhangwd/data/brain/gan/hospital_6/experiment_registration2/2.nii_file_ori' '/ssd2/zhangwd/data/brain/gan/hospital_6/experiment_registration2/4 Patient_nii_unity'

CUDA_VISIBLE_DEVICES=2 python gan_utils.py extract_cerebral_parenchyma_multiprocess '../data/gan/hospital_6/experiment_registration2/4 Patient_nii_unity' '../data/gan/hospital_6/experiment_registration2/4 Patient_nii_unity' _NCCT.nii.gz _brain.nii.gz

# python gan_utils.py change_names_batch '../data/gan/hospital_6/experiment_registration2/4 Patient_nii_unity' '../data/gan/hospital_6/experiment_registration2/4 Patient_nii_unity' '_NCCT.nii.gz' '_NCCT_bk.nii.gz'
# python gan_utils.py change_names_batch '../data/gan/hospital_6/experiment_registration2/4 Patient_nii_unity' '../data/gan/hospital_6/experiment_registration2/4 Patient_nii_unity' '_NCCT_crop.nii.gz' '_NCCT.nii.gz'

# python gan_cta_rigid_align.py



# python gan_utils.py ncct_generate_cerebral_parenchyma_multiprocess '../data/gan/hospital_6/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_6/experiment_registration2/8.out/cerebral_parenchyma' *_brain.nii.gz 12

# python gan_utils.py ncct_generate_cerebral_parenchyma_middle_layer_multiprocess '../data/gan/hospital_6/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_6/experiment_registration2/8.1.out/cerebral_parenchyma' *_brain.nii.gz 12

# python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/hospital_6/experiment_registration2/8.1.out/cerebral_parenchyma' '../data/gan/hospital_6/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_6/experiment_registration2/8.1.out/NCCT' *brain*.nii.gz *NCCT.nii.gz
# python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/hospital_6/experiment_registration2/8.1.out/cerebral_parenchyma' '../data/gan/hospital_6/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_6/experiment_registration2/8.1.out/DWI_B0' *brain*.nii.gz *DWI_B0.nii.gz
# python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/hospital_6/experiment_registration2/8.1.out/cerebral_parenchyma' '../data/gan/hospital_6/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_6/experiment_registration2/8.1.out/DWI_BXXX' *brain*.nii.gz *DWI_BXXX.nii.gz

# python cta_to_dwi_dataset.py extract_region_by_mask_cut_multiprocess '../data/gan/hospital_6/experiment_registration2/8.1.out/cerebral_parenchyma' '../data/gan/hospital_6/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_6/experiment_registration2/8.2.out/NCCT' *brain*.nii.gz *NCCT.nii.gz
# python cta_to_dwi_dataset.py extract_region_by_mask_cut_multiprocess '../data/gan/hospital_6/experiment_registration2/8.1.out/cerebral_parenchyma' '../data/gan/hospital_6/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_6/experiment_registration2/8.2.out/DWI_BXXX' *brain*.nii.gz *DWI_BXXX.nii.gz
# python cta_to_dwi_dataset.py extract_region_by_mask_cut_multiprocess '../data/gan/hospital_6/experiment_registration2/8.1.out/cerebral_parenchyma' '../data/gan/hospital_6/experiment_registration2/8.1.out/cerebral_parenchyma' '../data/gan/hospital_6/experiment_registration2/8.2.out/cerebral_parenchyma' *brain*.nii.gz *brain.nii.gz


# python gan_utils.py ncct_genereate_cta2dwi_config_file_with_cerebral_parenchyma ../data/gan/hospital_6/experiment_registration2/8.1.out ../data/gan/hospital_6/experiment_registration2/8.1.out/config