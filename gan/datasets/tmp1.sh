# CUDA_VISIBLE_DEVICES=4 python gan_utils.py extract_cerebral_parenchyma_multiprocess '../data/gan/hospital_4_2/experiment_registration2/4 Patient_nii_unity' '../data/gan/hospital_4_2/experiment_registration2/4 Patient_nii_unity' _NCCT.nii.gz _brain.nii.gz
# python gan_utils.py change_names_batch '../data/gan/hospital_4_2/experiment_registration2/4 Patient_nii_unity' '../data/gan/hospital_4_2/experiment_registration2/4 Patient_nii_unity' '_NCCT.nii.gz' '_NCCT_bk.nii.gz'
# python gan_utils.py change_names_batch '../data/gan/hospital_4_2/experiment_registration2/4 Patient_nii_unity' '../data/gan/hospital_4_2/experiment_registration2/4 Patient_nii_unity' '_NCCT_crop.nii.gz' '_NCCT.nii.gz'
# python gan_ncct_rigid_align_4_2.py


# python gan_utils.py ncct_set_original_point '../data/gan/hospital_4/experiment_registration2/1.nii_file' '../data/gan/hospital_4/experiment_registration2/2.nii_file_ori'
# ln -s '/ssd2/zhangwd/data/brain/gan/hospital_4/experiment_registration2/2.nii_file_ori' '/ssd2/zhangwd/data/brain/gan/hospital_4/experiment_registration2/4 Patient_nii_unity'
# CUDA_VISIBLE_DEVICES=5 python gan_utils.py extract_cerebral_parenchyma_multiprocess '../data/gan/hospital_4/experiment_registration2/4 Patient_nii_unity' '../data/gan/hospital_4/experiment_registration2/4 Patient_nii_unity' _NCCT.nii.gz _brain.nii.gz
# python gan_utils.py change_names_batch '../data/gan/hospital_4/experiment_registration2/4 Patient_nii_unity' '../data/gan/hospital_4/experiment_registration2/4 Patient_nii_unity' '_NCCT.nii.gz' '_NCCT_bk.nii.gz'
# python gan_utils.py change_names_batch '../data/gan/hospital_4/experiment_registration2/4 Patient_nii_unity' '../data/gan/hospital_4/experiment_registration2/4 Patient_nii_unity' '_NCCT_crop.nii.gz' '_NCCT.nii.gz'
python gan_ncct_rigid_align.py

python gan_utils.py ncct_generate_cerebral_parenchyma_middle_layer_only_multiprocess '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.7.out/cerebral_parenchyma' *_brain.nii.gz 12
python cta_to_dwi_dataset.py extract_region_by_mask_cut_only_multiprocess '../data/gan/hospital_4/experiment_registration2/8.7.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.8.out/NCCT' *brain*.nii.gz *NCCT.nii.gz
python cta_to_dwi_dataset.py extract_region_by_mask_cut_only_multiprocess '../data/gan/hospital_4/experiment_registration2/8.7.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.8.out/DWI_B0' *brain*.nii.gz *DWI_B0.nii.gz
python cta_to_dwi_dataset.py extract_region_by_mask_cut_only_multiprocess '../data/gan/hospital_4/experiment_registration2/8.7.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.8.out/DWI_BXXX' *brain*.nii.gz *DWI_BXXX.nii.gz
python cta_to_dwi_dataset.py extract_region_by_mask_cut_only_multiprocess '../data/gan/hospital_4/experiment_registration2/8.7.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.8.out/ADC' *brain*.nii.gz *ADC.nii.gz
python cta_to_dwi_dataset.py extract_region_by_mask_cut_only_multiprocess '../data/gan/hospital_4/experiment_registration2/8.7.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/8.7.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/8.8.out/cerebral_parenchyma' *brain*.nii.gz *brain.nii.gz



