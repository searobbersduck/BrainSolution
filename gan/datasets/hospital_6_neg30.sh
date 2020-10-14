python gan_utils.py cta_extract_series_to_patient_negative_samples '../data/gan/hospital_6/ori_neg' '../data/gan/hospital_6/0.raw_dcm_neg'

python gan_utils.py ncct_convert_dcm_to_niigz_multiprocess '../data/gan/hospital_6/0.raw_dcm_neg' '../data/gan/hospital_6/experiment_registration_neg/1.nii_file_neg' 24 False

python gan_utils.py ncct_set_original_point '../data/gan/hospital_6/experiment_registration_neg/1.nii_file_neg' '../data/gan/hospital_6/experiment_registration_neg/2.nii_file_ori'

ln -s '/ssd2/zhangwd/data/brain/gan/hospital_6/experiment_registration_neg/2.nii_file_ori' '/ssd2/zhangwd/data/brain/gan/hospital_6/experiment_registration_neg/4 Patient_nii_unity'

CUDA_VISIBLE_DEVICES=2 python gan_utils.py extract_cerebral_parenchyma_multiprocess '../data/gan/hospital_6/experiment_registration_neg/4 Patient_nii_unity' '../data/gan/hospital_6/experiment_registration_neg/4 Patient_nii_unity' '_NCCT.nii.gz' '_brain.nii.gz'

python gan_utils.py change_names_batch '../data/gan/hospital_6/experiment_registration_neg/4 Patient_nii_unity' '../data/gan/hospital_6/experiment_registration_neg/4 Patient_nii_unity' '_NCCT.nii.gz' '_NCCT_bk.nii.gz'
python gan_utils.py change_names_batch '../data/gan/hospital_6/experiment_registration_neg/4 Patient_nii_unity' '../data/gan/hospital_6/experiment_registration_neg/4 Patient_nii_unity' '_NCCT_crop.nii.gz' '_NCCT.nii.gz'

python gan_cta_rigid_align.py

python gan_utils.py ncct_generate_cerebral_parenchyma_middle_layer_multiprocess '../data/gan/hospital_6/experiment_registration_neg/5 dwi_rigid_align_ncct' '../data/gan/hospital_6/experiment_registration_neg/8.1.out/cerebral_parenchyma' *_brain.nii.gz 12

python cta_to_dwi_dataset.py extract_region_by_mask_cut_multiprocess '../data/gan/hospital_6/experiment_registration_neg/8.1.out/cerebral_parenchyma' '../data/gan/hospital_6/experiment_registration_neg/5 dwi_rigid_align_ncct' '../data/gan/hospital_6/experiment_registration_neg/8.2.out/NCCT' *brain*.nii.gz *NCCT.nii.gz
python cta_to_dwi_dataset.py extract_region_by_mask_cut_multiprocess '../data/gan/hospital_6/experiment_registration_neg/8.1.out/cerebral_parenchyma' '../data/gan/hospital_6/experiment_registration_neg/5 dwi_rigid_align_ncct' '../data/gan/hospital_6/experiment_registration_neg/8.2.out/DWI_BXXX' *brain*.nii.gz *DWI_BXXX.nii.gz
python cta_to_dwi_dataset.py extract_region_by_mask_cut_multiprocess '../data/gan/hospital_6/experiment_registration_neg/8.1.out/cerebral_parenchyma' '../data/gan/hospital_6/experiment_registration_neg/8.1.out/cerebral_parenchyma' '../data/gan/hospital_6/experiment_registration_neg/8.2.out/cerebral_parenchyma' *brain*.nii.gz *brain.nii.gz

python gan_utils.py ncct_genereate_cta2dwi_config_file_with_cerebral_parenchyma ../data/gan/hospital_6/experiment_registration_neg/8.2.out ../data/gan/hospital_6/experiment_registration_neg/8.2.out/config