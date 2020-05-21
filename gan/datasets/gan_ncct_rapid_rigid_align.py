import os
import sys
from glob import glob
import SimpleITK as sitk
import time
import shutil
import numpy as np
from tqdm import tqdm

def resample_sitkImage_by_spacing(sitkImage, newSpacing, vol_default_value='min', interpolator=sitk.sitkNearestNeighbor):
    """
    :param sitkImage:
    :param newSpacing:
    :return:
    """
    if sitkImage == None:
        return None
    if newSpacing is None:
        return None

    dim = sitkImage.GetDimension()
    if len(newSpacing) != dim:
        return None

    # determine the default value
    vol_value = 0.0
    if vol_default_value == 'min':
        vol_value = float(np.ndarray.min(sitk.GetArrayFromImage(sitkImage)))
    elif vol_default_value == 'zero':
        vol_value = 0.0
    elif str(vol_default_value).isnumeric():
        vol_value = float(vol_default_value)

    # calculate new size
    np_oldSize = np.array(sitkImage.GetSize())
    np_oldSpacing = np.array(sitkImage.GetSpacing())

    np_newSpacing = np.array(newSpacing)
    np_newSize = np.divide(np.multiply(np_oldSize, np_oldSpacing), np_newSpacing)
    newSize = tuple(np_newSize.astype(np.uint).tolist())

    # resample sitkImage into new specs
    transform = sitk.Transform()

    return sitk.Resample(sitkImage, newSize, transform, interpolator, sitkImage.GetOrigin(),
                         newSpacing, sitkImage.GetDirection(), vol_value, sitkImage.GetPixelID())


def rigidRegistration_Multimodal(fixedSitkImages, movingSitkImages,
                                   interpolator=sitk.sitkNearestNeighbor,
                                   initializer='Center_GEOMETRY',
                                   transit_spacing=None):
    if fixedSitkImages == None or movingSitkImages == None:
        return None
    fixedImages = [sitk.Cast(fixedSitkImage, sitk.sitkFloat32) for fixedSitkImage in fixedSitkImages]
    movingImages = [sitk.Cast(movingSitkImage, sitk.sitkFloat32) for movingSitkImage in movingSitkImages]

    fixed_image_transits = [0]*len(fixedImages)
    if transit_spacing:
        # fixed_image_transits = [resample_sitkImage_by_spacing(fixedImage, transit_spacing,
        #                                                     vol_default_value='min',
        #                                                     interpolator=sitk.sitkLinear) for fixedImage in fixedImages]
        fixed_image_transits[0] = resample_sitkImage_by_spacing(fixedImages[0], transit_spacing,
                                                            vol_default_value=0,
                                                            interpolator=sitk.sitkLinear)
        fixed_image_transits[1] = resample_sitkImage_by_spacing(fixedImages[1], transit_spacing,
                                                            vol_default_value=-1024,
                                                            interpolator=sitk.sitkLinear)                                                    
        moving_image_transits = [resample_sitkImage_by_spacing(movingImage, transit_spacing,
                                                            vol_default_value='min',
                                                            interpolator=sitk.sitkLinear) for movingImage in movingImages]

    # initialize the tranform
    if initializer=='Center_GEOMETRY':
        transform = sitk.CenteredTransformInitializer(fixedImages[0], movingImages[0],
                                                      sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
    else:
        transform = sitk.Euler3DTransform()

    # multi-resolution rigid registration using Mutual Information
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=5, minStep=0.00001, numberOfIterations=600)
    registration_method.SetOptimizerScales((1.0, 1.0, 1.0, 1.0/500, 1.0/500, 1.0/500)) #SetOptimizerScalesFromPhysicalShift()
    #registration_method.SetOptimizerScales(scales=[1,1,1,1,1,1])
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 2, 1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(transform)
    
    fixed_image_transit = fixed_image_transits[0]
    out_fixed_image_transits = []
    out_transforms = []
    # 每种模态用相同的配准矩阵
    moving_image_transit = moving_image_transits[0]
    transform = registration_method.Execute(fixed_image_transit, moving_image_transit)
    for moving_image_transit in moving_image_transits:
        # 每种模态，都有自己的配准矩阵
        # transform = registration_method.Execute(fixed_image_transit, moving_image_transit)
        out_transforms.append(transform)
        out_fixed_image_transits.append(sitk.Resample(moving_image_transit, fixed_image_transit, transform,
                         interpolator, 0, moving_image_transit.GetPixelIDValue()))

    return fixed_image_transits, out_fixed_image_transits, out_transforms


def rigid_align_CT_DWI(fixed_files, moving_files):
    
    for moving_file in moving_files:
        if not os.path.exists(moving_file):
            print("not exist {}".format(moving_file))
            return None, None
    for fixed_file in fixed_files:
        if not os.path.exists(fixed_file):
            print("not exist {}".format(fixed_file))
            return None, None

    fixedSitkImages = []
    for fixed_file in fixed_files:
        fixedSitkImage = sitk.ReadImage(fixed_file, sitk.sitkFloat32)
        fixedSitkImages.append(fixedSitkImage)

    movingSitkImages = []
    for moving_file in moving_files:
        movingSitkImage = sitk.ReadImage(moving_file, sitk.sitkFloat32)
        movingSitkImages.append(movingSitkImage)
    
    align_spacing = fixedSitkImages[0].GetSpacing()

    align_spacing = [align_spacing[0]]*3

    #fixedSitkImage = resample_sitkImage_by_spacing(fixedSitkImage, align_spacing, vol_default_value='min', interpolator=sitk.sitkBSpline)

    fixed_image_transits, out_fixed_image_transits, out_transforms = rigidRegistration_Multimodal(fixedSitkImages, movingSitkImages,
                                   interpolator=sitk.sitkLinear,
                                   initializer='Center_GEOMETRY',
                                   transit_spacing=align_spacing)

    return fixed_image_transits, out_fixed_image_transits, out_transforms

def align_cohort_CT_DWI_single_process(input_dir, output_dir, ref_pair_files, moving_pair_files, additional_file_flag=None):
    
    os.makedirs(output_dir, exist_ok=True)


    failed_list = []
    for i, ref_pair_file in tqdm(enumerate(ref_pair_files)):
        start_time = time.time()
        print('now process the {0}th case: {1}'.format(i, os.path.basename(ref_pair_file[0])))

        ref_files = ref_pair_file
        moving_files = moving_pair_files[i]

        print('align_cohort_CT_DWI_single_process 1')

        ref_image_transits, moving_image_transits, transforms = rigid_align_CT_DWI(ref_files, moving_files)

        print('align_cohort_CT_DWI_single_process 2')

        for i in range(len(ref_files)):
            output_ref_file = os.path.join(output_dir, os.path.basename(ref_files[i]))
            sitk.WriteImage(ref_image_transits[i], output_ref_file)

        for i in range(len(moving_files)):
            output_moving_file = os.path.join(output_dir, os.path.basename(moving_files[i]))
            sitk.WriteImage(moving_image_transits[i], output_moving_file)

        print("it takes {} seconds for the current case!".format(time.time()-start_time))

    # print(failed_list)

    return True

def multiprocess_main(base_dir_fix, base_dir_mov):
    '''
    multiprocess_main('/ssd2/zhangwd/data/brain/gan/hospital_4/experiment_registration2/', '/ssd2/zhangwd/data/brain/gan/hospital_4/experiment_registration3/')
    '''
    print('Python %s on %s' % (sys.version, sys.platform))

    # base_dir = r'/ssd2/zhangwd/data/brain/gan/ncct2dwi/experiment_registration2/'
    base_dir = r'/ssd2/zhangwd/data/brain/gan/hospital_4/experiment_registration2/'
    # base_dir = r'/ssd2/zhangwd/data/brain/gan/hospital_4_2/experiment_registration2/'
    input_dir_fix = base_dir_fix + r'4 Patient_nii_unity'
    input_dir_mov = base_dir_mov + r'4 Patient_nii_unity'
    output_dir = base_dir_mov + r'5 dwi_rigid_align_ncct'
    ref_file_flag1 = r'BS_brain.nii.gz'
    ref_file_flag2 = r'BS_NCCT.nii.gz'
    moving_file_flag1 = r'FU_DWI_BXXX.nii.gz'
    moving_file_flag2 = r'FU_ISCHEMIC_PENUMBRA_MASK.nii.gz'
    moving_file_flag3 = r'FU_DWI_INFARCT_MASK.nii.gz'
    additional_file_flag = None

    os.makedirs(output_dir, exist_ok=True)

    # ref_files1 = glob(os.path.join(input_dir, '*{}'.format(ref_file_flag1)))
    # ref_files2 = [ref_file.replace(ref_file_flag1, ref_file_flag2) for ref_file in ref_files1]

    # moving_files1 = [ref_file.replace(ref_file_flag1, moving_file_flag1) for ref_file in ref_files1]
    # moving_files2 = [ref_file.replace(ref_file_flag1, moving_file_flag2) for ref_file in ref_files1]
    # moving_files3 = [ref_file.replace(ref_file_flag1, moving_file_flag3) for ref_file in ref_files1]


    moving_files2x = glob(os.path.join(input_dir_mov, '*{}'.format(moving_file_flag2)))

    ref_files1 = []
    ref_files2 = []
    moving_files1 = []
    moving_files2 = []
    moving_files3 = []
    for f in moving_files2x:
        moving_file2 = f
        basename = os.path.basename(moving_file2)
        moving_file1 = os.path.join(input_dir_fix, basename.replace(moving_file_flag2, moving_file_flag1))
        moving_file3 = os.path.join(input_dir_mov, basename.replace(moving_file_flag2, moving_file_flag3))
        
        ref_file1 = os.path.join(input_dir_fix, basename.replace(moving_file_flag2, ref_file_flag1))
        ref_file2 = os.path.join(input_dir_fix, basename.replace(moving_file_flag2, ref_file_flag2))

        print(moving_file1)
        if not os.path.isfile(moving_file1):
            continue
        if not os.path.isfile(moving_file2):
            continue
        if not os.path.isfile(moving_file3):
            continue
        if not os.path.isfile(ref_file1):
            continue
        if not os.path.isfile(ref_file2):
            continue
        ref_files1.append(ref_file1)
        ref_files2.append(ref_file2)
        moving_files1.append(moving_file1)
        moving_files2.append(moving_file2)
        moving_files3.append(moving_file3)

    ref_files = list(zip(ref_files1, ref_files2))
    moving_files = list(zip(moving_files1, moving_files2, moving_files3))

    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    process_num = 12
    results = []

    num_per_process = (len(moving_files) + process_num - 1)//process_num

    for i in range(process_num):
        sub_ref_files = ref_files[num_per_process*i:min(num_per_process*(i+1), len(ref_files)-1)]
        sub_moving_files = moving_files[num_per_process*i:min(num_per_process*(i+1), len(moving_files)-1)]

        result = pool.apply_async(align_cohort_CT_DWI_single_process, args=(input_dir_mov, output_dir, sub_ref_files, sub_moving_files, additional_file_flag))
        results.append(result)

    pool.close()
    pool.join()

def singleprocess_main():
    print('Python %s on %s' % (sys.version, sys.platform))

    base_dir = r'/ssd2/zhangwd/data/brain/gan/ncct2dwi/experiment_registration_small/'
    input_dir = base_dir + r'4 Patient_nii_unity'
    output_dir = base_dir + r'5 dwi_rigid_align_ncct'
    ref_file_flag1 = r'BS_brain.nii.gz'
    ref_file_flag2 = r'BS_NCCT.nii.gz'
    moving_file_flag3 = r'FU_ADC.nii.gz'
    moving_file_flag2 = r'FU_DWI_B0.nii.gz'
    moving_file_flag1 = r'FU_DWI_BXXX.nii.gz'
    additional_file_flag = None

    os.makedirs(output_dir, exist_ok=True)

    ref_files1 = glob(os.path.join(input_dir, '*{}'.format(ref_file_flag1)))
    ref_files2 = [ref_file.replace(ref_file_flag1, ref_file_flag2) for ref_file in ref_files1]

    moving_files1 = [ref_file.replace(ref_file_flag1, moving_file_flag1) for ref_file in ref_files1]
    moving_files2 = [ref_file.replace(ref_file_flag1, moving_file_flag2) for ref_file in ref_files1]
    moving_files3 = [ref_file.replace(ref_file_flag1, moving_file_flag3) for ref_file in ref_files1]

    ref_files = list(zip(ref_files1, ref_files2))
    moving_files = list(zip(moving_files1, moving_files2, moving_files3))

    align_cohort_CT_DWI_single_process(input_dir, output_dir, ref_files, moving_files, additional_file_flag)


if __name__ == '__main__':
    multiprocess_main('/ssd2/zhangwd/data/brain/gan/hospital_4_2/experiment_registration2/', '/ssd2/zhangwd/data/brain/gan/hospital_4_2/experiment_registration3/')
    # singleprocess_main()
