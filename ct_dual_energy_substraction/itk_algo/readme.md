## reference
1. [Threshold An Image Using Binary Thresholding](https://itk.org/ITKExamples/src/Filtering/Thresholding/ThresholdAnImageUsingBinary/Documentation.html)
2. [Threshold An Image Using Otsu](https://itk.org/ITKExamples/src/Filtering/Thresholding/ThresholdAnImageUsingOtsu/Documentation.html)
3. [Image Segmentation with Python and SimpleITK](https://pyscience.wordpress.com/2014/10/19/image-segmentation-with-python-and-simpleitk/)
4. [InsightSoftwareConsortium/SimpleITK-Notebooks](https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/tree/master/Python)
5. [SimpleITK-Notebooks/Python/30_Segmentation_Region_Growing.ipynb](https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/30_Segmentation_Region_Growing.ipynb)
   1. [Segmentation: Region Growing](http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/30_Segmentation_Region_Growing.html)
6. [itk::simple::ConnectedComponentImageFilter Class Reference](https://itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1ConnectedComponentImageFilter.html)
   1. Label the objects in a binary image.

   2. ConnectedComponentImageFilter labels the objects in a binary image (non-zero pixels are considered to be objects, zero-valued pixels are considered to be background). Each distinct object is assigned a unique label. The filter experiments with some improvements to the existing implementation, and is based on run length encoding along raster lines. The final object labels start with 1 and are consecutive. Objects that are reached earlier by a raster order scan have a lower label. This is different to the behaviour of the original connected component image filter which did not produce consecutive labels or impose any particular ordering.

   3. After the filter is executed, ObjectCount holds the number of connected components.
7. [SimpleITK Filters](https://simpleitk.readthedocs.io/en/latest/Documentation/docs/source/filters.html)
8. [How to extract labels from a Binary Image in SimpleITK in python](https://stackoverflow.com/questions/40720176/how-to-extract-labels-from-a-binary-image-in-simpleitk-in-python)
9. [SimpleITK not working with any data type, be it 3D or 2D](https://github.com/SimpleITK/SimpleITK/issues/592)
   * 数据类型转换   
10. [Python SimpleITK.WriteImage() Examples](https://www.programcreek.com/python/example/96382/SimpleITK.WriteImage)
11. [How to get Meta data of Dicom image in SimpleITK using Python](https://stackoverflow.com/questions/46984220/how-to-get-meta-data-of-dicom-image-in-simpleitk-using-python)