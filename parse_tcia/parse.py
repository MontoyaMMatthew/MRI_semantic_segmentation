import SimpleITK as sitk
from pathlib import Path

# /Volumes/X9 Pro/CS/appliedML/Data/manifest-1695134609823/ReMIND/ReMIND-001/12-25-1982-NA-Intraop-90478/1.000000-USpredura-64615
file_path = "/Volumes/X9 Pro/CS/appliedML/Data/manifest-1695134609823/ReMIND/ReMIND-001"
reader = sitk.ImageSeriesReader()
dicom_series = reader.GetGDCMSeriesFileNames(file_path, recursive=True)
print(dicom_series)
reader.SetFileNames(dicom_series)
image = reader.Execute()

# Convert SimpleITK image to a NumPy array
mri_data = sitk.GetArrayFromImage(image)

print(mri_data.shape)
print(mri_data)
