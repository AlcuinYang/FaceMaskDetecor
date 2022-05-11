""" Add images into a pandas Dataframe
"""
from pathlib import Path

import pandas as pd
# from google_drive_downloader import GoogleDriveDownloader as gdd
from tqdm import tqdm


# choose your path

datasetPath = Path('/mnt/self-built-masked-face-recognition-dataset')
maskPath = datasetPath/'AFDB_masked_face_dataset'
maskPath2 = datasetPath/'train'
nonMaskPath = datasetPath/'AFDB_face_dataset'
maskDF = pd.DataFrame()

for subject in tqdm(list(maskPath.iterdir()), desc='mask photos'):
    for imgPath in subject.iterdir():
        maskDF = maskDF.append({
            'image': str(imgPath),
            'mask': 1
        }, ignore_index=True)

      
for subject in tqdm(list(maskPath2.iterdir()), desc='mask photos'):
    for imgPath in subject.iterdir():
        
        maskDF = maskDF.append({
            'image': str(imgPath),
            'mask': 1
        }, ignore_index=True)
        
for subject in tqdm(list(nonMaskPath.iterdir()), desc='non mask photos'):
    for imgPath in subject.iterdir():
        maskDF = maskDF.append({
            'image': str(imgPath),
            'mask': 0
        }, ignore_index=True)

dfName = 'covid-mask-detector/data/mask_df.csv'
print(f'saving Dataframe to: {dfName}')
maskDF.to_csv(dfName)