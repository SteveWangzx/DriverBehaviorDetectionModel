"""
    This script contains the process of saving image information (Label, Path) into a csv file
    For Training set Only.

    Author: Zhixiang Wang
    Date: April 12, 2023
"""

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from IPython.display import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

for dirname, _, filenames in os.walk('Revitsone-5classes'):
    for filename in filenames:
        os.path.join(dirname, filename)

# dir path
BASE_DIR = 'Revitsone-5classes/'
other_activities = BASE_DIR + 'other_activities/'
safe_driving = BASE_DIR + 'safe_driving/'
talking_phone = BASE_DIR + 'talking_phone/'
texting_phone = BASE_DIR + 'texting_phone/'
turning = BASE_DIR + 'turning/'

files_in_other_activities = sorted(os.listdir(other_activities))
files_in_safe_driving = sorted(os.listdir(safe_driving))
files_in_talking_phone = sorted(os.listdir(talking_phone))
files_in_texting_phone = sorted(os.listdir(texting_phone))
files_in_turning = sorted(os.listdir(turning))

other_activity_images = ['other_activities/' + i for i in files_in_other_activities]
safe_driving_images = ['safe_driving/' + i for i in files_in_safe_driving]
talking_phone_images = ['talking_phone/' + i for i in files_in_talking_phone]
texting_phone_images = ['texting_phone/' + i for i in files_in_texting_phone]
turning_images = ['turning/' + i for i in files_in_turning]

images = other_activity_images + safe_driving_images + talking_phone_images + texting_phone_images + turning_images

df = pd.DataFrame()
df['column'] = [str(x) for x in images]

df[['activity_type', 'image_id']] = df['column'].str.split('/', expand=True)
df['path'] = BASE_DIR + df['column']
df = df.drop(columns='column', axis=0)

print(df.head())

plt.title("Test Image")
plt.xlabel("X pixel scaling")
plt.ylabel("Y pixels scaling")

image = mpimg.imread(talking_phone + df['image_id'].iloc[6100])
plt.imshow(image)
plt.show()

df.to_csv('driver_behavior.csv', index=True)
