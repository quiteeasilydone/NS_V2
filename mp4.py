import os
import glob


path = '../data/video/'
output = glob.glob(os.path.join(path, '*.mp4'))

for file in output:
    if 'vehicle' in file:
        new_name = file.replace('..mp4','.mp4')
        os.rename(file, new_name)