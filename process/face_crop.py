from MARLIN.util.face_sdk.face_crop import process_videos
# git clone from https://github.com/ControlNet/MARLIN
# make sure the import is correct (files under files)
# make sure the path is correct
# make sure disable all logging

if __name__ == '__main__':
    # cremad
    #process_videos('/home/tangyimi/emotion/data/cremad/VideoMp4', '/home/tangyimi/emotion/data/cremad/cropped_face', ext="mp4")

    # mosei
    process_videos('/project/msoleyma_1026/CMU-MOSEI/MOSEI/Videos/Full/Combined', '/project/msoleyma_1026/MOSEI_cropped_face1', ext="mp4")