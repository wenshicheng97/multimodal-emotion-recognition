from MARLIN.util.face_sdk.face_crop import process_videos
# git clone from https://github.com/ControlNet/MARLIN
# make sure the import is correct (files under files)
# make sure the path is correct
# make sure disable all logging

if __name__ == '__main__':
    process_videos('/home/tangyimi/emotion/data/cremad/VideoMp4', '/home/tangyimi/emotion/data/cremad/cropped_face', ext="mp4")