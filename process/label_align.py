import pandas as pd
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import random

def align_label(common_name, label_name):
    label = pd.read_csv(label_name)
    label_dict = label.to_dict('records')
    common_label = []
    for label in label_dict:
        if label["name"] in common_name:
            common_label.append(label)
    return pd.DataFrame(common_label)

def common_seg(audio_dir, video_dir):
    audio_files = {file.split('.')[0] for file in os.listdir(audio_dir) if file.endswith('.wav')}
    video_files = {file.split('.')[0] for file in os.listdir(video_dir) if file.endswith('.mp4')}
    common_files = audio_files.intersection(video_files)
    return list(common_files)

def reduce_neutral_label(final_df):
    zero_label_rows = final_df[final_df['label'] == 0]
    num_to_remove = len(zero_label_rows) // 2
    rows_to_remove = random.sample(list(zero_label_rows.index), num_to_remove)
    final_df = final_df.drop(rows_to_remove)
    return final_df

if __name__ == '__main__':
    audio_dir = '/project/msoleyma_1026/mosei_edit/audio_seg'
    video_dir = '/project/msoleyma_1026/mosei_edit/video_seg'
    label_name = '/project/msoleyma_1026/mosei_edit/mosei_label.csv'
    common_files = common_seg(audio_dir, video_dir)
    common_label = align_label(common_files, label_name)
    res_label = reduce_neutral_label(common_label)
    print(len(res_label))
    print(res_label["label"].value_counts())
    res_label.to_csv('/scratch1/rliu0866/mosei_label_common.csv', index=False)

    