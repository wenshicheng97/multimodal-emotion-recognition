import subprocess
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from tqdm import tqdm

def segment_video(input_file, start_time, end_time, output_file):
    # Calculate duration from start to end times
    duration = end_time - start_time
    
    # Build the FFmpeg command to cut the video
    command = [
        'ffmpeg',
        '-ss', str(start_time),    # The start time for the cut
        '-to', str(end_time),      # The end time for the cut
        '-i', input_file,          # The input file
        '-vcodec', 'mpeg4',
        '-avoid_negative_ts', 'make_zero',  # Avoid negative timestamps
        output_file                # The output file
    ]
    
    # Run the command with output suppressed
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        out, err = process.communicate()  # This will capture stdout and stderr, suppressing it
        # print(err.decode('utf-8'))
        return process.returncode  # Return the status code

def segment_from_transcript(text_dir, video_dir, output_dir):
    for file_name in tqdm(sorted(os.listdir(text_dir + '/'))):
        file1 = open(text_dir + "/" + file_name, 'r')
        Lines = file1.readlines()
        file_id = file_name.split(".")[0]
        video_name = video_dir + "/" + file_id + ".mp4"

        for l in Lines:
            split_lines = l.split("___")
            seg_num = split_lines[1]
            start = float(split_lines[2])
            end = float(split_lines[3])
            video_segname = output_dir + "/" + file_id + "_" + str(seg_num) + ".mp4"
            status_code = segment_video(video_name, start, end, video_segname)
            if status_code ==1:
                print(file_id + "_" + str(seg_num))
        print(file_id)



if __name__ == '__main__':
    text_dir = "/project/msoleyma_1026/CMU-MOSEI/MOSEI/Transcript/Segmented/Combined"
    video_dir = '/project/msoleyma_1026/MOSEI_cropped_face'
    output_dir = '/project/msoleyma_1026/mosei_edit/video_seg'
    segment_from_transcript(text_dir, video_dir, output_dir)