import os, ffmpeg, shutil
    


def selective_copy(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for idx, filename in enumerate(os.listdir(src_dir)):
        print(f'Processing file {idx+1}: {filename}')
        src_item = os.path.join(src_dir, filename)
        dst_item = os.path.join(dst_dir, filename)
        
        # Check if the current item should be skipped
        if filename in ['FrS-6P6FdCM.mp4', '-m9KtvCk_L8.mp4', 'vi0FsBCqfRE.mp4', '0OC3wf9x3zQ.mp4', '7l3BNtSE0xc.mp4',
                        'VN1IPN1TL3Q.mp4', 'BZOB3r5AoKE.mp4', 'xsiHAO0gq74.mp4', 'S6S2XL0-ZpM.mp4', 'k1ca_xbhohk.mp4',
                        'FdTczBnXtYU.mp4', 'WZVfvgTeFPo.mp4', 'PHZIx22aFhU.mp4', 'kf3fZcx8nIo.mp4', 'HuIKyKkEL0Q.mp4',
                        'Q2uHXKA8cq8.mp4', 'wO8fUOC4OSE.mp4', 'zsRTbbKlsEg.mp4', 'SaNXCez-9iQ.mp4']:
            print(f'Skipping file {idx+1}: {filename}')
            continue
        
        shutil.copy2(src_item, dst_item)


if __name__ == '__main__':
    source_directory = '/project/msoleyma_1026/CMU-MOSEI/MOSEI/Videos/Full/Combined/'
    destination_directory = '/project/msoleyma_1026/ms_videos/'

    selective_copy(source_directory, destination_directory)

    # path = '/project/msoleyma_1026/CMU-MOSEI/MOSEI/Videos/Full/Combined/'
    # count = 0 
    # for idx, file in enumerate(os.listdir(path)):
    #     if file in ['FrS-6P6FdCM.mp4', '-m9KtvCk_L8.mp4', 'vi0FsBCqfRE.mp4', '0OC3wf9x3zQ.mp4', '7l3BNtSE0xc.mp4',
    #                     'VN1IPN1TL3Q.mp4', 'BZOB3r5AoKE.mp4', 'xsiHAO0gq74.mp4', 'S6S2XL0-ZpM.mp4', 'k1ca_xbhohk.mp4',
    #                     'FdTczBnXtYU.mp4', 'WZVfvgTeFPo.mp4', 'PHZIx22aFhU.mp4', 'kf3fZcx8nIo.mp4', 'HuIKyKkEL0Q.mp4',
    #                     'Q2uHXKA8cq8.mp4', 'wO8fUOC4OSE.mp4', 'zsRTbbKlsEg.mp4', 'SaNXCez-9iQ.mp4']:
    #         print(f'Skipping file {idx+1}: {file}')
    #         continue
    #     fps = eval(ffmpeg.probe(f'{path}{file}')["streams"][0]["avg_frame_rate"])
    #     print(f'Processing file {idx+1}: {file} fps = {fps}')
        

            
            
