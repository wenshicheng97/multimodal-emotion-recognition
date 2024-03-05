for feature in 'gaze_landmarks' 'face_landmarks' 'audio'
    do 
        srun --gres=gpu:a6000:4 --time 3000 python -m experiment.train_autoencoder --feature $feature
    done
    