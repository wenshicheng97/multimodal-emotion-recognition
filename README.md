# Emotion
## Environment
Python = 3.9
```
pip install lightning>=2.2.0 wandb scikit-learn yaml ffmpeg
```
## Run
```
torchrun --nproc_per_node=#gpu -m experiment.train --experiment=gatefusion --sweep_name=test
```
Make sure experiment is equal to cfgs.yaml files and ```#gpu``` is the gpu you will use. Sweep name is for wandb logging.
