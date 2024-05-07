# Emotion
## Environment
Python = 3.9 and make sure you have wandb logged in.
```
pip install lightning>=2.2.0 wandb scikit-learn yaml ffmpeg
```
## Run
```
torchrun --nproc_per_node=#gpu -m experiment.train --experiment=gatefusion --sweep_name=test
```
Make sure experiment is equal to cfgs.yaml files and ```#gpu``` is the gpu you will use. Sweep name is for wandb logging.

## Evaluate
```
python -m experiment.evaluate --model_path=/your/path/to/checkpoint/ \
                                --seed=0 \
                                --data=cremad \
                                --average=micro \
```
```--fine_tune``` is required when your base model is fine-tuned. ```--shared``` is required when you want to evaluate on shared encoder. ```--data``` can be ```cremad``` or ```mosei```.
