#### Installation instructions
Creat a new conda environment
```
conda create -n test python=3.10
```
Install packages (adjust to cuda version you require)

```
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torchgeo torchmetrics tabulate tqdm timm pyyaml tensorboard termcolor
```

#### Running the code
To replicate all experiments given in the paper we must first change the root directories for each dataset in config.py
```
_C.DATASET.EUROSAT_ROOT = "your-path"
_C.DATASET.BIGEARTHNET_ROOT = "your-path"
```

Then we can use the python script ./scripts/complete_runs.py. First run the BigEarthNet experiments:
```
python ./scripts/complete_runs.py --dataset bigearthnet --arch resnet50
python ./scripts/complete_runs.py --dataset bigearthnet --arch densenet121
python ./scripts/complete_runs.py --dataset bigearthnet --arch vit --lr 0.0001
```

Then we can peform the Eurosat experiments. Change the paths in ./scripts/complete_runs.py, to the best performing sentinel-2 weights for transfer learning from the BigEarthNet dataset:
BIGEARTH_WEIGHTS = {
    "resnet50": "your-path",
    "densenet121": "your-path",
    "vit": "your-path",
}
```
python ./scripts/complete_runs.py --arch resnet50
python ./scripts/complete_runs.py --arch densenet121
python ./scripts/complete_runs.py --arch vit --lr 0.0001
```

Finally to summarise the results
```
python csvgen.py runs/bigearthnet 
```

#### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
