# Source-free Video Domain Adaptation by Learning from Noisy Labels

This is the official code repository for "Source-free Video Domain Adaptation by
Learning from Noisy Labels", Arxiv'22. An initial version of this work is published at ICVGIP'22.


## Requirements

To install dependencies, please use the following command -

```
conda env create -f environment.yml
```

## Training:

To reproduce the results reported in the paper, please follow the steps given below - 

### Step 1: Prepare the dataset
```
data
├── flow
├── rgb
|   ├── ucf101
|   |   ├──  v_YoYo_g25_c05
|   |   ├──  ...
|   ├── hmdb51
```
### Step 2: Source-only Pre-training
You may need to adjust the <code>data</code> path in the script

```
bash scripts/source_only_train.sh ucf101 hmdb51 Joint
```

### Step 2: Source-only Pre-training

```
bash scripts/generate_pseudo_labels.sh ucf101 hmdb51 Joint 12
```


### Step 2: Adaptation Training
To run the CleanAdapt, assuming <code>\tau = 0.5</code> - 

```
bash scripts/adaptation_uh.sh ucf101 hmdb51 Joint 0.5
```


To run the CleanAdapt + TS, assuming <code>\tau = 0.5</code> - 

```
bash scripts/adaptation_uh_ema.sh ucf101 hmdb51 Joint 0.5
```

Please check the <code>parse_args.py</code> for more details on the argumments. 

## Citation:
Please consider citing the following work if you make use of this repository:
```
@inproceedings{dasgupta2024source,
  title={Source-free Video Domain Adaptation by Learning from Noisy Labels},
  author={Dasgupta, Avijit and Jawahar, CV and Alahari, Karteek},
  booktitle={Arxiv},
  year={2024}

@inproceedings{dasgupta2022overcoming,
  title={Overcoming Label Noise for Source-free Unsupervised Video Domain Adaptation},
  author={Dasgupta, Avijit and Jawahar, CV and Alahari, Karteek},
  booktitle={ICVGIP},
  year={2022}
}
```

## Contact

In case of any issues, feel free to create a pull request. Or reach out to [Avijit Dasgupta](https://avijit9.github.io).
