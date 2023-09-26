# Overcoming Label Noise for Source-free Unsupervised Video Domain Adaptation


This is the official code repository for "Overcoming Label Noise for Source-free Unsupervised Video Domain Adaptation".


## Requirements

To install dependencies, please use the following command -

```
conda env create -f environment.yml
```

## Training:

To reproduce the results reported in the paper please follow the steps given below - 

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