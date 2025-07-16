# MPFormer - Multi-scale Physics-informed Transformer for Extreme Precipitation Nowcasting

This repository contains the code for MPFormer, a novel model designed for extreme precipitation forecasting, as presented in the paper:
**"Multi-scale Physics-informed Transformer With Spatio-temporal Feature Adapter For Extreme Precipitation Nowcasting"**.

## Features:

* Multi-scale transformer architecture
* Physics-informed feature adaptation
* Spatio-temporal feature fusion for extreme precipitation prediction
* Efficient training and testing scripts for rapid deployment

## Requirements

The following environment setup is required to run the model:

* Python >= 3.6
* GPU (CUDA-capable GPU recommended for training)
* Install dependencies using pip:

  ```bash
  pip install -r requirements.txt
  ```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ziyi0227/MPFormer.git
   cd MPFormer
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, run the following script:

```bash
python code/run.py
```

Make sure to set your training data and configuration in the respective file or modify the script as needed.

### Testing

To test the model on new data, run the following script:

```bash
python code/test.py
```

Ensure that you have your test data and configuration set up as per your use case.

## Dataset



## Citation

If you use this repository in your research, please cite the following paper:


## License

