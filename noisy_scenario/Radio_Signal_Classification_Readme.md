## RF Signal Experimental Study

The RF signal experimental study focuses on the classification of communications RF signals, covering a variety of modulation types. We utilize the dataset provided at [Panoradio SDR - Radio Signal Classification Dataset](https://panoradio-sdr.de/radio-signal-classification-dataset/), which contains signals representing 18 different transmission modes commonly found in the HF band.

The code in this repository includes five Python files:

### 1. **Hyperparameters**
This file contains all the hyperparameters available for testing:
- `is_ADC`: Controls whether an ADC is included in the joint network test.
- `trainable_adc`: Controls whether the ADC is trainable or uniform.
- `noisy_inference`: Enables noise during inference.
- `noisy_training`: Enables noise during training.
- `distillation_training`: Enables the use of a teacher model for distillation. To use this function, specify the teacher model's location in `teacher_model = ""` and set appropriate `alpha` and `T` parameters.
- `q_bits`: Dictates the number of bits in the ADC.
- `p`: Sets the packet size.
- `noise_sigma`: Sets the noise sigma (higher sigma introduces stronger noise).
- `beta`: Hyperparameter for balancing the tradeoff between power and accuracy.
- `gamma`: Hyperparameter for weight regularization.
- `alpha`: Hyperparameter for the cross-entropy between the student and teacher models. Must be `0` if distillation is not used; otherwise, it must be greater than `0`.
- `T`: Teacher model temperature parameter. Must be `0` if distillation is not used; otherwise, it must be greater than `0`.
- `Vdd`: Sets the full-scale voltage.
- `Rf`: Sets the value of the reference resistor in the SAR ADC.
- `number_of_epochs`: Sets the number of training epochs.
- `batch_sizes`: Sets the batch size.
- `learning_rates`: Sets the learning rate.

### 2. **dataset_preparation**
This Python file includes all necessary functions to load the HF data and transform it into channel spectrograms.

### 3. **deep_learning_models_torch**
This Python file contains the feature extractor neural network and related ResNet blocks.

### 4. **functions**
This file includes all functions used during the experiment, such as training, testing, and result printing. To test the RF scenario, the user must specify the paths for the data and labels in the `load_data` function:
```python
load_data(file_path_data=r"PLACE HERE DATA PATH FOR HF RADIO DATASET .NPY FILE",
              file_path_labels=r"PLACE HERE DATA PATH FOR HF RADIO LABELS .CSV FILE")
```

### 5. Thesis**
This file contains the main function for running the experiment.
