## MNIST Experimental Study

In this experiment, we evaluate the proposed memristive task-based acquisition system for a scenario based on imaging applications. Specifically, we focus on handwritten digit recognition using the MNIST dataset.

The code in this repository includes one main Python file:
- `Thesis_MNIST.py`

### Hyperparameters:
To run the code, the user must set the following hyperparameters:

- `trainable_analog`: Indicates if the system will have trainable analog or fixed analog processing.
- `trainable_adc`: Indicates if the system will use a uniform ADC or the proposed trainable ADC.
- `noisy_inference`: Enables noise during inference.
- `noisy_training`: Enables noise during training.
- `distillation_training`: Enables distillation. Requires setting appropriate `alpha` and `T` values (>0) and specifying the teacher model filename.
- `noise_sigma`: Indicates the strength of the noise.
- `q_bits`: Sets the number of bits in the ADC.
- `p`: Sets the number of ADCs (the actual number of ADCs is `p * 2`).
- `gamma`: Weight regularization hyperparameter.
- `beta`: Power-accuracy trade-off hyperparameter.
- `alpha`: Hyperparameter for the cross-entropy between the teacher and student. Must be `0` if distillation is false, and greater than `0` if distillation is true.
- `T`: Distillation temperature. Must be `0` if distillation is false, and greater than `0` if distillation is true.
- `number_of_epochs`: Dictates the number of training epochs.
- `learning_rates`: Dictates the learning rate.
- `batch_sizes`: Dictates the batch size.
- `filename`: If distillation is true, this is used to load the teacher network Python file.
