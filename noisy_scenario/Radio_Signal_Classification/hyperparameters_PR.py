"""In this file we determine the hyperparameters"""
import torch

seed = 555
is_ADC = True
trainable_adc = True
noisy_inference = True
noisy_training = False
distillation_training = False
teacher_model = ".pth"
q_bits = 4  # Number of bits
p = 2048  # Packet size
noise_sigma = 0.03
beta = 1  # Power accuracy trade-off
gamma = 1  # Weight regularization
alpha = 0  # Distillation parameter
T = 0  # Teacher model temperature parameter
Vdd = 1  # Full scale voltage
Rf = 45e3  # Reference resistor
number_of_epochs = 600
batch_sizes = [32]
learning_rates = [0.001]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
