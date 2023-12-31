# Synthetic dataset instructions

#### <u>**Global parameters:**</u>

trainable_analog = True  # Dictates if the analog layer should be trained as well or not  
trainable_adc = True  # Dictates if we want a neuromorphic ADC or a uniform ADC  
q_bits = 4  # Number of bits  
number_of_epochs = 20  # Number of epochs  
p = 3     # Number of ADCs  
Vdd = 1.8  # Full scale voltage  
Vr = Vdd / (2 ** q_bits)  # Writing voltage  
Vref = Vr  # Reference voltage  
Rf = 45e3  # Reference resistor  
gamma = 0  # Weight regularization  
beta = 0  # Power accuracy trade-off  
torch.manual_seed(342)  # Seed for data creation  
np.random.seed(342)  # Seed for data creation  
learning_rates = [0.0005]  # Learning rate  
batch_sizes = [512, 1024]  # Batch size  
N = int(3.2e4)  # The size of the dataset  
n = 16  # Number of Antennas  
k = 5  # Number of transmitted bits  
f0 = 1e3  # Frequencies  
L = 20  
L_Sample = np.arange(1, L, 5)  # Sampling times  
number_of_samples = len(L_Sample)  

load_trained_model = False  # Dictates if to train for a new model or load an exisiting model  
In order to test your own set of parameters you can choose any desired value for the hyperparameters and set "load_trained_model" to "False"  
In order to load an existing model you need to set "load_trained_model" to "True"  


#### <u>**Existing model structure explanation:**</u>  

Before testing an existing model please align the global parameters based on the naming structure of the trained model  

#### <u>**Example 1:**</u>  
"Synthetic_model_onlydig_342_0_0_2_1"  
"onlydig" means that trainable_analog & trainable_adc both set to False  
seed = 342  
gamma = 0  
beta = 0  
q_bits = 2  
p = 1  

#### <u>**Example 2:**</u>  
"Synthetic_model_dig_and_analog_342_0_0_2_3"  
"dig_and_analog" means that trainable_analog = True and trainable_adc = False  
seed = 342  
gamma = 0  
beta = 0  
q_bits = 2  
p = 3  

#### <u>**Example 3:**</u>  
"Synthetic_model_342_150_300_4_3"  
trainable_analog = True  
trainable_adc = True  
seed = 342  
gamma = 150  
beta = 300  
q_bits = 4  
p = 3  
