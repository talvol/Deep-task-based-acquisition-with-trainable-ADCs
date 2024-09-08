# MNIST dataset instructions

Make sure you also have the attached "MNIST Dataset.zip" file in order to test or train this system.  

#### <u>**Global parameters:**</u>

trainable_analog = False  # Dictates if the analog layer should be trained as well or not  
trainable_adc = True  # Dictates if we want a neuromorphic ADC or a uniform ADC  
q_bits = 4  # Number of bits  
number_of_epochs = 3  # Number of epochs  
p = 28     # Number of ADCs  
Vdd = 1  # Full scale voltage  
Vr = Vdd / (2 ** q_bits)  # Writing voltage  
Vref = Vr  # Reference voltage  
Rf = 45e3  # Reference resistor  
gamma = 0  # Weight regularization  
beta = 0  # Power accuracy trade-off   
learning_rates = [0.001]  # Learning rate  
batch_sizes = [32]  # Batch size  
train_size = 50000  # Number of images for the train dataset
valid_size = 10000  # Number of images for the validation dataset
test_size = 10000  # Number of images for the test dataset

load_trained_model = False  # Dictates if to train for a new model or load an exisiting model  
In order to test your own set of parameters you can choose any desired value for the hyperparameters and set "load_trained_model" to "False"  
In order to load an existing model you need to set "load_trained_model" to "True"  


#### <u>**Existing model structure explanation:**</u>  

Before testing an existing model please align the global parameters based on the naming structure of the trained model  

#### <u>**Example 1:**</u>  
"MNISTmodel_0_0_2_28_onlydig"  
"onlydig" means that trainable_analog & trainable_adc both set to False  
q_bits = 2  
p = 28  

#### <u>**Example 2:**</u>  
"MNISTmodel_0_1_4_28_neuroadc"  
"neuroadc" means that trainable_analog = False and trainable_adc = True   
q_bits = 4  
p = 28  