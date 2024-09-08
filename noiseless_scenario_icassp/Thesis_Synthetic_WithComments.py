"""
POWER-AWARE TASK-BASED LEARNING OF NEUROMORPHIC ADCS
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import copy

"""
Task: detect a binary vector s whose entries take values in a discrete set S = {-1, 1}
Create the input dataset 𝐱(𝚝)=𝐆(𝚝)⋅𝘀+𝐰(𝚝)
(𝐆(t))_{i,j} = (1+0.5*cos(2*pi*f*t))e^{-|i-j|}
"""

"Global parameters definition"

trainable_analog = True
trainable_adc = True
q_bits = 4  # Number of bits
number_of_epochs = 20
p = 3     # Number of ADCs
Vdd = 1.8  # Full scale voltage
Vr = Vdd / (2 ** q_bits)  # Writing voltage
Vref = Vr  # Reference voltage
Rf = 45e3  # Reference resistor
gamma = 0  # Weight regularization
beta = 0  # Power accuracy trade-off
torch.manual_seed(342)  # Seed for data creation
np.random.seed(342)  # Seed for data creation
learning_rates = [0.0005]
batch_sizes = [512, 1024]
weight_reg = []
N = int(3.2e4)  # The size of the dataset
n = 16  # Number of Antennas
k = 5  # Number of transmitted bits
f0 = 1e3  # Frequencies
L = 20
L_Sample = np.arange(1, L, 5)  # Sampling times
number_of_samples = len(L_Sample)
load_trained_model = False  # Dictates if to train for a new model or load an exisiting model


"""
createMatrixP ensures there's no overlap between levels in a quantization process.
Overlapping levels can cause a loss of resolution.
The function creates a matrix P and a vector b.
To ensure that we don't have overlap we will multiply P by weights W,
and make sure that the results are larger than vector b. This guarantees non-overlapping levels.
"""


def createMatrixP():
    # Create the list of decimal arrays
    list_of_arrays = [np.linspace(0, 2 ** q_bits - 1, 2 ** q_bits, dtype=int)]
    for i in reversed(range(q_bits - 1)):
        list_of_arrays.append(np.linspace(2 ** (i + 1) - 1, 0, 2 ** (i + 1), dtype=int))

    # Convert decimal arrays into arrays of bits
    for i in reversed(range(q_bits)):
        list_of_arrays[q_bits - 1 - i] = (
                (list_of_arrays[q_bits - 1 - i].reshape(-1, 1) & (2 ** np.arange(i, -1, -1))) != 0).astype(int)
    for i in range(q_bits):
        list_of_arrays[i] = np.where(list_of_arrays[i][:] == 0, -1, list_of_arrays[i][:])

    # Create zero padding vectors
    P = list_of_arrays[0]
    for b in range(1, q_bits):
        padding = np.zeros([2 ** b - 1, q_bits - b])
        np_arr = list_of_arrays[b]
        # Insert zeros between the bits
        for i in range(1, 2 ** q_bits - (2 ** b - 1), 2 ** b):
            np_arr = np.insert(np_arr, i, padding, axis=0)
        pad_beginning = np.zeros((2 ** (b - 1), q_bits - b))
        pad_ending = np.zeros((2 ** (b - 1) - 1, q_bits - b))
        np_arr = np.concatenate((pad_beginning, np_arr), axis=0)
        np_arr = np.concatenate((np_arr, pad_ending), axis=0)
        P = np.concatenate((P, np_arr), axis=1)

    P = torch.from_numpy(P).float()

    # Create b
    b = torch.zeros(2 ** q_bits, 1)
    b[0] = -2 ** q_bits

    return P, b


P, b = createMatrixP()


"""
quantizerOut takes as input an analog input y and the weights for the SAR ADC and outputs the quantized vector q
"""


def quantizerOut(y, W_tensor):
    num_of_adcs = 2 * p
    delta = 1e-30
    length = len(y)
    Q = torch.zeros([length, num_of_adcs, q_bits])
    q = torch.zeros([length, num_of_adcs])
    m = W_tensor.size(0) - 1
    for j in reversed(range(q_bits)):
        bit_sum = W_tensor[m] * Vref
        m = m - 1
        for k in range(j+1, q_bits):
            bit_sum = bit_sum + (Q[:, :, k] + 1) / 2 * W_tensor[m] * Vr
            m = m - 1
        Q[:, :, j] = torch.sign(y - bit_sum + delta)
    for b in reversed(range(q_bits)):
        q = q + (Q[:, :, b] + 1) / 2 * Vr * (2 ** b)
    return q.type(torch.FloatTensor)


"""Calculate the L2 Norm:"""


def L2norm(x):
    return np.sqrt(torch.sum(torch.pow(x, 2)))


"""Standardization"""


def standardization(train, valid, test):
    train_mean = train.mean()
    train_std = train.std()
    train = (train - train_mean) / (train_std * 3)
    valid = (valid - train_mean) / (train_std * 3)
    test = (test - train_mean) / (train_std * 3)
    return train, valid, test


"""
Analog Layer:
This function performs analog signal processing on a complex exponential signal x using a matrix A. 
The matrix A is used to perform a linear transformation on the input signal x to generate an output signal y.
Input:
x - data acquired from the sensors
p - number of ADCs
num_of_sensors - number of sensors in the sensor array
num_blocks - number of time samples
Output:
y - analog signal
"""


class AnalogLayer(nn.Module):
    def __init__(self, num_of_adcs=p, num_of_sensors=16, num_blocks=number_of_samples):
        super(AnalogLayer, self).__init__()
        self.length = int(num_of_sensors)  # Number of sensors
        self.num_of_adcs = num_of_adcs
        self.num_blocks = num_blocks
        self.p_values = torch.arange(self.num_of_adcs, dtype=torch.float).view(-1, 1)
        self.m_values = torch.arange(self.length)
        if trainable_analog:
            self.p_values = torch.nn.Parameter(self.p_values, requires_grad=True)

    def forward(self, x):
        phi = self.p_values * self.m_values
        A_complex = torch.exp(-phi * 2j * math.pi / self.length)
        A_real = torch.real(A_complex)
        A_imag = torch.imag(A_complex)
        A1 = torch.cat((A_real, A_imag), dim=0)
        A2 = torch.cat((-A_imag, A_real), dim=0)
        A = torch.cat((A1, A2), dim=1).float()
        A = torch.block_diag(*([A] * self.num_blocks))
        y = torch.matmul(A, x.t()).t()
        return y


"""
Quantization Layer:
This function takes as input a tensor y and the number of bits and returns a quantized version of y the 
static power of the quantization process.
The quantization is implemented using a successive approximation. 
The main loop iterates over the bits of the quantized output, 
starting with the most significant bit. At each iteration, 
the algorithm computes an estimate of the signal that takes into account the bits that have already been quantized. 
This estimate is subtracted from the input signal, 
and the sign of the result is stored in the corresponding bit of the quantized output. 
The function then computes the quantized value of the signal by summing up the contributions of all the bits.
Input:
y - analog signal
bits - number of bits
p - number of ADCs
Output:
q - quantized signal
Q - quantized signal in bits format
"""


class QuantizationLayer(nn.Module):
    def __init__(self, num_of_bits, num_of_adcs=p):
        super(QuantizationLayer, self).__init__()
        self.num_of_adcs = 2 * num_of_adcs * number_of_samples
        self.num_of_bits = num_of_bits
        W_init = torch.zeros([self.num_of_bits, self.num_of_bits])
        for i in range(self.num_of_bits):
            W_init[i, 0] = 2 ** i
            for j in range(self.num_of_bits):
                if j > i:
                    W_init[i, j] = 2 ** j
        W_tensor = matrix2vector(W_init)
        if trainable_adc:
            self.W_tensor = torch.nn.Parameter(W_tensor, requires_grad=True)
        else:
            self.W_tensor = W_tensor
        self.amplitude = 10000

    def forward(self, x):
        length = int(x.size(0))
        delta = 1e-30
        Q = torch.zeros([length, self.num_of_adcs, self.num_of_bits])
        q = torch.zeros([length, self.num_of_adcs])
        m = self.W_tensor.size(0)-1
        for j in reversed(range(self.num_of_bits)):
            bit_sum = self.W_tensor[m] * Vref
            m = m - 1
            for k in range(j + 1, self.num_of_bits):
                bit_sum = bit_sum + (Q[:, :, k] + 1) / 2 * self.W_tensor[m] * Vr
                m = m - 1
            Q[:, :, j] = torch.tanh(self.amplitude * (x - bit_sum + delta))
        for b in reversed(range(self.num_of_bits)):
            q = q + (Q[:, :, b] + 1) / 2 * Vr * (2 ** b)
        return q.type(torch.FloatTensor), Q.type(torch.FloatTensor)


"""
True Quantization Layer:
Act similar as the original quantization function but instead of using the "soft to hard" approach it uses a real
sign function in the process of building the digital output.
"""


class TrueQuantizationLayer(nn.Module):
    def __init__(self, W_tensor, num_of_bits, num_of_adcs=p):
        super(TrueQuantizationLayer, self).__init__()
        self.num_of_adcs = 2 * num_of_adcs * number_of_samples
        self.num_of_bits = num_of_bits
        if trainable_adc:
            self.W_tensor = W_tensor.detach()
        else:
            self.W_tensor = W_tensor

    def forward(self, x):
        length = int(x.size(0))
        delta = 1e-30
        Q = torch.zeros([length, self.num_of_adcs, self.num_of_bits])
        q = torch.zeros([length, self.num_of_adcs])
        m = self.W_tensor.size(0)-1
        for j in reversed(range(self.num_of_bits)):
            bit_sum = self.W_tensor[m] * Vref
            m = m - 1
            for k in range(j + 1, self.num_of_bits):
                bit_sum = bit_sum + (Q[:, :, k] + 1) / 2 * self.W_tensor[m] * Vr
                m = m - 1
            Q[:, :, j] = torch.sign(x - bit_sum + delta)
        for b in reversed(range(self.num_of_bits)):
            q = q + (Q[:, :, b] + 1) / 2 * Vr * (2 ** b)
        return q.type(torch.FloatTensor), Q.type(torch.FloatTensor)


"""
Digital Processing:
The input to the network expects a tensor shape according to the number of ADCs and time samples.
The output is later used for classification tasks when combined with a cross-entropy loss function.
"""


class digitalDNN(nn.Module):
    def __init__(self, num_of_adcs):
        super(digitalDNN, self).__init__()
        self.L1 = torch.nn.Linear(2 * num_of_adcs * number_of_samples, 64, bias=True).float()
        self.L2 = torch.nn.Linear(64, 32, bias=True).float()

    def forward(self, x):
        y = F.relu(self.L1(x))
        y = self.L2(y)
        return y


"""
The AGC implements an automatic gain control module.
It first normalizes the input signal then scales and shifts it within a voltage range defined by Vdd.
"""


class AGC(nn.Module):
    def __init__(self, num_of_adcs, Vdd=Vdd):
        super(AGC, self).__init__()
        self.Vdd = Vdd
        self.batch_norm = nn.BatchNorm1d(num_of_adcs * 2)

    def forward(self, x):
        x = self.batch_norm(x)
        x = (x + 1) * (Vdd / 2)
        return x


"""
Full Layer:
The full layer includes all the modules described above and will be used to train them all together
"""


class FullNet(nn.Module):
    def __init__(self, num_of_bits, num_of_adcs=p):
        super(FullNet, self).__init__()
        self.analog_layer = AnalogLayer(num_of_adcs)
        self.AGC = AGC(num_of_adcs * number_of_samples, Vdd)
        self.quantization_layer = QuantizationLayer(num_of_bits, num_of_adcs)
        self.digital_network = digitalDNN(num_of_adcs)
        W = torch.zeros([q_bits, q_bits])
        for i in range(q_bits):
            W[i, 0] = 2 ** i
            for j in range(q_bits):
                if j > i:
                    W[i, j] = 2 ** j
        W_tensor = matrix2vector(W)
        self.true_quantization_layer = TrueQuantizationLayer(W_tensor, num_of_bits, num_of_adcs)

    def forward(self, x):
        x = self.analog_layer(x)
        x = self.AGC(x)
        Vin = x
        if not trainable_adc and not trainable_analog:
            x, Q = self.true_quantization_layer(x)
        else:
            x, Q = self.quantization_layer(x)
        x = self.digital_network(x)
        return x, Q, Vin


"""
Power:
The function iterates over the bits of the quantized weights, calculating the power consumption at each bit. 
The power is affected by three main contributors, the input voltage, the reference voltage and the quantized weights.
The output of the function is the total power consumption of the neural network.
Input:
Vin - Input voltages
W - The relative weight of each input bit to the output bit
Q - The quantized input in bit representation
bits - The number of bits
Vref - The reference voltage
Rf - The resistance value of the feedback resistor used in the ADC circuit
Output: 
power_sum - The total power consumption of the network
"""


def synpPower(Vin, W_tensor, Q):
    p_syn = torch.zeros([Q.size(0), Q.size(1), q_bits])
    m = W_tensor.size(0) - 1
    Y_read = (Q + 1) / 2
    Vk = Vr * Y_read
    for i in reversed(range(q_bits)):
        bit_sum = W_tensor[m] * (Vref ** 2) / Rf
        m = m - 1
        for k in range(i+1, q_bits):
            bit_sum = bit_sum + (Vk[:, :, k] ** 2) * W_tensor[m]/Rf
            m = m - 1
        p_syn[:, :, i] = (Vin ** 2) / Rf + bit_sum
    synp_power = p_syn.sum(dim=2)
    power_avg = synp_power.mean(dim=0).view(p*2, number_of_samples)  # Average the values for all inputs
    power_avg_time = power_avg.mean(dim=1)  # Average the values in each ADC at each timestamp
    power_sum = power_avg_time.sum()  # Sum the power of all the ADCs to a single total power
    return power_sum


def intPower(Vin, W_tensor, Q):
    p_int = torch.zeros([Q.size(0), Q.size(1), q_bits])
    m = W_tensor.size(0) - 1
    Y_read = (Q + 1) / 2
    Vk = Vr * Y_read
    for i in reversed(range(q_bits)):
        bit_sum = W_tensor[m] * Vref
        m = m - 1
        for k in range(i+1, q_bits):
            bit_sum = bit_sum + Vk[:, :, k] * W_tensor[m]
            m = m - 1
        p_int[:, :, i] = ((Vin - bit_sum) ** 2)/Rf
    int_power = p_int.sum(dim=2)
    power_avg = int_power.mean(dim=0).view(p*2, number_of_samples)  # Average the values for all inputs
    power_avg_time = power_avg.mean(dim=1)  # Average the values in each ADC at each timestamp
    power_sum = power_avg_time.sum()  # Sum the power of all the ADCs to a single total power
    return power_sum


def totalPower(Vin, W_tensor, Q):
    activationPower = 3e-6
    synapsePower = synpPower(Vin, W_tensor, Q)
    integrationPower = intPower(Vin, W_tensor, Q)
    total_power = synapsePower + integrationPower + activationPower
    # if not trainable_adc:
    #     print("The total power per ADC is equal to \u2248", total_power * 1_000_000 / 8, "\u00B5W")
    return total_power, synapsePower, integrationPower


"""
Plots the quantizer levels for the given input and quantized signals.
x: The input signal values
quan_x: The corresponding quantized signal values
"""


def plotQuant(x, quan_x):
    x_vec = x.detach().numpy().flatten()
    quan_x = quan_x.detach().numpy().flatten()
    plt.step(x_vec, quan_x)
    if trainable_adc:
        plt.title("Trained Quantizer Levels")
    else:
        plt.title("Uniform Quantizer Levels")
    plt.xlabel("x")
    plt.ylabel("quantized x")
    plt.grid()
    plt.show()
    return None


"""Convert Matrix to Vector:"""


def matrix2vector(W):
    W_tensor = torch.Tensor([W[i, j] for
                             i in range(q_bits) for j in reversed(range(q_bits)) if W[i, j] != 0]).view(-1, 1)
    return W_tensor


"""Create loss functions:"""


def createConstraints(W_tensor):
    Pw = torch.matmul(P, W_tensor)
    regularizations = Pw - b
    return regularizations


def exp_relu(x):
    return torch.exp(torch.relu(x)) - 1.0


"""
Creates the constraints for the weights regularisation loss:
"""


def constraint_loss(W_tensor, margin=0):
    violations = createConstraints(W_tensor)
    margin_tensor = torch.tensor(margin, dtype=torch.float)
    # Hinge loss for existing constraints
    hinge_losses = exp_relu(margin_tensor - violations)
    # Hinge loss for positive weight constraints
    non_positive_weights_loss = torch.relu(-W_tensor)
    # Combine hinge losses and return the sum
    total_hinge_loss = torch.sum(hinge_losses)
    total_non_positive_weights_loss = torch.sum(non_positive_weights_loss)
    loss = total_hinge_loss + total_non_positive_weights_loss
    return loss


"""
A function for loss calculation:
Loss = CrossEntropy + beta*TotalPower + gamma*LevelMergingError
CrossEntropy - is the regular cross-entropy loss used for classification
total power - is the total power consumed by the system that we want to reduce
level merging error - is the error caused by the changes of the weights that might cause to loss of resolution
"""


def calcLoss(output, labels, Vin, W_tensor, Q):
    CEloss = nn.CrossEntropyLoss()
    totpower, _, _ = totalPower(Vin, W_tensor, Q)
    level_merg_error = constraint_loss(W_tensor)
    weight_reg.append(level_merg_error.detach().numpy())
    loss = CEloss(output, labels) + beta * totpower + gamma * level_merg_error
    return loss


"""Decision Regions Plot:"""


def calcDecisionRegions(q_vec):
    q_vec = q_vec.detach().numpy().flatten()
    decision_regions = list(set(q_vec))
    num_decision_regions = len(decision_regions)
    return num_decision_regions


def plotDecisionRegions(q_vec):
    q_vec = q_vec.detach().numpy().flatten()
    q_vec_list = list(q_vec)
    decision_regions = list(set(q_vec))
    num_decision_regions = len(decision_regions)
    print(f"Number of decision regions: {num_decision_regions}")
    plt.bar(decision_regions, [q_vec_list.count(element) for element in decision_regions], width=1, align='center')
    if trainable_adc:
        plt.title("Trained Quantizer Decision Regions")
    else:
        plt.title("Uniform Quantizer Decision Regions")
    tick_locations = np.arange(min(decision_regions), max(decision_regions), 0.5)
    plt.xticks(tick_locations)
    plt.show()
    return None


"""Training:"""


def train(network, learning_rate, batch_size, train_loader, validation_loader, x_train, x_valid, s_train, s_valid):
    lr = learning_rate
    Nepochs = number_of_epochs
    # Epoch - one forward and backward pass of ALL training samples
    # Batch size - number of training samples used in one forward or backward pass
    # Nepochs - number of total passes, each pass (forward+backward) using batch size as number of samples
    optimizer = torch.optim.Adam(network.parameters(), lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    x = np.linspace(0, Vdd, 1000 * 2 * p, dtype=np.single)
    x = torch.tensor(x.reshape(1000, 2*p))
    train_acc, valid_acc = [], []
    train_err, valid_err = [], []
    power_epoch, totpower, synppower, intpower = [], [], [], []
    valid_loss_min = np.Inf
    for epoch in range(Nepochs):
        network.train()
        train_loss = 0.0
        valid_loss = 0.0
        total_power, synapse_power, integration_power = 0.0, 0.0, 0.0
        if epoch % 5 == 0:
            print("epoch = ", epoch)
            print("W_tensor = ", network.quantization_layer.W_tensor)
            print("Phase = ", network.analog_layer.p_values)
        for i, (inputs_t, labels_t) in enumerate(train_loader):
            # Clear the gradient
            optimizer.zero_grad()
            # Forward pass to compute outputs
            output_t, Q_t, Vin_t = network(inputs_t)
            # Batch loss
            if trainable_adc:
                loss = calcLoss(output_t, labels_t, Vin_t, network.quantization_layer.W_tensor, Q_t)
            else:
                loss = calcLoss(output_t, labels_t, Vin_t, network.true_quantization_layer.W_tensor, Q_t)
            # Compute gradients of the loss
            loss.backward()
            # Weight update and parameter optimization
            optimizer.step()
            # Calculate the batch loss - loss.item() is returing the loss of the entire mini-batch
            train_loss = train_loss + loss.item() * batch_size
        train_loss = train_loss / train_size
        train_err.append(train_loss)
        print('The training loss at epoch {} is {:.3f}'.format(epoch, train_loss))
        train_out, _, _ = network(torch.FloatTensor(x_train).T)
        score, predicted = torch.max(train_out, 1)
        acc = (s_train == predicted).float().sum() / len(s_train)
        train_acc.append(acc)
        # Model Validation
        validation_net = copy.deepcopy(network)
        validation_net.eval()
        with torch.no_grad():
            if trainable_adc:
                validation_net.quantization_layer = TrueQuantizationLayer(network.quantization_layer.W_tensor, q_bits)
            for i, (inputs_v, labels_v) in enumerate(validation_loader):
                # Forward pass to compute outputs
                output_v, Q_v, Vin_v = validation_net(inputs_v)
                # Batch loss
                if trainable_adc:
                    loss = calcLoss(output_v, labels_v, Vin_v, validation_net.quantization_layer.W_tensor, Q_v)
                else:
                    loss = calcLoss(output_v, labels_v, Vin_v, validation_net.true_quantization_layer.W_tensor, Q_v)
                # Calculate the validation loss
                valid_loss = valid_loss + loss.item() * batch_size
                # Calculate the validation power
                if trainable_adc:
                    totalP, synpP, intP = totalPower(Vin_v, validation_net.quantization_layer.W_tensor, Q_v)
                else:
                    totalP, synpP, intP = totalPower(Vin_v, validation_net.true_quantization_layer.W_tensor, Q_v)
                total_power += totalP.item() * batch_size
                synapse_power += synpP.item() * batch_size
                integration_power += intP.item() * batch_size
        # Calculate the validation loss of the current epoch
        valid_loss = valid_loss / valid_size
        valid_err.append(valid_loss)
        print('The validation loss at epoch {} is {:.3f}'.format(epoch, valid_loss))
        total_power = total_power / valid_size
        synapse_power = synapse_power / valid_size
        integration_power = integration_power / valid_size
        totpower.append(total_power * 1_000_000)
        synppower.append(synapse_power * 1_000_000)
        intpower.append(integration_power * 1_000_000)
        power_epoch.append(epoch)
        if epoch % 1 == 0 and trainable_adc:
            print(calcDecisionRegions(quantizerOut(x, validation_net.quantization_layer.W_tensor)))
        if epoch % 1 == 0 and not trainable_adc:
            print(calcDecisionRegions(quantizerOut(x, validation_net.true_quantization_layer.W_tensor)))
        valid_out, _, _ = validation_net(torch.FloatTensor(x_valid).T)
        score, predicted_v = torch.max(valid_out, 1)
        acc = (s_valid == predicted_v).float().sum() / len(s_valid)
        valid_acc.append(acc)
        # scheduler.step()
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            final_network = copy.deepcopy(network)
            valid_loss_min = valid_loss
        # final_network = copy.deepcopy(network)
    return train_err, train_acc, valid_err, valid_acc, final_network, power_epoch, totpower, synppower, intpower


"""Test"""


def test(network, batch_size, test_loader, x_test, s_test, q_bits):
    test_loss = 0.0
    x = np.linspace(0, Vdd, 1000 * 2 * p, dtype=np.single)
    x = torch.tensor(x.reshape(1000, 2*p))
    # Model Validation
    test_net = copy.deepcopy(network)
    test_net.eval()
    with torch.no_grad():
        if trainable_adc:
            test_net.quantization_layer = TrueQuantizationLayer(network.quantization_layer.W_tensor, q_bits)
        if trainable_adc:
            print(calcDecisionRegions(quantizerOut(x, test_net.quantization_layer.W_tensor)))
        if not trainable_adc:
            print(calcDecisionRegions(quantizerOut(x, test_net.true_quantization_layer.W_tensor)))
        for i, (inputs, labels) in enumerate(test_loader):
            # Forward pass to compute outputs
            output, Q, Vin = test_net(inputs)
            # Batch loss
            if trainable_adc:
                loss = calcLoss(output, labels, Vin, test_net.quantization_layer.W_tensor, Q)
            else:
                loss = calcLoss(output, labels, Vin, test_net.true_quantization_layer.W_tensor, Q)
            # Calculate the test loss
            test_loss = test_loss + loss.item() * batch_size
        # Calculate the validation loss of the current epoch
    test_loss = test_loss / test_size
    print('The test loss is {:.3f}'.format(test_loss))
    test_out, _, _ = test_net(torch.FloatTensor(x_test).T)
    score, predicted = torch.max(test_out, 1)
    acc = (s_test == predicted).float().sum() / len(s_test)
    print('The test accuracy is {:.3f}'.format(acc))
    # Calc power
    if trainable_adc:
        _, Q, Vin_t = test_net(torch.transpose(x_test, 0, 1))
        totalp, synpp, intp = totalPower(Vin_t, test_net.quantization_layer.W_tensor, Q)
    else:
        _, Q, Vin_t = test_net(torch.transpose(x_test, 0, 1))
        totalp, synpp, intp = totalPower(Vin_t, test_net.true_quantization_layer.W_tensor, Q)
    print('The total power for the test set is: ', totalp.detach() * 1_000_000, "\u00B5W")
    print('The synapse power for the test set is: ', synpp.detach() * 1_000_000, "\u00B5W")
    print('The integration power for the test set is: ', intp.detach() * 1_000_000, "\u00B5W")
    return None


"""Plot histograms"""


def plotHist(y_input, q_output):
    lent = int(y_input.size(1) * y_input.size(0))
    Yt = torch.tensor(np.zeros([lent]))
    Qt = torch.tensor(np.zeros([lent]))
    for i in range(2 * p):
        start = i * y_input.size(1)
        end = y_input.size(1) + y_input.size(1) * i
        Yt[start:end] = y_input[i, :]
        Qt[start:end] = q_output[i, :]
    plt.hist(Yt, bins=32, label="Input")
    plt.hist(Qt, bins=32, label="Quantized Output")
    plt.xlabel("Voltage value")
    plt.ylabel("Amount of values")
    plt.title("Train data")
    plt.legend(loc='best')
    plt.xlim(-1, 1)
    plt.show()
    return


"""Calculate Distortion:"""


def calcDist(Yt, Qt, Yv, Qv):
    dist_t = L2norm(Yt - Qt) / L2norm(Yt)
    print("Train data distortion: ", dist_t)
    dist_v = L2norm(Yv - Qv) / L2norm(Yv)
    print("Validation data distortion: ", dist_v)


"""Dataset creation & loading"""


class LoadData(Dataset):
    def __init__(self, data):
        super(LoadData, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def loadData(data, symbol, size, batch_size):
    data_set = []
    for i in range(size):
        data_set.append([data[:, i], symbol[i]])
    data_set = LoadData(data_set)
    dataset = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return dataset


"""Plot loss & accuracy graphs:"""


def plotGraphs(title, results):
    # Extract data from results
    train_err = np.array([res['train_err'] for res in results])
    valid_err = np.array([res['valid_err'] for res in results])
    train_acc = np.array([res['train_acc'] for res in results])
    valid_acc = np.array([res['valid_acc'] for res in results])
    learning_rates = [res['learning_rate'] for res in results]
    batch_sizes = [res['batch_size'] for res in results]

    # Plot loss
    Epochs = range(0, number_of_epochs)
    for i in range(len(results)):
        print("train loss: {:.3f} \u03BC = {} batch size = {}".format(train_err[i, number_of_epochs - 1],
                                                                      learning_rates[i], batch_sizes[i]))
        print("validation loss: {:.3f} \u03BC = {} batch size = {}".format(valid_err[i, number_of_epochs - 1],
                                                                           learning_rates[i], batch_sizes[i]))
        plt.plot(Epochs, train_err[i, :],
                 label="Train Loss, \u03BC = " + str(learning_rates[i]) + " batch size = " + str(batch_sizes[i]))
        plt.plot(Epochs, valid_err[i, :],
                 label="Valid Loss, \u03BC = " + str(learning_rates[i]) + " batch size = " + str(batch_sizes[i]))

    plot_title = title + " Loss Curves"
    plt.title(plot_title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    # Plot accuracy
    for i in range(len(results)):
        print("train accuracy: {:.3f} \u03BC = {} batch size = {}".format(train_acc[i, number_of_epochs - 1],
                                                                         learning_rates[i], batch_sizes[i]))
        print("validation accuracy: {:.3f} \u03BC = {} batch size = {}".format(valid_acc[i, number_of_epochs - 1],
                                                                               learning_rates[i], batch_sizes[i]))
        plt.plot(Epochs, train_acc[i, :],
                 label="Train Acc, \u03BC = " + str(learning_rates[i]) + " batch size = " + str(batch_sizes[i]))
        plt.plot(Epochs, valid_acc[i, :],
                 label="Valid Acc, \u03BC = " + str(learning_rates[i]) + " batch size = " + str(batch_sizes[i]))

    plot_title = title + " Accuracy Curves"
    plt.title(plot_title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()


def plotTest(title, Number_of_epochs, learning_rates, batch_sizes, test_err, test_acc):
    # Plot loss
    Epochs = range(0, Number_of_epochs)
    print("test loss: {:.3f} \u03BC = {} batch size = {}".format(test_err[Number_of_epochs - 1], learning_rates[0],
                                                                 batch_sizes[0]))
    plt.plot(Epochs, test_err,
             label="Test Loss, \u03BC = " + str(learning_rates[0]) + " batch size = " + str(batch_sizes[0]))
    plot_title = title + " Loss Curves"
    plt.title(plot_title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    # Plot accuracy
    print("test accuracy: ", test_acc[Number_of_epochs - 1], " \u03BC = ", learning_rates[0],
          " batch size = ", batch_sizes[0])
    plt.plot(Epochs, test_acc,
             label="Test Acc, \u03BC = " + str(learning_rates[0]) + " batch size = " + str(batch_sizes[0]))
    plot_title = title + " Accuracy Curves"
    plt.title(plot_title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()


"""Plot Power"""


def plotPower(power_epoch, power, title):
    plt.plot(power_epoch, power)
    plot_title = title
    plt.title(plot_title)
    plt.xlabel("Epoch")
    plt.ylabel("Power [\u00B5W]")
    plt.show()


"""Entropy:"""


def calcEntropy(x):
    x = x.detach().numpy().flatten() + 10
    values, counts = np.unique(x, return_counts=True)
    pi = counts / len(x)
    entropy = - np.sum(pi * np.log2(pi))
    max_entropy = np.log2(len(counts))
    print("Entropy:", entropy, "Max Entropy:", max_entropy)
    return None


"""Run experiments"""


def run_experiment(batch_size, learning_rate, x_train, x_valid, s_train, s_valid, train_size, valid_size):
    train_loader = loadData(x_train, s_train, train_size, batch_size)
    validation_loader = loadData(x_valid, s_valid, valid_size, batch_size)
    network = FullNet(q_bits)
    train_err, train_acc, valid_err, valid_acc, final_net, power_epoch, total_power, synp_power, int_power = \
        train(network, learning_rate, batch_size, train_loader, validation_loader, x_train, x_valid, s_train,
              s_valid)
    return train_err, train_acc, valid_err, valid_acc, final_net, power_epoch, total_power, synp_power, int_power


"""Main part"""
w = torch.randn(n, L, N)  # Gaussian noise creation
w_sample = w[:, ::5, :]
w_sample = w_sample.view(n * len(L_Sample), N)
SNR = 1

# Create the task vector
symbol_set = [-1, 1]
s = torch.tensor(np.random.choice(symbol_set, [k, N]))
# Convert the binary number to a decimal number
dec_s = torch.tensor(np.zeros(N))
for i in range(k):
    dec_s[:] = dec_s[:] + (s[i, :] + 1) / 2 * pow(2, i)
dec_s = dec_s.to(torch.long)


def compute_G_t(t):
    """Compute matrix G(t) for a given time t."""
    G_t = torch.tensor(np.zeros((n, k), dtype="complex_"))
    for i in range(n):
        for j in range(k):
            G_t[i, j] = np.sqrt(SNR)*(1 + 0.5 * np.cos(2 * np.pi * f0 * t)) * np.exp(-abs(i - j))
    return G_t


# Compute the matrix for each time sample
G_list = [compute_G_t(t) for t in L_Sample]
G = torch.stack(G_list, dim=1)
G = G.view(n * len(L_Sample), k)

# Calculate x
x = np.matmul(G, s) + w_sample
x = torch.tensor(np.concatenate((torch.real(x), torch.imag(x)), axis=0)).float()


"""Separate the dataset into train and validation and normalize according to the train"""

train_prec = 0.7
valid_prec = 0.15
train_size = int(N * train_prec)
valid_size = int(N * valid_prec)
test_size = N - train_size - valid_size
x_train = x[:, :train_size]
x_valid = x[:, train_size:(valid_size + train_size)]
x_test = x[:, valid_size + train_size:]
s_train = dec_s[:train_size]
s_valid = dec_s[train_size:(valid_size + train_size)]
s_test = dec_s[valid_size + train_size:]

"""Standardization and Scaling"""
x_train, x_valid, x_test = standardization(x_train, x_valid, x_test)
# plotDecisionRegions(x_train)

trained_model = FullNet(q_bits)

results = []
test_power = []

if not load_trained_model:
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            train_err, train_acc, valid_err, valid_acc, final_net, power_epoch, total_power, synp_power, int_power = \
                run_experiment(batch_size, learning_rate, x_train, x_valid, s_train, s_valid, train_size, valid_size)
            results.append({
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "train_err": train_err,
                "train_acc": train_acc,
                "valid_err": valid_err,
                "valid_acc": valid_acc
            })
    plotGraphs("Training & Validation", results)

    # Print Power vs Epoch
    if trainable_adc:
        plotPower(power_epoch, total_power, "Total power VS # of Epochs")
        plotPower(power_epoch, synp_power, "Synapse power VS # of Epochs")
        plotPower(power_epoch, int_power, "Integration power VS # of Epochs")


# Test
test_loader = loadData(x_test, s_test, test_size, batch_sizes[0])
"""If load_trained_model = TRUE, we must change "filename" to an existing model before testing"""
if load_trained_model:
    trained_model.load_state_dict(torch.load('filename'))
    test(trained_model, batch_sizes[0], test_loader, x_test, s_test, q_bits)
else:
    test(final_net, batch_sizes[0], test_loader, x_test, s_test, q_bits)

plt.plot(weight_reg)
plt.show()

gamma_int = int(gamma)
beta_int = int(beta)
q_bits_int = int(q_bits)
p_int = int(p)
# Format a string to use as the filename.
# filename = f"Synthetic_model_342_{gamma_int}_{beta_int}_{q_bits_int}_{p_int}.pth"
#
# torch.save(final_net.state_dict(), filename)
