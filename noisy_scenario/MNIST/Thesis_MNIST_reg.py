"""
POWER-AWARE TASK-BASED LEARNING OF NEUROMORPHIC ADCS
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import copy
import struct
from array import array
from os.path import join
from sklearn.model_selection import train_test_split

"""
Task: MNIST regression task
"""

"Global parameters definition"
load_trained_model = False  # Dictates if to train for a new model or load an existing model
load_file_name = f""
save_file = True
trainable_analog = True
trainable_adc = True
noisy_inference = True
noisy_training = True
distillation_training = True
q_bits = 4  # Number of bits
p = 7  # Number of ADCs
noise_sigma = 0.3
beta = 150  # Power accuracy trade-off
gamma = 1  # Weight regularization
alpha = 0.05  # Distillation parameter
T = 2  # Teacher model temperature parameter
eta = 0  # L2 parameter

Vdd = 1  # Full scale voltage
Vr = Vdd / (2 ** q_bits)  # Writing voltage
Vref = Vr  # Reference voltage
Rf = 45e3  # Reference resistor
seed = 10
torch.manual_seed(seed)  # Seed for data creation
np.random.seed(seed)
number_of_epochs = 120
batch_sizes = [32]
learning_rates = [0.001]
train_size = 50000
valid_size = 10000
test_size = 10000
if distillation_training:
    # filename = f"Teacher_MNIST_{gamma}_{beta}_{q_bits}_{p}.pth"
    teacher_filename = f"Teacher_MNIST_0_1_1_0_4_7.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(seed)

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
    P = P.to(device)
    b = b.to(device)
    return P, b


P, b = createMatrixP()


def complex_matrix_norm(matrix):
    # Calculate the element-wise magnitude of the complex matrix
    magnitude = torch.abs(matrix)
    # Calculate the norm of each entry
    entry_norm = torch.norm(magnitude, dim=1)
    return torch.mean(entry_norm)


"""
quantizerOut takes as input an analog input y and the weights for the SAR ADC and outputs the quantized vector q
"""


def quantizerOut(y, W_tensor):
    nadcs = p
    delta = 1e-30
    length = len(y)
    Q = torch.zeros([length, nadcs, q_bits],device=device)
    q = torch.zeros([length, nadcs],device=device)
    m = W_tensor.size(0) - 1
    for j in reversed(range(q_bits)):
        bit_sum = W_tensor[m] * Vref
        m = m - 1
        for k in range(j + 1, q_bits):
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
    train = (train - train_mean) / train_std
    valid = (valid - train_mean) / train_std
    test = (test - train_mean) / train_std
    return train, valid, test


"""Convert Matrix to Vector:"""


def matrix2vector(W):
    W_tensor = torch.Tensor([W[i, j] for
                             i in range(q_bits) for j in reversed(range(q_bits)) if W[i, j] != 0]).view(-1, 1)
    return W_tensor


"""
Analog Layer:
The AnalogLayer class processes MNIST images represented as long vectors through a DCT.
The parameter phi can either be trainable or fixed, and a weight matrix, A, derived from the input length.
During the forward pass, the image is multiplied with a phase-shifted cosine matrix and the weight matrix,
producing the analog output y.
"""


class AnalogLayer(torch.nn.Module):
    def __init__(self, num_of_adcs=p, device=None):
        super(AnalogLayer, self).__init__()
        self.length = 784
        self.p = num_of_adcs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize phi
        if trainable_analog:
            self.phi = torch.nn.Parameter(torch.randn(self.length, p, device=self.device), requires_grad=True)
        else:
            self.phi = torch.zeros([self.length, p], device=self.device)

        # Initialize A
        self.A = torch.zeros([self.length, p], device=self.device)
        weights = torch.sqrt(torch.cat([
            torch.tensor([1.], device=self.device).div(self.length),
            torch.tensor([2.], device=self.device).div(self.length).expand(self.length - 1)
        ]))

        for i in range(self.length):
            for j in range(self.p):
                self.A[i, j] = weights[i]

    def forward(self, x):
        # Ensure runtime tensors are created on the same device as the input
        device = x.device
        phase_shifted_cos = torch.cos(
            np.pi / self.length *
            (torch.arange(self.p, device=device).unsqueeze(0) + 0.5) *
            torch.arange(self.length, dtype=torch.float32, device=device).unsqueeze(1) +
            self.phi
        )
        y = torch.mm(x, self.A * phase_shifted_cos)
        return y.type(torch.FloatTensor).to(device)


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
        self.num_of_adcs = num_of_adcs
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
            self.W_tensor = W_tensor.to(device)
        self.amplitude = 10000

    def forward(self, x):
        if noisy_training:
            training_noise = torch.randn_like(self.W_tensor) * (noise_sigma ** 0.5)
            W_tensor_noisy = self.W_tensor + training_noise
        else:
            W_tensor_noisy = self.W_tensor
        W_tensor_noisy = W_tensor_noisy.to(device)
        length = int(x.size(0))
        delta = 1e-30
        Q = torch.zeros([length, self.num_of_adcs, self.num_of_bits], device=device)
        q = torch.zeros([length, self.num_of_adcs], device=device)
        m = self.W_tensor.size(0) - 1
        for j in reversed(range(self.num_of_bits)):
            bit_sum = W_tensor_noisy[m] * Vref
            m = m - 1
            for k in range(j + 1, self.num_of_bits):
                bit_sum = bit_sum + (Q[:, :, k] + 1) / 2 * W_tensor_noisy[m] * Vr
                m = m - 1
            Q[:, :, j] = torch.tanh(self.amplitude * (x - bit_sum + delta))
        for b in reversed(range(self.num_of_bits)):
            q = q + (Q[:, :, b] + 1) / 2 * Vr * (2 ** b)
        return q.type(torch.FloatTensor), Q.type(torch.FloatTensor), W_tensor_noisy


"""
True Quantization Layer:
Act similar as the original quantization function but instead of using the "soft to hard" approach it uses a real
sign function in the process of building the digital output.
"""


class TrueQuantizationLayer(nn.Module):
    def __init__(self, W_tensor, num_of_bits, num_of_adcs=p):
        super(TrueQuantizationLayer, self).__init__()
        self.num_of_adcs = num_of_adcs
        self.num_of_bits = num_of_bits
        if trainable_adc:
            self.W_tensor = W_tensor.detach()
        else:
            self.W_tensor = W_tensor

    def forward(self, x):
        if noisy_inference:
            inference_noise = torch.randn_like(self.W_tensor) * (noise_sigma ** 0.5)
            W_tensor_noisy = self.W_tensor + inference_noise
        else:
            W_tensor_noisy = self.W_tensor
        W_tensor_noisy = W_tensor_noisy.to(device)
        length = int(x.size(0))
        delta = 1e-30
        Q = torch.zeros([length, self.num_of_adcs, self.num_of_bits], device=device)
        q = torch.zeros([length, self.num_of_adcs], device=device)
        m = self.W_tensor.size(0) - 1
        for j in reversed(range(self.num_of_bits)):
            bit_sum = W_tensor_noisy[m] * Vref
            m = m - 1
            for k in range(j + 1, self.num_of_bits):
                bit_sum = bit_sum + (Q[:, :, k] + 1) / 2 * W_tensor_noisy[m] * Vr
                m = m - 1
            Q[:, :, j] = torch.sign(x - bit_sum + delta)
        for b in reversed(range(self.num_of_bits)):
            q = q + (Q[:, :, b] + 1) / 2 * Vr * (2 ** b)
        return q.type(torch.FloatTensor), Q.type(torch.FloatTensor), W_tensor_noisy


"""
Digital Processing:
The input to the network expects a tensor shape according to the number of ADCs and time samples.
The output is later used for classification tasks when combined with a cross-entropy loss function.
"""


class digitalDNN(nn.Module):
    def __init__(self, num_of_adcs):
        super(digitalDNN, self).__init__()
        # self.L1 = torch.nn.Linear(784, 64, bias=True).float()
        self.L1 = torch.nn.Linear(num_of_adcs, num_of_adcs * 2, bias=True).float()
        self.L2 = torch.nn.Linear(num_of_adcs * 2, 1, bias=True).float()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y = F.relu(self.L1(x))
        y = self.L2(y)
        # Sigmoid is added to make sure the values are between 0 and 1
        y = self.sigmoid(y)
        return y


"""
The AGC implements an automatic gain control module.
It first normalizes the input signal then scales and shifts it within a voltage range defined by Vdd.
"""


class AGC(nn.Module):
    def __init__(self, num_of_adcs, Vdd=Vdd):
        super(AGC, self).__init__()
        self.Vdd = Vdd
        self.batch_norm = nn.BatchNorm1d(num_of_adcs)

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
        self.AGC = AGC(num_of_adcs, Vdd)
        self.quantization_layer = QuantizationLayer(num_of_bits, num_of_adcs)
        self.digital_network = digitalDNN(num_of_adcs)

    def forward(self, x):
        x = self.analog_layer(x)
        x = self.AGC(x)
        Vin = x
        x, Q, W = self.quantization_layer(x)
        x = x.to(device)
        Q = Q.to(device)
        W = W.to(device)
        x = self.digital_network(x)
        return x, Q, Vin, W


"""
Load teacher model:
"""

if distillation_training:
    teacher_model = FullNet(q_bits).to(device)
    teacher_model.load_state_dict(torch.load(teacher_filename))
    for param in teacher_model.parameters():
        param.requires_grad = False

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
    p_syn = torch.zeros([Q.size(0), Q.size(1), q_bits], device=device)
    m = W_tensor.size(0) - 1
    Y_read = (Q + 1) / 2
    Vk = Vr * Y_read
    for i in reversed(range(q_bits)):
        bit_sum = W_tensor[m] * (Vref ** 2) / Rf
        m = m - 1
        for k in range(i + 1, q_bits):
            bit_sum = bit_sum + (Vk[:, :, k] ** 2) * W_tensor[m] / Rf
            m = m - 1
        p_syn[:, :, i] = (Vin ** 2) / Rf + bit_sum
    synp_power = p_syn.sum(dim=2)
    power_avg = torch.mean(synp_power, dim=0).view(1, p)  # Average the values in each ADC
    power_sum = power_avg.sum()  # Sum the power of all the ADCs to a single total power
    return power_sum


def intPower(Vin, W_tensor, Q, Rf=45e3):
    p_int = torch.zeros([Q.size(0), Q.size(1), q_bits], device=device)
    m = W_tensor.size(0) - 1
    Y_read = (Q + 1) / 2
    Vk = Vr * Y_read
    for i in reversed(range(q_bits)):
        bit_sum = W_tensor[m] * Vref
        m = m - 1
        for k in range(i + 1, q_bits):
            bit_sum = bit_sum + Vk[:, :, k] * W_tensor[m]
            m = m - 1
        p_int[:, :, i] = ((Vin - bit_sum) ** 2) / Rf
    int_power = p_int.sum(dim=2)
    power_avg = int_power.mean(dim=0).view(1, p)  # Average the values in each ADC
    power_sum = power_avg.sum()  # Sum the power of all the ADCs to a single total power
    return power_sum


def totalPower(Vin, W_tensor, Q):
    activationPower = 3e-6
    synapsePower = synpPower(Vin, W_tensor, Q)
    integrationPower = intPower(Vin, W_tensor, Q)
    total_power = synapsePower + integrationPower + activationPower
    return total_power, synapsePower, integrationPower


"""
Plots the quantizer levels for the given input and quantized signals.
x: The input signal values
quan_x: The corresponding quantized signal values
"""


def plotQuant(x, quan_x):
    x_vec = x.detach().cpu().numpy().flatten()
    quan_x = quan_x.detach().cpu().numpy().flatten()
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


"""Create loss functions:"""


def createConstraints(W_tensor):
    Pw = torch.matmul(P, W_tensor)
    regularizations = Pw - b
    return regularizations


def exp_relu(x):
    return torch.exp(torch.relu(x)) - 1.0


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
A function for the distillation loss:
First the function is applying the temprature parameter
Then it will calculate the KL divergence between the student logits and the teacher logits
"""


def distillation_loss(student_output, teacher_output):
    scaled_student = student_output / T
    scaled_teacher = teacher_output / T
    loss = (T ** 2) * torch.mean((scaled_student - scaled_teacher) ** 2)
    return loss


def calcLoss(output_teacher, output, labels, Vin, W_tensor, Q):
    MSELoss = nn.MSELoss()
    totpower, _, _ = totalPower(Vin, W_tensor, Q)
    level_merg_error = constraint_loss(W_tensor)
    l2_reg = torch.norm(W_tensor)**2
    if distillation_training:
        dist_loss = distillation_loss(output, output_teacher)
        loss = MSELoss(output, labels.unsqueeze(-1)) + alpha * dist_loss + beta * totpower + gamma * level_merg_error + eta * l2_reg
       # print("Distillation Loss, ", alpha * dist_loss)
    else:
        loss = MSELoss(output, labels.unsqueeze(-1)) + beta * totpower + gamma * level_merg_error + eta * l2_reg
   # print("CrossEntropy, ", CEloss(output, labels))
    #print("Power, ", totpower * beta)
   # print("Merging, ", level_merg_error * gamma)
   # print("L2, ", eta * l2_reg)
    return loss


"""Decision Regions Plot:"""


def calcDecisionRegions(q_vec):
    q_vec = q_vec.detach().cpu().numpy().flatten()
    decision_regions = list(set(q_vec))
    num_decision_regions = len(decision_regions)
    return num_decision_regions


def plotDecisionRegions(q_vec):
    q_vec = q_vec.detach().cpu().numpy().flatten()
    # q_vec_list = list(q_vec)
    decision_regions = list(set(q_vec))
    num_decision_regions = len(decision_regions)
    print(f"Number of decision regions: {num_decision_regions}")
    # plt.bar(decision_regions, [q_vec_list.count(element) for element in decision_regions], width=1, align='center')
    # if trainable_adc:
    #     plt.title("Trained Quantizer Decision Regions")
    # else:
    #     plt.title("Uniform Quantizer Decision Regions")
    # tick_locations = np.arange(min(decision_regions), max(decision_regions), 0.5)
    # plt.xticks(tick_locations)
    # plt.show()
    return None


"""Training:"""


def train(network, learning_rate, batch_size, train_loader, validation_loader, x_train, x_valid, s_train, s_valid):
    final_network = network
    lr = learning_rate
    Nepochs = number_of_epochs
    optimizer = torch.optim.Adam(network.parameters(), lr, betas=(0.9, 0.999))
    x = np.linspace(0, Vdd, 784 * p, dtype=np.single)
    x = torch.tensor(x.reshape((784, p))).to(device)
    train_mse, valid_mse = [], []
    train_err, valid_err = [], []
    power_epoch, totpower, synppower, intpower = [], [], [], []
    valid_loss_min = np.Inf
    for epoch in range(Nepochs):
        network.train()
        train_loss = 0.0
        valid_loss = 0.0
        total_power, synapse_power, integration_power = 0.0, 0.0, 0.0
        if epoch % 10 == 0:
            print("epoch = ", epoch)
            print("W_tensor = ", network.quantization_layer.W_tensor)
            print("Phase = ", network.analog_layer.phi)
        for i, (inputs_t, labels_t) in enumerate(train_loader):
            inputs_t, labels_t = inputs_t.to(device), labels_t.to(device)
            optimizer.zero_grad()  # Clear the gradient
            output_t, Q_t, Vin_t, W_t = network(inputs_t)  # Forward pass to compute outputs
            # print("Training Loss: ")
            if distillation_training:
                # Get output from the teacher model
                output_teacher, _, _, _ = teacher_model(inputs_t)
                # Batch loss
                loss = calcLoss(output_teacher, output_t, labels_t, Vin_t, W_t, Q_t)
            else:
                loss = calcLoss(None, output_t, labels_t, Vin_t, W_t, Q_t)
            # Compute gradients of the loss
            loss.backward()  # Compute gradients of the loss
            optimizer.step()  # Weight update and parameter optimization
            # Calculate the batch loss - loss.item() is returning the loss of the entire mini-batch
            train_loss = train_loss + loss.item() * batch_size
        train_loss = train_loss / train_size
        train_err.append(train_loss)
        print('The training loss at epoch {} is {:.5f}'.format(epoch, train_loss))
        train_out, _, _, _ = network(x_train)
        mse = torch.mean((train_out - s_train.unsqueeze(-1)) ** 2)
        train_mse.append(mse.detach().cpu().numpy())
        # Model Validation
        validation_net = copy.deepcopy(network)
        validation_net.eval()
        with torch.no_grad():
            if trainable_adc:
                validation_net.quantization_layer = TrueQuantizationLayer(network.quantization_layer.W_tensor, q_bits)
            for i, (inputs_v, labels_v) in enumerate(validation_loader):
                inputs_v, labels_v = inputs_v.to(device), labels_v.to(device)
                output_v, Q_v, Vin_v, W_v = validation_net(inputs_v)  # Forward pass to compute outputs
                # Batch loss
                # print("Validation Loss: ")
                if distillation_training:
                    # Get output from the teacher model
                    output_teacher, _, _, _ = teacher_model(inputs_v)
                    # Batch loss
                    loss_v = calcLoss(output_teacher, output_v, labels_v, Vin_v, W_v, Q_v)
                else:
                    loss_v = calcLoss(None, output_v, labels_v, Vin_v, W_v, Q_v)
                # Calculate the validation loss
                valid_loss = valid_loss + loss_v.item() * batch_size
                # Calculate the validation power
                totalP, synpP, intP = totalPower(Vin_v, W_v, Q_v)
                total_power += totalP.item() * batch_size
                synapse_power += synpP.item() * batch_size
                integration_power += intP.item() * batch_size
        # Calculate the validation loss of the current epoch
        valid_loss = valid_loss / valid_size
        valid_err.append(valid_loss)
        print('The validation loss at epoch {} is {:.5f}'.format(epoch, valid_loss))
        total_power = total_power / valid_size
        synapse_power = synapse_power / valid_size
        integration_power = integration_power / valid_size
        totpower.append(total_power * 1_000_000)
        synppower.append(synapse_power * 1_000_000)
        intpower.append(integration_power * 1_000_000)
        power_epoch.append(epoch)
        if epoch % 1 == 0 and trainable_adc:
            print("The number of decision regions: ")
            print(calcDecisionRegions(quantizerOut(x, validation_net.quantization_layer.W_tensor)))
        if epoch % 1 == 0 and not trainable_adc:
            print("The number of decision regions: ")
            print(calcDecisionRegions(quantizerOut(x, validation_net.quantization_layer.W_tensor)))
        valid_out, _, _, _ = validation_net(x_valid)
        mse = torch.mean((valid_out - s_valid.unsqueeze(-1)) ** 2)
        valid_mse.append(mse.detach().cpu().numpy())
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            final_network = copy.deepcopy(network)
            valid_loss_min = valid_loss
    return train_err, train_mse, valid_err, valid_mse, final_network, power_epoch, totpower, synppower, intpower


"""Test"""


def test(network, batch_size, test_loader, x_test, s_test):
    test_loss = 0.0
    # Model Validation
    test_net = copy.deepcopy(network).to(device)
    test_net.eval()
    with torch.no_grad():
        if trainable_adc:
            test_net.quantization_layer = TrueQuantizationLayer(network.quantization_layer.W_tensor, q_bits)
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass to compute outputs
            output, Q, Vin, W = test_net(inputs)
            # Get output from the teacher model
            if distillation_training:
                output_teacher, _, _, _ = teacher_model(inputs)
                # Batch loss
                loss = calcLoss(output_teacher, output, labels, Vin, W, Q)
            else:
                loss = calcLoss(None, output, labels, Vin, W, Q)
            # Calculate the test loss
            test_loss = test_loss + loss.item() * batch_size
        # Calculate the validation loss of the current epoch
    test_loss = test_loss / test_size
    print('The test loss is {:.5f}'.format(test_loss))
    if noise_sigma > 0:
        num_runs = 400
    else:
        num_runs = 1
    sum_total_power = 0
    sum_synp_power = 0
    sum_intp_power = 0
    sum_test_mse = 0
    for r in range(num_runs):
        # Calc mse
        test_out, Q, Vin_t, W_t = test_net(x_test)
        mse = torch.mean((test_out - s_test.unsqueeze(-1)) ** 2)
        sum_test_mse += mse
        # Calc power
        totalp, synpp, intp = totalPower(Vin_t, W_t, Q)
        sum_total_power += totalp
        sum_synp_power += synpp
        sum_intp_power += intp
        print("Run number: ", r)
        print('The test mse is {:.5f}'.format(mse))
        print('Total power: ', totalp.detach() * 1_000_000, "\u00B5W")
        print('Synapse power: ', synpp.detach() * 1_000_000, "\u00B5W")
        print('Integration power: ', intp.detach() * 1_000_000, "\u00B5W")
    expected_total_power = sum_total_power / num_runs
    expected_synp_power = sum_synp_power / num_runs
    expected_intp_power = sum_intp_power / num_runs
    expected_mse = sum_test_mse / num_runs
    print('The averaged test mse is {:.5f}'.format(expected_mse))
    print('The total power for the test set is: ', expected_total_power.detach() * 1_000_000, "\u00B5W")
    print('The synapse power for the test set is: ', expected_synp_power.detach() * 1_000_000, "\u00B5W")
    print('The integration power for the test set is: ', expected_intp_power.detach() * 1_000_000, "\u00B5W")
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


"""Plot loss & mse graphs:"""


def plotGraphs(title, results):
    # Extract data from results
    train_err = np.array([res['train_err'] for res in results])
    valid_err = np.array([res['valid_err'] for res in results])
    train_mse = np.array([res['train_mse'] for res in results])
    valid_mse = np.array([res['valid_mse'] for res in results])
    learning_rates = [res['learning_rate'] for res in results]
    batch_sizes = [res['batch_size'] for res in results]

    # Plot loss
    Epochs = range(0, number_of_epochs)
    for i in range(len(results)):
        print("train loss: {:.5f} \u03BC = {} batch size = {}".format(train_err[i, number_of_epochs - 1],
                                                                      learning_rates[i], batch_sizes[i]))
        print("validation loss: {:.5f} \u03BC = {} batch size = {}".format(valid_err[i, number_of_epochs - 1],
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

    # Plot mse
    for i in range(len(results)):
        print("train MSE: {:.5f} \u03BC = {} batch size = {}".format(train_mse[i, number_of_epochs - 1],
                                                                          learning_rates[i], batch_sizes[i]))
        print("validation MSE: {:.5f} \u03BC = {} batch size = {}".format(valid_mse[i, number_of_epochs - 1],
                                                                               learning_rates[i], batch_sizes[i]))
        plt.plot(Epochs, train_mse[i, :],
                 label="Train MSE, \u03BC = " + str(learning_rates[i]) + " batch size = " + str(batch_sizes[i]))
        plt.plot(Epochs, valid_mse[i, :],
                 label="Valid MSE, \u03BC = " + str(learning_rates[i]) + " batch size = " + str(batch_sizes[i]))

    plot_title = title + " MSE Curves"
    plt.title(plot_title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
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


"""Scatter Plots"""


def scatterPlot(vec1, vec2):
    plt.scatter(range(len(vec1)), vec1, color='blue', label='Before analog processing')
    plt.scatter(range(len(vec2)), vec2, color='orange', label='After analog processing')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Image 1 Before Analog VS After Analog')
    plt.legend()
    plt.show()
    return None


"""Show MNIST Images:"""


def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    plt.subplots_adjust(hspace=0.5)
    index = 1
    for x in zip(images, title_texts):
        image = torch.reshape((x[0]), (28, 28))
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=10)
        index += 1
    plt.show()


"""Entropy:"""


def calcEntropy(x):
    x = x.detach().cpu().numpy().flatten() + 10
    values, counts = np.unique(x, return_counts=True)
    pi = counts / len(x)
    entropy = - np.sum(pi * np.log2(pi))
    max_entropy = np.log2(len(counts))
    print("Entropy:", entropy, "Max Entropy:", max_entropy)
    return None


"""Run experiments"""


def run_experiment(batch_size, learning_rate, x_train, x_valid, s_train, s_valid, train_loader, validation_loader):
    network = FullNet(q_bits).to(device)
    train_err, train_mse, valid_err, valid_mse, final_net, power_epoch, total_power, synp_power, int_power = \
        train(network, learning_rate, batch_size, train_loader, validation_loader, x_train, x_valid, s_train,
              s_valid)
    return train_err, train_mse, valid_err, valid_mse, final_net, power_epoch, total_power, synp_power, int_power


"""MNIST Dataset"""


class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())
        labels = torch.Tensor(labels)  # Convert labels to PyTorch Tensor

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            images[i][:] = img / 255.0
        images = torch.Tensor(images)  # Convert images to PyTorch Tensor
        return images, labels

    def load_data(self, validation_split=1 / 6):
        img_train, labels_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        img_test, labels_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)

        img_train, img_val, labels_train, labels_val = train_test_split(img_train, labels_train,
                                                                        test_size=validation_split, random_state=42)

        return (img_train, labels_train), (img_val, labels_val), (img_test, labels_test)


"""Set file paths for MNIST"""
input_path = 'MNIST'
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

"""Load MNIST"""
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                   test_labels_filepath)
(x_train, s_train), (x_valid, s_valid), (x_test, s_test) = mnist_dataloader.load_data()

x_train = x_train.to(device)
x_valid = x_valid.to(device)
x_test = x_test.to(device)
s_train = s_train.to(device)
s_valid = s_valid.to(device)
s_test = s_test.to(device)

# Dividing the labels by 9 to fit the output of the network after the sigmoid
s_train = s_train.long() / 9
s_valid = s_valid.long() / 9
s_test = s_test.long() / 9


"""Main part"""


x_train, x_valid, x_test = standardization(x_train, x_valid, x_test)

train_data = TensorDataset(x_train, s_train)
valid_data = TensorDataset(x_valid, s_valid)
test_data = TensorDataset(x_test, s_test)
train_loader = DataLoader(train_data, batch_size=batch_sizes[0], shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_sizes[0], shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_sizes[0], shuffle=True)

# network = FullNet(q_bits).to(device)
# trained_model = FullNet(q_bits).to(device)

results = []
test_power = []

if not load_trained_model:
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            train_err, train_mse, valid_err, valid_mse, final_net, power_epoch, total_power, synp_power, int_power = \
                run_experiment(batch_size, learning_rate, x_train, x_valid, s_train, s_valid, train_loader,
                               valid_loader)
            results.append({
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "train_err": train_err,
                "train_mse": train_mse,
                "valid_err": valid_err,
                "valid_mse": valid_mse
            })
    plotGraphs("Training & Validation", results)

    # Print Power vs Epoch
    if trainable_adc:
        plotPower(power_epoch, total_power, "Total power VS # of Epochs")
        plotPower(power_epoch, synp_power, "Synapse power VS # of Epochs")
        plotPower(power_epoch, int_power, "Integration power VS # of Epochs")

# Test
"""If load_trained_model = TRUE, we must change "filename" to an existing model before testing"""
if load_trained_model:
    trained_model = FullNet(q_bits).to(device)
    trained_model.load_state_dict(torch.load(load_file_name))
    for param in trained_model.parameters():
        param.requires_grad = False
    test(trained_model, batch_sizes[0], test_loader, x_test, s_test)
else:
    test(final_net, batch_sizes[0], test_loader, x_test, s_test)

# Format a string to use as the filename
if save_file:
    if not trainable_analog:
        filename = f"U_MNIST_{alpha}_{beta}_{gamma}_{T}_{q_bits}_{p}.pth"
        torch.save(final_net.state_dict(), filename)
    elif distillation_training:
        filename = f"D_MNIST_{alpha}_{beta}_{gamma}_{T}_{q_bits}_{p}.pth"
        torch.save(final_net.state_dict(), filename)
    elif noisy_training:
        filename = f"NT_MNIST_{alpha}_{beta}_{gamma}_{T}_{q_bits}_{p}.pth"
        torch.save(final_net.state_dict(), filename)
    elif noisy_inference:
        filename = f"NI_MNIST_{alpha}_{beta}_{gamma}_{T}_{q_bits}_{p}.pth"
        torch.save(final_net.state_dict(), filename)
    else:
        filename = f"Teacher_MNIST_{alpha}_{beta}_{gamma}_{T}_{q_bits}_{p}.pth"
        torch.save(final_net.state_dict(), filename)
