import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import GradScaler, autocast
import copy
import random
import time
from sklearn.model_selection import train_test_split
from dataset_preparation_PR import *
from deep_learning_models_torch_PR import *
import hyperparameters_PR as hp
import gc

Vr = hp.Vdd / (2 ** hp.q_bits)  # Writing voltage
Vref = Vr  # Reference voltage
# Set seed
torch.manual_seed(hp.seed)  # Seed for accuracy tests
np.random.seed(hp.seed)  # Seed for accuracy tests
random.seed(hp.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(hp.seed)


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True) if torch.cuda.is_available() else data


class DeviceDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dataloader)


def load_data(file_path_data=r"PLACE HERE DATA PATH FOR HF RADIO DATASET .NPY FILE",
              file_path_labels=r"PLACE HERE DATA PATH FOR HF RADIO LABELS .CSV FILE"):
    LoadDatasetObj = LoadDataset()
    data, label = LoadDatasetObj.load_iq_samples(file_path_data, file_path_labels)
    #plotTime(data, label)
    #plotTime_with_CIS(data)
    # Split the dataset using stratified sampling to maintain class distribution
    train_data, valid_test_data, train_labels, valid_test_labels = train_test_split(
        data, label, test_size=0.3, random_state=42, shuffle=True, stratify=label)

    # Now split the valid_test_data into validation and test sets using stratified sampling
    valid_data, test_data, valid_labels, test_labels = train_test_split(
        valid_test_data, valid_test_labels, test_size=0.5, random_state=42, shuffle=True, stratify=valid_test_labels)

    # Convert to PyTorch tensors
    train_d = torch.tensor(train_data)
    valid_d = torch.tensor(valid_data)
    test_d = torch.tensor(test_data)

    train_l = torch.tensor(train_labels, dtype=torch.long).squeeze()
    valid_l = torch.tensor(valid_labels, dtype=torch.long).squeeze()
    test_l = torch.tensor(test_labels, dtype=torch.long).squeeze()

    return train_d, valid_d, test_d, train_l, valid_l, test_l


def analyze_data_distribution(data, sample_size=100000):
    data = data.cpu().detach().numpy()
    flattened_data = data.flatten()
    mean = np.mean(flattened_data)
    median = np.median(flattened_data)
    std_dev = np.std(flattened_data)
    percentiles = np.percentile(flattened_data, [5, 25, 75, 95])
    print("Mean:", mean)
    print("Median:", median)
    print("Standard Deviation:", std_dev)
    print("5th Percentile:", percentiles[0])
    print("25th Percentile:", percentiles[1])
    print("75th Percentile:", percentiles[2])
    print("95th Percentile:", percentiles[3])
    if len(flattened_data) > sample_size:
        sampled_data = np.random.choice(flattened_data, sample_size, replace=False)
    else:
        sampled_data = flattened_data
    plt.figure(figsize=(10, 6))
    plt.hist(sampled_data, bins=50, edgecolor='k', alpha=0.7)
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(median, color='g', linestyle='dashed', linewidth=1, label='Median')
    plt.axvline(percentiles[3], color='b', linestyle='dashed', linewidth=1, label='95th Percentile')
    plt.title('Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    return {
        "mean": mean,
        "median": median,
        "std_dev": std_dev,
        "5th_percentile": percentiles[0],
        "25th_percentile": percentiles[1],
        "75th_percentile": percentiles[2],
        "95th_percentile": percentiles[3]
    }


def createMatrixP():
    # Create the list of decimal arrays
    list_of_arrays = [np.linspace(0, 2 ** hp.q_bits - 1, 2 ** hp.q_bits, dtype=int)]
    for i in reversed(range(hp.q_bits - 1)):
        list_of_arrays.append(np.linspace(2 ** (i + 1) - 1, 0, 2 ** (i + 1), dtype=int))

    # Convert decimal arrays into arrays of bits
    for i in reversed(range(hp.q_bits)):
        list_of_arrays[hp.q_bits - 1 - i] = (
                (list_of_arrays[hp.q_bits - 1 - i].reshape(-1, 1) & (2 ** np.arange(i, -1, -1))) != 0).astype(int)
    for i in range(hp.q_bits):
        list_of_arrays[i] = np.where(list_of_arrays[i][:] == 0, -1, list_of_arrays[i][:])

    # Create zero padding vectors
    P = list_of_arrays[0]
    for b in range(1, hp.q_bits):
        padding = np.zeros([2 ** b - 1, hp.q_bits - b])
        np_arr = list_of_arrays[b]
        # Insert zeros between the bits
        for i in range(1, 2 ** hp.q_bits - (2 ** b - 1), 2 ** b):
            np_arr = np.insert(np_arr, i, padding, axis=0)
        pad_beginning = np.zeros((2 ** (b - 1), hp.q_bits - b))
        pad_ending = np.zeros((2 ** (b - 1) - 1, hp.q_bits - b))
        np_arr = np.concatenate((pad_beginning, np_arr), axis=0)
        np_arr = np.concatenate((np_arr, pad_ending), axis=0)
        P = np.concatenate((P, np_arr), axis=1)

    P = torch.from_numpy(P).float()

    # Create b
    b = torch.zeros(2 ** hp.q_bits, 1)
    b[0] = -2 ** hp.q_bits
    P = P.to(hp.device)
    b = b.to(hp.device)
    return P, b


P, b = createMatrixP()


def matrix2vector(W):
    W_tensor = torch.Tensor([W[i, j] for
                             i in range(hp.q_bits) for j in reversed(range(hp.q_bits)) if W[i, j] != 0]).view(-1, 1)
    return W_tensor


def quantizerOut(y, W_tensor):
    size = 1000
    delta = 1e-30
    Q = torch.zeros([size, size, hp.q_bits], device=hp.device)
    q = torch.zeros([size, size], device=hp.device)
    m = W_tensor.size(0) - 1
    for j in reversed(range(hp.q_bits)):
        bit_sum = W_tensor[m] * Vref
        m = m - 1
        for k in range(j + 1, hp.q_bits):
            bit_sum = bit_sum + (Q[:, :, k] + 1) / 2 * W_tensor[m] * Vr
            m = m - 1
        Q[:, :, j] = torch.sign(y - bit_sum + delta)
    for b in reversed(range(hp.q_bits)):
        q = q + (Q[:, :, b] + 1) / 2 * Vr * (2 ** b)
    return q.to(torch.float32)


class QuantizationLayer(nn.Module):
    def __init__(self, num_of_bits, packet_size, noise_level=1e-3):
        super(QuantizationLayer, self).__init__()
        self.packet_size = packet_size
        self.num_of_bits = num_of_bits

        # Initialize the uniform ADC values
        W_init = torch.zeros([self.num_of_bits, self.num_of_bits], device=hp.device)
        for i in range(self.num_of_bits):
            W_init[i, 0] = 2 ** i
            for j in range(self.num_of_bits):
                if j > i:
                    W_init[i, j] = 2 ** j

        # Convert the matrix to a vector
        W_tensor = matrix2vector(W_init)

        # # Add random noise to the initialization
        # if hp.trainable_adc:
        #     noise = torch.randn_like(W_tensor) * noise_level
        #     W_tensor = W_tensor + noise

        # Make the weights trainable if needed
        if hp.trainable_adc:
            self.W_tensor = torch.nn.Parameter(W_tensor, requires_grad=True)
        else:
            self.W_tensor = W_tensor

        self.amplitude = 10000

    def forward(self, x):
        device = x.device
        if hp.noisy_training:
            training_noise = torch.randn_like(self.W_tensor) * (hp.noise_sigma ** 0.5)
            W_tensor_noisy = self.W_tensor + training_noise
        else:
            W_tensor_noisy = self.W_tensor
        W_tensor_noisy = W_tensor_noisy.to(device)
        length = int(x.size(0))
        delta = 1e-30
        Q = torch.zeros([length, self.packet_size, self.num_of_bits], device=device)
        q = torch.zeros([length, self.packet_size], device=device)
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
        return q.to(torch.float32), Q.to(torch.float32), W_tensor_noisy


class TrueQuantizationLayer(nn.Module):
    def __init__(self, W_tensor, num_of_bits, packet_size):
        super(TrueQuantizationLayer, self).__init__()
        self.packet_size = packet_size
        self.num_of_bits = num_of_bits
        if hp.trainable_adc:
            self.W_tensor = W_tensor.detach()
        else:
            self.W_tensor = W_tensor
        self.W_tensor = self.W_tensor.to(hp.device)

    def forward(self, x):
        device = x.device
        if hp.noisy_inference:
            inference_noise = torch.randn_like(self.W_tensor) * (hp.noise_sigma ** 0.5)
            W_tensor_noisy = self.W_tensor + inference_noise
        else:
            W_tensor_noisy = self.W_tensor
        length = int(x.size(0))
        delta = 1e-30
        Q = torch.zeros([length, self.packet_size, self.num_of_bits], device=device)
        q = torch.zeros([length, self.packet_size], device=device)
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
        return q.to(torch.float32), Q.to(torch.float32), W_tensor_noisy


"""
The AGC implements an automatic gain control module.
It first normalizes the input signal then scales and shifts it within a voltage range defined by Vdd.
"""


class AGC(nn.Module):
    def __init__(self, packet_size=hp.p, Vdd=hp.Vdd):
        super(AGC, self).__init__()
        self.Vdd = Vdd
        self.batch_norm = nn.BatchNorm1d(packet_size, eps=1e-7, device=hp.device)
        self.epsilon = 1e-7

    def forward(self, x):
        x = self.batch_norm(x)
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min + self.epsilon) * self.Vdd
        # analyze_data_distribution(x)
        return x


def IQ_to_complex(I_part, Q_part):
    complex_x = torch.complex(I_part, Q_part)
    return complex_x


class FullNet(nn.Module):
    def __init__(self, num_of_bits, packet_size=hp.p):
        super(FullNet, self).__init__()
        self.agcI = AGC()
        self.agcQ = AGC()
        self.quantization_layerI = QuantizationLayer(num_of_bits, packet_size)
        self.quantization_layerQ = QuantizationLayer(num_of_bits, packet_size)
        self.ChannelIndSpectrogramObj = ChannelIndSpectrogram()
        self.Feature_Extractor = FeatureExtractor(height=410, width=6)

    def forward(self, x):
        Vin_I = self.agcI(x.real)
        Vin_Q = self.agcQ(x.imag)
        if hp.is_ADC:
            I_quant, QI, WI = self.quantization_layerI(Vin_I)
            Q_quant, QQ, WQ = self.quantization_layerQ(Vin_Q)
            complex_x = IQ_to_complex(I_quant, Q_quant)
        else:
            complex_x = IQ_to_complex(Vin_I, Vin_Q)
        cs = self.ChannelIndSpectrogramObj.channel_ind_spectrogram(complex_x)
        s = self.Feature_Extractor(cs)
        if hp.is_ADC:
            return s, QI, QQ, Vin_I, Vin_Q, WI, WQ
        return s

    def convert_to_complex(self, data):
        num_col = data.shape[1]
        assert num_col % 2 == 0, "Number of columns should be even."

        data_complex = data[:, :num_col // 2].type(torch.complex128) + 1j * data[:, num_col // 2:].type(
            torch.complex128)
        return data_complex


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
Power
"""


def synpPower(Vin, W_tensor, Q):
    p_syn = torch.zeros([Q.size(0), Q.size(1), hp.q_bits], device=hp.device)
    m = W_tensor.size(0) - 1
    Y_read = (Q + 1) / 2
    Vk = Vr * Y_read
    for i in reversed(range(hp.q_bits)):
        bit_sum = W_tensor[m] * (Vref ** 2) / hp.Rf
        m = m - 1
        for k in range(i + 1, hp.q_bits):
            bit_sum = bit_sum + (Vk[:, :, k] ** 2) * W_tensor[m] / hp.Rf
            m = m - 1
        p_syn[:, :, i] = (Vin ** 2) / hp.Rf + bit_sum
    synp_power = p_syn.sum(dim=2)
    synp_power_sum = synp_power.sum(dim=1)  # Sum across packets
    synp_power_avg = synp_power_sum.mean(dim=0)  # Average across the batch
    return synp_power_avg


def intPower(Vin, W_tensor, Q, Rf=45e3):
    p_int = torch.zeros([Q.size(0), Q.size(1), hp.q_bits], device=hp.device)
    m = W_tensor.size(0) - 1
    Y_read = (Q + 1) / 2
    Vk = Vr * Y_read
    for i in reversed(range(hp.q_bits)):
        bit_sum = W_tensor[m] * Vref
        m = m - 1
        for k in range(i + 1, hp.q_bits):
            bit_sum = bit_sum + Vk[:, :, k] * W_tensor[m]
            m = m - 1
        p_int[:, :, i] = ((Vin - bit_sum) ** 2) / Rf
    int_power = p_int.sum(dim=2)
    int_power_sum = int_power.sum(dim=1)  # Sum across packets
    int_power_avg = int_power_sum.mean(dim=0)  # Average across the batch
    return int_power_avg


def totalPower(Vin, W_tensor, Q):
    activationPower = 3e-6
    synapsePower = synpPower(Vin, W_tensor, Q)
    integrationPower = intPower(Vin, W_tensor, Q)
    total_power = synapsePower + integrationPower + activationPower
    return total_power, synapsePower, integrationPower


def distillation_loss(student_logits, teacher_logits):
    student_soft = F.log_softmax(student_logits / hp.T, dim=1)
    teacher_soft = F.softmax(teacher_logits / hp.T, dim=1)
    loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (hp.T ** 2)
    return loss


def calcLoss(output_teacher, output, labels, QI, QQ, Vin_I, Vin_Q, WI, WQ):
    CEloss = nn.CrossEntropyLoss()
    # Compute the total power for I and Q components separately
    totpowerI, synPowerI, intPowerI = totalPower(Vin_I, WI, QI)
    totpowerQ, synPowerQ, intPowerQ = totalPower(Vin_Q, WQ, QQ)
    # Sum the total power from both I and Q components
    totpower = totpowerI + totpowerQ
    synPower = synPowerI + synPowerQ
    intPower = intPowerI + intPowerQ
    # Compute the constraint loss for I and Q components separately
    level_merg_error = constraint_loss(WI) + constraint_loss(WQ)
    # Calculate the loss
    if hp.distillation_training:
        loss = (CEloss(output, labels) + hp.beta * totpower + hp.gamma * level_merg_error + hp.alpha
                * distillation_loss(output, output_teacher))
    else:
        loss = CEloss(output, labels) + hp.beta * totpower + hp.gamma * level_merg_error
    return loss, totpower, synPower, intPower


def calcDecisionRegions(q_vec):
    q_vec = q_vec.detach().cpu().numpy().flatten()
    decision_regions = list(set(q_vec))
    num_decision_regions = len(decision_regions)
    return num_decision_regions


def get_random_subset(data, labels, subset_size):
    indices = torch.randperm(len(data))[:subset_size]
    data_subset = data[indices]
    labels_subset = labels[indices]
    data_subset = data_subset.to(hp.device)
    labels_subset = labels_subset.to(hp.device)
    return data_subset, labels_subset


def print_gpu_memory(prefix=""):
    print(f"{prefix} GPU memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"{prefix} GPU memory reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

def print_teacher_model_params(teacher_model, msg="Teacher Model Parameters"):
    params = {name: param.clone() for name, param in teacher_model.named_parameters()}
    print(f"{msg}:")
    for name, param in params.items():
        print(f"{name}: {param.mean().item()}")


def train(network, train_loader, validation_loader, learning_rate, batch_size, train_size, valid_size, teacher_model):
    CEloss = nn.CrossEntropyLoss()
    if hp.trainable_adc:
        # Set different LR for quantization
        quantization_paramsI = list(network.quantization_layerI.parameters())
        quantization_paramsQ = list(network.quantization_layerQ.parameters())
        quantization_params = quantization_paramsI + quantization_paramsQ
        other_params = [param for name, param in network.named_parameters() if
                        'quantization_layer' not in name and 'quantization_layer_I' not in name and 'quantization_layer_Q' not in name]
        optimizer = torch.optim.Adam([{'params': quantization_params, 'lr': 0.0001},
                                      {'params': other_params, 'lr': learning_rate}], betas=(0.9, 0.999), weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    x = torch.linspace(0, hp.Vdd, 1000 * 1000, dtype=torch.float32, device=hp.device).view(1000, 1000)
    train_acc, valid_acc = [], []
    train_err, valid_err = [], []
    valid_loss_min = np.Inf
    no_improvement_count = 0

    best_weights = copy.deepcopy(network.state_dict())

    for epoch in range(hp.number_of_epochs):
        network.train()
        train_loss = 0.0
        total_correct = 0
        total_samples = 0
        if epoch % 10 == 0 and hp.is_ADC:
            print("epoch = ", epoch)
            print("W_tensor_I = ", network.quantization_layerI.W_tensor)
            print("W_tensor_Q = ", network.quantization_layerQ.W_tensor)
        for inputs_t, labels_t in train_loader:
            inputs_t, labels_t = inputs_t.to(hp.device), labels_t.to(hp.device)
            optimizer.zero_grad()

            if hp.is_ADC:
                output_t, QI_t, QQ_t, Vin_It, Vin_Qt, WI_t, WQ_t = network(inputs_t)
                if hp.distillation_training:
                    output_teacher, _, _, _, _, _, _ = teacher_model(inputs_t)
                    loss, _, _, _ = calcLoss(output_teacher, output_t, labels_t, QI_t, QQ_t, Vin_It, Vin_Qt, WI_t, WQ_t)
                else:
                    loss, _, _, _ = calcLoss(None, output_t, labels_t, QI_t, QQ_t, Vin_It, Vin_Qt, WI_t, WQ_t)
            else:
                output_t = network(inputs_t)
                loss = CEloss(output_t, labels_t)
            loss.backward()
            optimizer.step()

            total_correct += (output_t.argmax(1) == labels_t).sum().item()
            total_samples += inputs_t.size(0)
            train_loss += loss.item() * batch_size

        train_loss /= train_size
        train_acc_epoch = total_correct / total_samples
        train_err.append(train_loss)
        train_acc.append(train_acc_epoch)

        print(f'The training loss at epoch {epoch} is {train_loss:.3f}')
        print(f'The training accuracy at epoch {epoch} is {train_acc_epoch:.3f}')
        print_gpu_memory(f"After training epoch {epoch}")

        # Model Validation
        validation_network = copy.deepcopy(network).to(hp.device)
        validation_network.eval()
        valid_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            if hp.is_ADC:
                validation_network.quantization_layerI = TrueQuantizationLayer(network.quantization_layerI.W_tensor,
                                                                               hp.q_bits, hp.p)
                validation_network.quantization_layerQ = TrueQuantizationLayer(network.quantization_layerQ.W_tensor,
                                                                               hp.q_bits, hp.p)
            for inputs_v, labels_v in validation_loader:
                inputs_v, labels_v = inputs_v.to(hp.device), labels_v.to(hp.device)
                if hp.is_ADC:
                    output_v, QI_v, QQ_v, Vin_Iv, Vin_Qv, WI_v, WQ_v = validation_network(inputs_v)
                    if hp.distillation_training:
                        output_teacher, _, _, _, _, _, _ = teacher_model(inputs_v)
                        loss_v, _, _, _ = calcLoss(output_teacher, output_v, labels_v, QI_v, QQ_v, Vin_Iv, Vin_Qv, WI_v, WQ_v)
                    else:
                        loss_v, _, _, _ = calcLoss(None, output_v, labels_v, QI_v, QQ_v, Vin_Iv, Vin_Qv, WI_v, WQ_v)
                else:
                    output_v = validation_network(inputs_v)
                    loss_v = CEloss(output_v, labels_v)

                valid_loss += loss_v.item() * batch_size
                total_correct += (output_v.argmax(1) == labels_v).sum().item()
                total_samples += inputs_v.size(0)

            valid_loss /= valid_size
            valid_acc_epoch = total_correct / total_samples
            valid_err.append(valid_loss)
            valid_acc.append(valid_acc_epoch)

            print('The validation loss at epoch {} is {:.3f}'.format(epoch, valid_loss))
            print('The validation accuracy at epoch {} is {:.3f}'.format(epoch, valid_acc_epoch))
            print_gpu_memory(f"After validation epoch {epoch}")

            if epoch % 50 == 0 and hp.trainable_adc:
                DecisionRegions = calcDecisionRegions(quantizerOut(x, validation_network.quantization_layerI.W_tensor))
                print("The number of decision regions for ADC 1 (I) is: ", DecisionRegions)
                DecisionRegions = calcDecisionRegions(quantizerOut(x, validation_network.quantization_layerQ.W_tensor))
                print("The number of decision regions for ADC 2 (Q) is: ", DecisionRegions)
            print_gpu_memory(f"After decision region calculation for epoch {epoch}")

        #scheduler.step(valid_loss)

        # Save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            best_weights = copy.deepcopy(network.state_dict())  # Save only the weights
            valid_loss_min = valid_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Profiling memory usage after epoch completion
        print_gpu_memory(f"End of epoch {epoch}")

    network.load_state_dict(best_weights)  # Load the best weights at the end of training
    return train_err, train_acc, valid_err, valid_acc, network


"""Plot training and validation accuracy and loss curves"""


def plotGraphs(title, results):
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return np.array(data)

    # Extract data from results and ensure they are in numpy format
    train_err = np.array([to_numpy(res['train_err']) for res in results])
    valid_err = np.array([to_numpy(res['valid_err']) for res in results])
    train_acc = np.array([to_numpy(res['train_acc']) for res in results])
    valid_acc = np.array([to_numpy(res['valid_acc']) for res in results])
    learning_rates = [to_numpy(res['learning_rate']) for res in results]
    batch_sizes = [to_numpy(res['batch_size']) for res in results]

    # Determine the number of epochs based on the length of the results
    number_of_epochs = train_err.shape[1]
    Epochs = range(number_of_epochs)

    # Plot loss
    for i in range(len(results)):
        print("train loss: {:.3f} \u03BC = {} batch size = {}".format(train_err[i, -1],
                                                                      learning_rates[i], batch_sizes[i]))
        print("validation loss: {:.3f} \u03BC = {} batch size = {}".format(valid_err[i, -1],
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
        print("train accuracy: {:.3f} \u03BC = {} batch size = {}".format(train_acc[i, -1],
                                                                          learning_rates[i], batch_sizes[i]))
        print("validation accuracy: {:.3f} \u03BC = {} batch size = {}".format(valid_acc[i, -1],
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


"""Print Test Results"""


def printTestResults(test_results):
    if hp.is_ADC:
        for i, result in enumerate(test_results):
            print(f"Trial {i + 1}: batch_size = {result['batch_size']}, learning_rate = {result['learning_rate']}, "
                  f"test_loss = {result['test_loss']:.4f}, test_acc = {result['test_acc']:.4f}, "
                  f"total_power = {result['total_power']:.4f}, synp_power = {result['synp_power']:.4f}, "
                  f"int_power = {result['int_power']:.4f}")
    else:
        for i, result in enumerate(test_results):
            print(f"Trial {i + 1}: batch_size = {result['batch_size']}, learning_rate = {result['learning_rate']}, "
                  f"test_loss = {result['test_loss']:.4f}, test_acc = {result['test_acc']:.4f}")


"""Plot Power"""


def plotPower(power_epoch, power, title):
    plt.plot(power_epoch, power)
    plot_title = title
    plt.title(plot_title)
    plt.xlabel("Epoch")
    plt.ylabel("Power [\u00B5W]")
    plt.show()


def test(network, batch_size, test_loader, teacher_model=None):
    CEloss = nn.CrossEntropyLoss()
    x = torch.linspace(0, hp.Vdd, 1000 * 1000, dtype=torch.float32, device=hp.device).view(1000, 1000)
    avg_total_power, avg_synp_power, avg_intp_power, avg_acc, avg_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    if hp.noisy_inference:
        num_reps = 50
    else:
        num_reps = 1
    test_netowrk = copy.deepcopy(network).to(hp.device)
    test_netowrk.eval()  # Set the network to evaluation mode
    if hp.is_ADC:
        test_netowrk.quantization_layerI = TrueQuantizationLayer(network.quantization_layerI.W_tensor, hp.q_bits, hp.p)
        test_netowrk.quantization_layerQ = TrueQuantizationLayer(network.quantization_layerQ.W_tensor, hp.q_bits, hp.p)

    with torch.no_grad():
        for i in range(num_reps):
            test_loss, test_correct, total_samples = 0.0, 0.0, 0
            sum_total_power, sum_synp_power, sum_intp_power = 0.0, 0.0, 0.0

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(hp.device), labels.to(hp.device)
                if hp.is_ADC:
                    output, QI, QQ, Vin_I, Vin_Q, WI, WQ = test_netowrk(inputs)
                    if hp.distillation_training:
                        output_teacher, _, _, _, _, _, _ = teacher_model(inputs)
                        loss, totalp, synpp, intp = calcLoss(output_teacher, output, labels, QI, QQ, Vin_I, Vin_Q, WI, WQ)
                    else:
                        loss, totalp, synpp, intp = calcLoss(None, output, labels, QI, QQ, Vin_I, Vin_Q, WI, WQ)
                    sum_total_power += totalp.item()
                    sum_synp_power += synpp.item()
                    sum_intp_power += intp.item()
                else:
                    output = test_netowrk(inputs)
                    loss = CEloss(output, labels)
                # Calculate the test loss
                test_loss += loss.item() * batch_size
                # Calculate accuracy
                _, predicted = torch.max(output, 1)
                test_correct += (predicted == labels).sum().item()  # Count correct predictions
                total_samples += labels.size(0)

            test_loss = test_loss / total_samples
            test_acc = test_correct / total_samples
            print("Inference trial #", i, " accuracy = ", test_acc)
            avg_loss += test_loss
            avg_acc += test_acc

            if hp.is_ADC:
                avg_total_power += sum_total_power / len(test_loader)
                avg_synp_power += sum_synp_power / len(test_loader)
                avg_intp_power += sum_intp_power / len(test_loader)
        if hp.trainable_adc:
            DecisionRegions = calcDecisionRegions(quantizerOut(x, test_netowrk.quantization_layerI.W_tensor))
            print("The number of decision regions for ADC 1 (I) is: ", DecisionRegions)
            DecisionRegions = calcDecisionRegions(quantizerOut(x, test_netowrk.quantization_layerQ.W_tensor))
            print("The number of decision regions for ADC 2 (Q) is: ", DecisionRegions)

    avg_loss /= num_reps
    avg_acc /= num_reps
    if hp.is_ADC:
        avg_total_power = avg_total_power / num_reps
        avg_synp_power = avg_synp_power / num_reps
        avg_intp_power = avg_intp_power / num_reps
        return avg_loss, avg_acc, avg_total_power, avg_synp_power, avg_intp_power

    return avg_loss, avg_acc


"""
Load teacher that will be used for distillation training
"""


def load_teacher():
    if hp.distillation_training:
        teacher_model = FullNet(hp.q_bits).to(hp.device)
        try:
            teacher = torch.load(hp.teacher_model)
            teacher_model.load_state_dict(teacher)
        except RuntimeError as e:
            print(f"Error loading model: {e}")
        for param in teacher_model.parameters():
            param.requires_grad = False
    else:
        teacher_model = None
    return teacher_model
