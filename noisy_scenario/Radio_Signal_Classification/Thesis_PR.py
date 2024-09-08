from functions_PR import *
from torch.utils.data import TensorDataset, DataLoader

# Set seed
torch.manual_seed(hp.seed)  # Seed for accuracy tests
np.random.seed(hp.seed)  # Seed for accuracy tests
random.seed(hp.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(hp.seed)
torch.set_default_dtype(torch.float32)

data_train, data_valid, data_test, label_train, label_valid, label_test = load_data()

# print(analyze_data_distribution(data_train))
train_size = data_train.size(dim=0)
valid_size = data_valid.size(dim=0)
train_data = TensorDataset(data_train, label_train)
valid_data = TensorDataset(data_valid, label_valid)
test_data = TensorDataset(data_test, label_test)
results = []
test_results = []
for lr in hp.learning_rates:
    for bs in hp.batch_sizes:
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, pin_memory=True, worker_init_fn=hp.seed)
        valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=True, pin_memory=True, worker_init_fn=hp.seed)
        test_loader = DataLoader(test_data, batch_size=bs, shuffle=True, pin_memory=True, worker_init_fn=hp.seed)
        train_loader = DeviceDataLoader(train_loader, hp.device)
        valid_loader = DeviceDataLoader(valid_loader, hp.device)
        test_loader = DeviceDataLoader(test_loader, hp.device)
        teacher_model = load_teacher()
        network = FullNet(hp.q_bits).to(hp.device).float()
        for name, param in network.named_parameters():
            print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        train_err, train_acc, valid_err, valid_acc, final_net = (
            train(network, train_loader, valid_loader, lr, bs, train_size, valid_size, teacher_model))
        results.append({"batch_size": bs, "learning_rate": lr, "train_err": train_err, "train_acc": train_acc,
                        "valid_err": valid_err, "valid_acc": valid_acc})
        del train_loader, valid_loader
        torch.cuda.empty_cache()  # Free up memory from train and validation loaders
        if hp.is_ADC:
            test_loss, test_acc, ttotal_power, tsynp_power, tint_power = test(final_net, bs, test_loader, teacher_model)
            test_results.append(
                {"batch_size": bs, "learning_rate": lr, "test_loss": test_loss, "test_acc": test_acc,
                 "total_power": ttotal_power, "synp_power": tsynp_power, "int_power": tint_power})
        else:
            test_loss, test_acc = test(final_net, bs, test_loader)
            test_results.append(
                {"batch_size": bs, "learning_rate": lr, "test_loss": test_loss, "test_acc": test_acc})
save_dict = {'model_state_dict': final_net.state_dict(), 'results': results}
beta_str = str(hp.beta).replace('.', '_')
gamma_str = str(hp.gamma).replace('.', '_')
T_str = str(hp.T).replace('.', '_')
alpha_str = str(hp.alpha).replace('.', '_')
if hp.is_ADC:
    if hp.distillation_training:
        filename = f"D_{int(hp.q_bits)}_bit_{beta_str}_beta_{gamma_str}_gamma_{T_str}_T_{alpha_str}_alpha.pth"
    elif hp.noisy_training:
        filename = f"NoisyT_{int(hp.q_bits)}_bit_{beta_str}_beta_{gamma_str}_gamma.pth"
    elif hp.trainable_adc:
        filename = f"TADC_{int(hp.q_bits)}_bit_{beta_str}_beta_{gamma_str}_gamma.pth"
    else:
        filename = f"Uniform_{int(hp.q_bits)}_bit.pth"
else:
    filename = f"Genie.pth"
torch.save(final_net.state_dict(), filename)
plotGraphs("Training & Validation", results)
printTestResults(test_results)
