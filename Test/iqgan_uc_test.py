import numpy as np
np.random.seed = 71
import matplotlib.pyplot as plt
from qiskit import BasicAer
from qiskit import IBMQ
from qiskit.circuit.library import TwoLocal, UniformDistribution
from qiskit.aqua import QuantumInstance
from Algorithm.iqgan import IQGAN
from qiskit.aqua.components.neural_networks import NumPyDiscriminator
from qiskit.aqua.utils.dataset_helper import discretize_and_truncate
from multiprocessing.pool import ThreadPool
from Algorithm.data_source import DataSource
pool = ThreadPool(processes=1)
auto_mode = True
real_backend = False

print('\nLearning of stationary lognormal distribution with mean value 1 and sigma value 1\n')

# Initial parameters:
# Set upper and lower data values as list of k min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
bounds = np.array([0., 3.])
# Set number of qubits per data dimension as list of k qubit values [#q_0,...,#q_k-1]
num_qubits = [2]
# Batch size
batch_size = 5
# Target relative entropy
target_rel_ent = 1e-10


# Predefined data source
data_source = DataSource([], batch_size=batch_size)
def take_from_data_source(size: int):
    return data_source.get_elements(size)


# Distribution functions:
def take_from_dist_1(size: int):
    """ Used as an underlying process """
    data = np.array([])
    while len(data) < size:
        dif = size - len(data)
        take_size = (dif + 10) * 2
        new_data = np.random.lognormal(mean=1, sigma=1, size=take_size)
        new_data, _ = discretize_and_truncate(new_data, bounds, num_qubits)
        if len(new_data) > dif:
            new_data = new_data[-dif:]
        data = np.append(data, new_data)
    return data


def take_from_dist_2(size: int):
    """ Used in the non-stationary process learning as an updated process """
    data = np.array([])
    while len(data) < size:
        dif = size - len(data)
        take_size = (dif + 10) * 2
        new_data = np.random.lognormal(mean=2, sigma=1, size=take_size)
        new_data, _ = discretize_and_truncate(new_data, bounds, num_qubits)
        if len(new_data) > dif:
            new_data = new_data[-dif:]
        data = np.append(data, new_data)
    return data


# Unknown target, used just for tests
unknown_target_data = take_from_dist_1(1000000)
_, _, _, unknown_target_data_prob = discretize_and_truncate(
    unknown_target_data, bounds, num_qubits,
    return_data_grid_elements=True, return_prob=True, prob_non_zero=True
)
# Predefined unknown target data probabilities
#unknown_target_data_prob = [0.01935999999999938, 0.28349000000014274, 0.45547000000031473, 0.24168000000010093]
# Number of initial data samples
N = 1*batch_size
# Maximum data length for concept drift
max_data_length = None
# Use frequency histogram as a storage instead of data stack
freq_storage = False

# Initial dataset
real_data = take_from_dist_1(N)

print('Data bounds:', bounds)
print('Batch size:', batch_size)
print('Number of qubits:', num_qubits)
print('Target relative entropy:', target_rel_ent)
print('Initial data length:', N)
print('Initial data:', real_data)
print('Unknown target probabilities:', unknown_target_data_prob)
print('')

# Set quantum instance to run the quantum generator
if real_backend:
    print('Real backend loading...')
    IBMQ.load_account()  # Load account from disk
    IBMQ.providers()  # List all available providers
    provider = IBMQ.get_provider(hub='ibm-q')
    backend = provider.get_backend('ibmq_santiago')
    quantum_instance = QuantumInstance(backend=backend)
    print('Backend:', backend)
    print('')
else:
    quantum_instance = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'))

# Initialize qGAN
iqgan = IQGAN(
    real_data,
    target_rel_ent,
    num_qubits,
    bounds,
    batch_size,
    freq_storage=freq_storage,
    verbose=False,
    max_data_length=max_data_length,
    prob_data_real=unknown_target_data_prob,
    quantum_instance=quantum_instance
)

# Set an initial state for the generator circuit
init_dist = UniformDistribution(sum(num_qubits))
# Generator's depth
k = 2
# Set the ansatz circuit
var_form = TwoLocal(int(np.sum(num_qubits)), 'ry', 'cz', reps=k)
# Generator's initial parameters
#init_params = [0.02789928, 0.02950349, 0.35598046, 0.24428748, 0.02875957, 0.02955343] #lognormal 1 1 (ent=0.001)
#init_params = [0.01436921, 0.02010115, 0.45493457, 0.32251614, 0.0178407,  0.02214568] #lognormal 1 1 (ent=0.1)
init_params = np.zeros(np.sum(num_qubits)*(k+1))
# Set generator circuit by adding the initial distribution infront of the ansatz
g_circuit = var_form.compose(init_dist, front=True)

print('Generator circuit:')
print(g_circuit)
print('')

# Set quantum generator
iqgan.set_generator(generator_circuit=g_circuit, generator_init_params=init_params)
# The parameters have an order issue that following is a temp., workaround
iqgan._generator._free_parameters = sorted(g_circuit.parameters, key=lambda p: p.name)
# Set classical discriminator neural network
discriminator = NumPyDiscriminator(len(num_qubits))
iqgan.set_discriminator(discriminator)

print('Initial relative entropy:', iqgan.get_rel_entr())
print('')


# iQGAN operations:
def operation_stop():
    print('Stop operation received\n')
    iqgan.stop_training()


def operation_start():
    print('Start operation received\n')
    iqgan.train()


def operation_update(size: int):
    print('Update operation received\n')
    iqgan.update(take_from_dist_1(size))


def operation_update_2(size: int):
    print('Update 2 operation received\n')
    iqgan.update(take_from_dist_2(size))


def operation_ent(show: bool = True):
    print('Relative entropy graph operation received\n')
    plt.figure(figsize=(7, 5))
    plt.title('Relative Entropy')
    rel_entr = iqgan.rel_entr
    plt.plot(np.linspace(0, iqgan.epoch, len(rel_entr)), rel_entr, color='b', lw=2, ls='-')
    plt.grid()
    plt.xlabel('time steps')
    plt.ylabel('relative entropy')
    if show:
        plt.show()
    else:
        plt.savefig('ent.png')


def operation_ent_real(show: bool = True):
    print('Real relative entropy graph operation received\n')
    plt.figure(figsize=(7, 5))
    plt.title('Real Relative Entropy')
    rel_entr_real = iqgan.rel_entr_real
    plt.plot(np.linspace(0, iqgan.epoch, len(rel_entr_real)), rel_entr_real, color='b', lw=2, ls='-')
    plt.grid()
    plt.xlabel('time steps')
    plt.ylabel('relative entropy')
    if show:
        plt.show()
    else:
        plt.savefig('ent_real.png')


def operation_ent_data(show: bool = True):
    print('Data relative entropy graph operation received\n')
    plt.figure(figsize=(7, 5))
    plt.title('Data Relative Entropy')
    rel_entr_data = iqgan.rel_entr_data
    plt.plot(np.linspace(0, iqgan.epoch, len(rel_entr_data)), rel_entr_data, color='b', lw=2, ls='-')
    plt.grid()
    plt.xlabel('time steps')
    plt.ylabel('relative entropy')
    if show:
        plt.show()
    else:
        plt.savefig('ent_data.png')


def operation_cdf(show: bool = True):
    print('Cumulative distribution function graph operation received\n')
    target = unknown_target_data
    target = np.round(target)
    target = target[target <= bounds[1]]
    temp = []
    for i in range(int(bounds[1] + 1)):
        temp += [np.sum(target == i)]
    target = np.array(temp / sum(temp))
    plt.figure(figsize=(7, 5))
    plt.title('Cumulative Distribution Function')
    samples_g, prob_g = iqgan.generator.get_output(iqgan.quantum_instance, shots=10000)
    samples_g = np.array(samples_g)
    samples_g = samples_g.flatten()
    plt.bar(samples_g, np.cumsum(prob_g), color='royalblue', width=0.8, label='approximation')
    plt.plot(np.cumsum(target), '-o', label='target', color='b', linewidth=4, markersize=12)
    plt.xticks(np.arange(min(samples_g), max(samples_g) + 1, 1.0))
    plt.grid()
    plt.legend(loc='best')
    if show:
        plt.show()
    else:
        plt.savefig('cdf.png')


def operation_real_hist(show: bool = True):
    print('Real histogram graph operation received\n')
    target = unknown_target_data
    temp = []
    for i in range(int(bounds[1] + 1)):
        temp += [np.sum(target == i)]
    target = np.array(temp / sum(temp))
    plt.figure(figsize=(7, 5))
    plt.title('Histogram')
    samples_g, prob_g = iqgan.generator.get_output(iqgan.quantum_instance, shots=10000)
    samples_g = np.array(samples_g)
    samples_g = samples_g.flatten()
    plt.bar(samples_g, prob_g, color='royalblue', width=0.8, label='approximation')
    plt.plot(target, '-o', label='unknown target', color='b', linewidth=4, markersize=12)
    plt.xticks(np.arange(min(samples_g), max(samples_g) + 1, 1.0))
    plt.grid()
    plt.legend(loc='best')
    if show:
        plt.show()
    else:
        plt.savefig('hist_real.png')


def operation_hist(show: bool = True):
    print('Histogram graph operation received\n')
    target = iqgan.training_data
    temp = []
    for i in range(int(bounds[1] + 1)):
        temp += [np.sum(target == i)]
    target = np.array(temp / sum(temp))
    plt.figure(figsize=(7, 5))
    plt.title('Histogram')
    samples_g, prob_g = iqgan.generator.get_output(iqgan.quantum_instance, shots=10000)
    samples_g = np.array(samples_g)
    samples_g = samples_g.flatten()
    plt.bar(samples_g, prob_g, color='royalblue', width=0.8, label='approximation')
    plt.plot(target, '-o', label='target', color='b', linewidth=4, markersize=12)
    plt.xticks(np.arange(min(samples_g), max(samples_g) + 1, 1.0))
    plt.grid()
    plt.legend(loc='best')
    if show:
        plt.show()
    else:
        plt.savefig('hist.png')


def operation_gen_loss(show: bool = True):
    print('Generator loss graph operation received\n')
    plt.figure(figsize=(7, 5))
    plt.title('Generator Loss')
    plt.ylim([0.5, 1])
    gen_loss = iqgan.g_loss
    plt.plot(np.linspace(0, iqgan.epoch, len(gen_loss)), gen_loss, color='b', lw=2, ls='-')
    plt.grid()
    plt.xlabel('time steps')
    plt.ylabel('loss')
    if show:
        plt.show()
    else:
        plt.savefig('gen_loss.png')


def operation_dis_loss(show: bool = True):
    print('Discriminator loss graph operation received\n')
    plt.figure(figsize=(7, 5))
    plt.title('Discriminator Loss')
    plt.ylim([0.5, 1])
    dis_loss = iqgan.d_loss
    plt.plot(np.linspace(0, iqgan.epoch, len(dis_loss)), dis_loss, color='b', lw=2, ls='-')
    plt.grid()
    plt.xlabel('time steps')
    plt.ylabel('loss')
    if show:
        plt.show()
    else:
        plt.savefig('dis_loss.png')


# Auto mode handler function
def training_handler(epoch: int, rel_ent: float):
    if epoch == 90:
        operation_stop()
    elif epoch % 10 == 0:
       operation_update(batch_size)


# Setup auto handler
if auto_mode:
    iqgan.set_training_handler(training_handler)

# Run training async
async_result = pool.apply_async(iqgan.run)

# Manual testing loop
while True:
    code = input('').lower()
    if code == 'stop':
        operation_stop()
    elif code == 'start':
        operation_start()
    elif code == 'update':
        operation_update(batch_size)
    elif code == 'update_2':
        operation_update_2(batch_size)
    elif code == 'cdf':
        operation_cdf()
    elif code == 'hist':
        operation_hist()
    elif code == 'hist_real':
        operation_real_hist()
    elif code == 'ent':
        operation_ent()
    elif code == 'ent_real':
        operation_ent_real()
    elif code == 'ent_data':
        operation_ent_data()
    elif code == 'gen_loss':
        operation_gen_loss()
    elif code == 'dis_loss':
        operation_dis_loss()
    elif code == 'q':
        # Quit
        break
    else:
        print('Operation', code, 'not found\n')

# Join async result
result = async_result.get()

print('Relative entropy:', iqgan.rel_entr)
print('Real relative entropy:', iqgan.rel_entr_real)
print('Data relative entropy:', iqgan.rel_entr_data)

# Save plots
operation_ent(show=False)
operation_ent_real(show=False)
operation_ent_data(show=False)
operation_gen_loss(show=False)
operation_dis_loss(show=False)
if not real_backend:
    # These operations require a lot of shots from generator
    operation_cdf(show=False)
    operation_real_hist(show=False)
    if not freq_storage:
        # This operation requires training data set
        operation_hist(show=False)