#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Generative Adversarial Network."""

from typing import Optional, Union, List, Dict, Any
import csv
import os
import logging
import threading

import numpy as np
from scipy.stats import entropy

from qiskit.circuit import QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.aqua import QuantumInstance, AquaError, aqua_globals
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.components.neural_networks.discriminative_network import DiscriminativeNetwork
from qiskit.aqua.components.neural_networks.generative_network import GenerativeNetwork
from qiskit.aqua.components.neural_networks.quantum_generator import QuantumGenerator
from qiskit.aqua.components.neural_networks.numpy_discriminator import NumPyDiscriminator
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.components.uncertainty_models import UnivariateVariationalDistribution
from qiskit.aqua.components.uncertainty_models import MultivariateVariationalDistribution
from qiskit.aqua.utils.dataset_helper import discretize_and_truncate
from qiskit.aqua.utils.validation import validate_min

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class IQGAN(QuantumAlgorithm):
    """Incremental Quantum Generative Adversarial Network.

    The code is forked from Zoufal et al.,
        `Quantum Generative Adversarial Networks for learning and loading random distributions
        <https://www.nature.com/articles/s41534-019-0223-2>`_
    """

    def __init__(self, initial_data: np.ndarray,
                 target_rel_ent: float,
                 num_qubits: np.ndarray,
                 bounds: np.ndarray,
                 batch_size: int = 500,
                 seed: int = 7,
                 freq_storage: bool = False,
                 max_data_length: Optional[int] = None,
                 discriminator: Optional[DiscriminativeNetwork] = None,
                 generator: Optional[GenerativeNetwork] = None,
                 verbose: bool = False,
                 prob_data_real: Optional[np.ndarray] = None,
                 snapshot_dir: Optional[str] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None) -> None:
        """
        Args:
            initial_data: Initial training data of dimension k
            target_rel_ent: Set target level for relative entropy.
                If the training achieves relative entropy equal or lower than tolerance it finishes.
            num_qubits: k numbers of qubits to determine representation resolution,
                i.e. n qubits enable the representation of 2**n values
                [num_qubits_0,..., num_qubits_k-1]
            bounds: k min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
                if univariate data: [min_0,max_0]
            batch_size: Batch size, has a min. value of 1.
            seed: Random number seed
            freq_storage: Flag indicating if storing only data probabilities is enabled.
                Probability distribution will store all passed data. No concept shift can be learned in this case.
            max_data_length: Maximum length of data array.
                When this length is exceeded, the oldest samples will be removed.
            discriminator: Discriminates between real and fake data samples
            generator: Generates 'fake' data samples
            verbose: Additional output on training process.
            prob_data_real: Unknown target data probabilities.
                Used only for real relative entropy output.
            snapshot_dir: Directory in to which to store cvs file with parameters,
                if None (default) then no cvs file is created.
            quantum_instance: Quantum Instance or Backend
        Raises:
            AquaError: invalid input
        """
        validate_min('batch_size', batch_size, 1)
        super().__init__(quantum_instance)
        if initial_data is None:
            raise AquaError('Initial training data not given.')

        self._stop_training = False
        self._training_thread = threading.Thread(target=self._train, daemon=True)
        self._epoch = 0

        self._bounds = np.array(bounds)
        self._num_qubits = num_qubits
        self._batch_size = batch_size
        self._freq_storage = freq_storage
        self._verbose = verbose
        self._snapshot_dir = snapshot_dir
        self._g_loss = []  # type: List[float]
        self._d_loss = []  # type: List[float]
        self._rel_entr = []  # type: List[float]
        self._target_rel_ent = target_rel_ent
        self._max_data_length = max_data_length
        self._random_seed = seed
        self._training_handler = None

        if prob_data_real is not None:
            self._prob_data_real = prob_data_real
            self._rel_entr_real = []
            self._rel_entr_data = []
        else:
            self._prob_data_real = None
            self._rel_entr_real = None
            self._rel_entr_data = None

        if self._freq_storage:
            self._data = []
            self._grid_elements = []
            self._prob_data = []
            self._prob_data_length = 0
            self._update(np.array(initial_data))
        else:
            self._data = np.array(initial_data)
            self._update([])

        if generator is None:
            self.set_generator()
        else:
            self._generator = generator

        if discriminator is None:
            self.set_discriminator()
        else:
            self._discriminator = discriminator

        self.seed = self._random_seed

        self._ret = {}  # type: Dict[str, Any]

    @property
    def epoch(self):
        """ Return current epoch number """
        return self._epoch

    def set_training_handler(self, func):
        """  Setup function for handling every training epoch with `epoch: int` input argument """
        self._training_handler = func

    @property
    def seed(self):
        """ Returns random seed """
        return self._random_seed

    @seed.setter
    def seed(self, s):
        """
        Sets the random seed for QGAN and updates the aqua_globals seed
        at the same time

        Args:
            s (int): random seed
        """
        self._random_seed = s
        aqua_globals.random_seed = self._random_seed
        self._discriminator.set_seed(self._random_seed)

    @property
    def target_rel_ent(self):
        """ Returns tolerance for relative entropy """
        return self._target_rel_ent

    @target_rel_ent.setter
    def target_rel_ent(self, t):
        """
        Set target for relative entropy

        Args:
            t (float): or None, Set tolerance level for relative entropy.
                If the training achieves relative
                entropy equal or lower than tolerance it finishes.
        """
        self._target_rel_ent = t

    @property
    def generator(self):
        """ Returns generator """
        return self._generator

    # pylint: disable=unused-argument
    def set_generator(self, generator_circuit: Optional[Union[QuantumCircuit,
                                                              UnivariateVariationalDistribution,
                                                              MultivariateVariationalDistribution]
                                                        ] = None,
                      generator_init_params: Optional[np.ndarray] = None,
                      generator_optimizer: Optional[Optimizer] = None):
        """Initialize generator.

        Args:
            generator_circuit: parameterized quantum circuit which sets
                the structure of the quantum generator
            generator_init_params: initial parameters for the generator circuit
            generator_optimizer: optimizer to be used for the training of the generator
        """
        self._generator = QuantumGenerator(self._bounds, self._num_qubits,
                                           generator_circuit, generator_init_params,
                                           generator_optimizer,
                                           self._snapshot_dir)

    @property
    def discriminator(self):
        """ Returns discriminator """
        return self._discriminator

    def set_discriminator(self, discriminator=None):
        """
        Initialize discriminator.

        Args:
            discriminator (Discriminator): discriminator
        """
        if discriminator is None:
            self._discriminator = NumPyDiscriminator(len(self._num_qubits))
        else:
            self._discriminator = discriminator
        self._discriminator.set_seed(self._random_seed)

    @property
    def g_loss(self) -> List[float]:
        """ Returns generator loss """
        return self._g_loss

    @property
    def d_loss(self) -> List[float]:
        """ Returns discriminator loss """
        return self._d_loss

    @property
    def rel_entr(self) -> List[float]:
        """ Returns relative entropy between target and trained distribution """
        return self._rel_entr

    def get_rel_entr(self) -> float:
        """ Get relative entropy between target and trained distribution """
        samples_gen, prob_gen = self._generator.get_output(self._quantum_instance)
        temp = np.zeros(len(self._grid_elements))
        for j, sample in enumerate(samples_gen):
            for i, element in enumerate(self._grid_elements):
                if sample == element:
                    temp[i] += prob_gen[j]
        prob_gen = temp
        print('Generator parameters: ', self._generator._bound_parameters)
        print('Generated probabilities: ', prob_gen)
        prob_gen = [1e-8 if x == 0 else x for x in prob_gen]
        rel_entr = entropy(prob_gen, self._prob_data)
        print('')
        return rel_entr

    @property
    def rel_entr_real(self) -> List[float]:
        """ Returns relative entropy between unknown target and trained distribution """
        return self._rel_entr_real

    def get_rel_entr_real(self) -> float:
        """ Get relative entropy between unknown target and trained distribution """
        samples_gen, prob_gen = self._generator.get_output(self._quantum_instance)
        temp = np.zeros(len(self._grid_elements))
        for j, sample in enumerate(samples_gen):
            for i, element in enumerate(self._grid_elements):
                if sample == element:
                    temp[i] += prob_gen[j]
        prob_gen = temp
        prob_gen = [1e-8 if x == 0 else x for x in prob_gen]
        rel_entr = entropy(prob_gen, self._prob_data_real)
        return rel_entr

    @property
    def rel_entr_data(self) -> List[float]:
        """ Returns relative entropy between unknown target and known data """
        return self._rel_entr_data

    def get_rel_entr_data(self) -> float:
        """ Get relative entropy between unknown target and known data """
        rel_entr = entropy(self._prob_data, self._prob_data_real)
        return rel_entr

    def _update(self, data: np.ndarray):
        print('Updating data...')
        if self._freq_storage:
            print('Old grid elements: ', self._grid_elements)
            print('Old data probabilities: ', self._prob_data)
            print('Old data count: ', self._prob_data_length)
            print('New data count: ', len(data))
            print('New data: ', data)
            new_data_length = len(data)
            _, _, new_grid_elements, new_prob_data = \
                discretize_and_truncate(data, self._bounds, self._num_qubits,
                                        return_data_grid_elements=True,
                                        return_prob=True, prob_non_zero=True)
            # Common merged grid elements
            temp_grid_elements = np.unique(np.concatenate((self._grid_elements, new_grid_elements), 0))
            temp_prob_data = np.zeros(len(temp_grid_elements))
            for j, sample in enumerate(temp_grid_elements):
                for i, element in enumerate(self._grid_elements):
                    if sample == element:
                        temp_prob_data[j] += self._prob_data[i] * self._prob_data_length
                        break
                for i, element in enumerate(new_grid_elements):
                    if sample == element:
                        temp_prob_data[j] += new_prob_data[i] * new_data_length
                        break
                # Normalize data
                temp_prob_data[j] /= (self._prob_data_length + new_data_length)
            self._prob_data_length += new_data_length
            self._grid_elements = temp_grid_elements
            self._prob_data = temp_prob_data
            print('Processed data count: ', self._prob_data_length)
            print('Processed grid elements: ', self._grid_elements)
            print('Processed data probabilities: ', self._prob_data)
            if self._prob_data_real is not None:
                print('Unknown real data probabilities: ', self._prob_data_real)
            print('')
        else:
            print('Old data count: ', len(self._data))
            print('New data count: ', len(data))
            print('New data: ', data)
            if self._max_data_length is None:
                self._data = np.append(self._data, np.array(data))
            else:
                elements_left = self._max_data_length - len(data)
                if elements_left > 0:
                    self._data = np.append(self._data[-elements_left:], np.array(data))
                else:
                    self._data = np.array(data[-self._max_data_length:])
            self._data, _, self._grid_elements, self._prob_data = \
                discretize_and_truncate(self._data, self._bounds, self._num_qubits,
                                        return_data_grid_elements=True,
                                        return_prob=True, prob_non_zero=True)
            print('Processed data count: ', len(self._data))
            print('Processed grid elements: ', self._grid_elements)
            print('Processed data probabilities: ', self._prob_data)
            if self._prob_data_real is not None:
                print('Unknown real data probabilities: ', self._prob_data_real)
            print('')

    def update(self, data: np.ndarray, target_rel_ent: Optional[float] = None):
        self.stop_training()
        self._update(data)
        if target_rel_ent is not None:
            self._target_rel_ent = target_rel_ent
        current_rel_entr = self.get_rel_entr()
        print('Current relative entropy: ', current_rel_entr)
        print('Target relative entropy: ', self._target_rel_ent)
        print('')
        if current_rel_entr > self._target_rel_ent:
            self.train()

    def stop_training(self):
        print('Stopping training...')
        self._stop_training = True
        self._training_thread.join()
        self._stop_training = False
        print('Training stopped')
        print('')

    def _store_params(self, e, d_loss, g_loss, rel_entr):
        with open(os.path.join(self._snapshot_dir, 'output.csv'), mode='a') as csv_file:
            fieldnames = ['epoch', 'loss_discriminator',
                          'loss_generator', 'params_generator', 'rel_entropy']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'epoch': e, 'loss_discriminator': np.average(d_loss),
                             'loss_generator': np.average(g_loss), 'params_generator':
                                 self._generator.generator_circuit.params, 'rel_entropy': rel_entr})
        self._discriminator.save_model(self._snapshot_dir)  # Store discriminator model

    def train(self):
        """
        Train the model

        Raises:
            AquaError: Batch size bigger than the number of items in the truncated data set
        """
        print('Training...')
        print('')
        if self._training_thread.is_alive():
            print('Training is already in process')
            print('')
        else:
            self._training_thread = threading.Thread(target=self._train, daemon=True)
            self._training_thread.start()

    def _train(self):
        if self._snapshot_dir is not None:
            with open(os.path.join(self._snapshot_dir, 'output.csv'), mode='w') as csv_file:
                fieldnames = ['epoch', 'loss_discriminator', 'loss_generator', 'params_generator',
                              'rel_entropy']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

        if len(self._data) < self._batch_size and not self._freq_storage:
            raise AquaError(
                'The batch size needs to be less than the '
                'truncated data size of {}'.format(len(self._data)))

        if self._epoch == 0:
            # First training launch
            # Add initial relative entropy
            self._rel_entr.append(np.around(self.get_rel_entr(), 4))
            if self._rel_entr_real is not None:
                self._rel_entr_real.append(np.around(self.get_rel_entr_real(), 4))
            if self._rel_entr_data is not None:
                self._rel_entr_data.append(np.around(self.get_rel_entr_data(), 4))

        while True:
            training_data = np.array(self._data)
            aqua_globals.random.shuffle(training_data)
            index = 0
            while (not self._freq_storage and (index + self._batch_size) <= len(training_data)) \
                    or (self._freq_storage and index <= self._prob_data_length):
                if self._freq_storage:
                    real_batch = np.random.choice(self._grid_elements, self._batch_size, p=self._prob_data)
                else:
                    real_batch = training_data[index: index + self._batch_size]
                index += self._batch_size
                generated_batch, generated_prob = self._generator.get_output(self._quantum_instance,
                                                                             shots=self._batch_size)

                if self._verbose:
                    #print('Real batch: ', real_batch)
                    print('Generated batch: ', generated_batch)
                    print('Generated probabilities: ', generated_prob)
                    print('Discriminator training...')
                # 1. Train Discriminator
                ret_d = self._discriminator.train([real_batch, generated_batch],
                                                  [np.ones(len(real_batch)) / len(real_batch),
                                                   generated_prob])
                d_loss_min = ret_d['loss']
                if self._verbose:
                    print('Discriminator loss: ', d_loss_min)
                    print('')
                    print('Generator training...')
                # 2. Train Generator
                self._generator.set_discriminator(self._discriminator)
                ret_g = self._generator.train(self._quantum_instance, shots=self._batch_size)
                g_loss_min = ret_g['loss']
                if self._verbose:
                    print('Generator loss: ', g_loss_min)
                    print('')

            self._d_loss.append(np.around(float(d_loss_min), 4))
            self._g_loss.append(np.around(g_loss_min, 4))

            rel_entr = self.get_rel_entr()
            rel_entr_real = None
            self._rel_entr.append(np.around(rel_entr, 4))
            if self._rel_entr_real is not None:
                rel_entr_real = self.get_rel_entr_real()
                self._rel_entr_real.append(np.around(rel_entr_real, 4))
            if self._rel_entr_data is not None:
                self._rel_entr_data.append(np.around(self.get_rel_entr_data(), 4))
            self._ret['params_d'] = ret_d['params']
            self._ret['params_g'] = ret_g['params']
            self._ret['loss_d'] = np.around(float(d_loss_min), 4)
            self._ret['loss_g'] = np.around(g_loss_min, 4)
            self._ret['rel_entr'] = np.around(rel_entr, 4)
            self._epoch += 1

            if self._snapshot_dir is not None:
                self._store_params(self._epoch, np.around(d_loss_min, 4),
                                   np.around(g_loss_min, 4), np.around(rel_entr, 4))

            print('Epoch ', self._epoch)
            print('Loss Discriminator: ', np.around(float(d_loss_min), 4))
            print('Loss Generator: ', np.around(g_loss_min, 4))
            print('Relative Entropy: ', np.around(rel_entr, 4))
            if rel_entr_real is not None:
                print('Real Relative Entropy: ', np.around(rel_entr_real, 4))
            print('----------------------')
            print('')

            if self._training_handler is not None:
                thread = threading.Thread(target=self._training_handler, args=([self._epoch, rel_entr]))
                thread.daemon = True
                thread.start()
                thread.join(timeout=0.2)

            if rel_entr <= self._target_rel_ent:
                break

            if self._stop_training:
                self._stop_training = False
                break

    def _run(self):
        """
        Run qGAN training

        Returns:
            dict: with generator(discriminator) parameters & loss, relative entropy
        Raises:
            AquaError: invalid backend
        """
        if self._quantum_instance.backend_name == ('unitary_simulator' or 'clifford_simulator'):
            raise AquaError(
                'Chosen backend not supported - '
                'Set backend either to statevector_simulator, qasm_simulator'
                ' or actual quantum hardware')
        self.train()
        self._training_thread.join()

        return self._ret

    @property
    def training_data(self):
        return self._data
