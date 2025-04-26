from stat_kernel import StatKernel
from stat_kernel import make_kernel
import cudaq
from typing import List
import numpy as np

############################################################################
# Quantum Fourier Transform Algorithm
############################################################################

init_input_state = [1, 0, 1]
precision = 2
qubit_count = 3 # must manually add bc of QuakeValue limitations

# Initialize kernel and input_state parameter
quantum_fourier_transform, input_state = make_kernel(List[int])
# Initialize qubits.
qubits = quantum_fourier_transform.qalloc(qubit_count)

# Initialize the quantum circuit to the initial state.
for i in range(qubit_count):
    if input_state[i] == 1:
        quantum_fourier_transform.x(qubits[i])

# should be in a classical state right now
print(cudaq.sample(quantum_fourier_transform, init_input_state))
print(quantum_fourier_transform.classical_assertion(0.05, params=[init_input_state]))
print(quantum_fourier_transform.uniform_assertion(0.05, params=[init_input_state]))
print(quantum_fourier_transform.product_assertion(0.05, 2, 1, params=[init_input_state]))

# Apply Hadamard gates and controlled rotation gates.
for i in range(qubit_count):
    quantum_fourier_transform.h(qubits[i])
    for j in range(i + 1, qubit_count):
        angle = (2 * np.pi) / (2**(j - i + 1))
        quantum_fourier_transform.cr1(angle, [qubits[j]], qubits[i])

# should be superposition
print(cudaq.sample(quantum_fourier_transform, init_input_state))
print(quantum_fourier_transform.classical_assertion(0.05, params=[init_input_state])) 
print(quantum_fourier_transform.uniform_assertion(0.05, params=[init_input_state]))
print(quantum_fourier_transform.product_assertion(0.05, 2, 1, params=[init_input_state]))

# Draw the quantum circuit
print(cudaq.draw(quantum_fourier_transform, init_input_state))

# Print the statevector to the specified precision
statevector = np.array(cudaq.get_state(quantum_fourier_transform, init_input_state))
print(np.round(statevector, precision))
