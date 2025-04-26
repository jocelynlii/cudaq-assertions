from stat_kernel import StatKernel
from stat_kernel import make_kernel
import cudaq

############################################################################
# Bernstein-Vazirani Algorithm
############################################################################
qubit_count = 5  # Can set to around 30 qubits if you have GPU access

secret_string = [1, 1, 0, 1, 0]  # Change the secret string to whatever you prefer

assert qubit_count == len(secret_string)

k = make_kernel()
qubits = k.qalloc(len(secret_string))
q = k.qalloc(1)
k.x(q)
k.h(q)
k.h(qubits)
print(k.uniform_assertion(pcrit=0.05))
print(k.product_assertion(pcrit=0.05, q0len=5, q1len=1))

# oracle
for index, bit in enumerate(secret_string):
    if bit == 1:
        k.cx(qubits[index], q)

print(k.uniform_assertion(pcrit=0.05))
print(k.product_assertion(pcrit=0.05, q0len=5, q1len=1))

k.h(qubits)
k.mz(qubits)

print(k.classical_assertion(pcrit=0.05))