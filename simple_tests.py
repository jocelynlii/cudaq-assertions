from stat_kernel import StatKernel
from stat_kernel import make_kernel
import cudaq

#############################################################################
# basic test - product state, but not uniform nor classical
#############################################################################
k1 = make_kernel()
q = k1.qalloc(2)
k1.h(q[0])
b0 = k1.mz(q[0])
k1.reset(q[0])
k1.x(q[0])
if b0:
    k1.h(q[1])

print(cudaq.sample(k1))
print(k1.classical_assertion(0.05))
print(k1.uniform_assertion(0.05))
print(k1.product_assertion(1, 1, 0.05))

############################################################################
# uniform and product, not classical 
############################################################################

k1 = StatKernel()
q = k1.qalloc(4)
k1.h(q)
k1.cx(q[0], q[1])
k1.cy([q[0], q[1]],q[2])
k1.z(q[2])
k1.swap(q[0], q[1])
k1.swap(q[0], q[3])
k1.swap(q[1], q[2])
k1.r1(3.14159, q[0])
k1.tdg(q[1])
k1.s(q[2])

print(cudaq.sample(k1))
print(k1.classical_assertion(0.05))
print(k1.uniform_assertion(0.05))
print(k1.product_assertion(2, 2, 0.05))

#############################################################################
# basic test - uniform and product state
#############################################################################
k1 = make_kernel()
qubits = k1.qalloc(2)
for i in range(0,2):
    k1.h(qubits[i])

print(cudaq.sample(k1))
print(k1.classical_assertion(0.05))
print(k1.uniform_assertion(0.05))
print(k1.product_assertion(1, 1, 0.05))

#############################################################################
# basic test - entanglement (not product), and not uniform nor classical
#############################################################################
k1 = make_kernel()
qubits = k1.qalloc(2)
k1.h(qubits[0])
for i in range(1,2):
    k1.cx(qubits[0], qubits[i])
k1.mz(qubits)

print(cudaq.sample(k1))
print(k1.classical_assertion(0.05))
print(k1.uniform_assertion(0.05))
print(k1.product_assertion(1, 1, 0.05))

################################################################
# basic test - classical state (x gate only)
################################################################
kernel = StatKernel()
qubits = kernel.qalloc(3)
kernel.x(qubits)
print(cudaq.sample(kernel))
print(kernel.classical_assertion(0.05))
print(kernel.uniform_assertion(0.05))
print(kernel.product_assertion(2,1, 0.05))
