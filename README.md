# cudaq-assertions

In this work, we present statistical assertions that can be used for debugging in CUDA-Q.

Specifically, `stat_kernel.py` contains the class StatKernel, which inherits from CUDA-Q's
PyKernel class, and is the class that contains all the statistical assertions. `stat_kernel.py`
also contains `make_kernel`, which is a function used to create an instance of StatKernel. 

The three assertions we present are `classical_assertion`, `uniform_assertion`, and
`product_assertion`.

The typical workflow to use our tool involves creating a CUDA-Q dynamic kernel by calling
the `make_kernel` function, defined in `stat_kernel.py`. Then, a user can build their circuit 
like a normal dynamic kernel. At any point within the circuit, any statistical assertion can 
be invoked by calling the assertion for the specific kernel.

bv.py, qft.py, and simple_tests.py depict tests used to verify the correctness of our 
debugging tool. These tests serve as usage examples. 
