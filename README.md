# CAMPS.jl

Julia implementation of Clifford-Augmented Matrix Product States for simulating Clifford+T circuits.

## What is this?

This package implements the CAMPS representation from [Liu & Clark (arXiv:2412.17209)](https://arxiv.org/abs/2412.17209). The basic idea is to represent quantum states as:

```
|ψ⟩ = C · |ψ_MPS⟩
```

where C is a Clifford operator and |ψ_MPS⟩ is an MPS. The key insight is that Clifford gates are "free" - we just update C symbolically without touching the MPS. Only non-Clifford gates (like T gates) actually modify the MPS.

We use [QuantumClifford.jl](https://github.com/QuantumSavory/QuantumClifford.jl) for the Clifford tracking and [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) for the MPS backend.

## Installation

```julia
using Pkg
Pkg.develop(path="/path/to/CAMPS.jl")
```

## Basic usage

```julia
using CAMPS

# 5-qubit state
state = CAMPSState(5)
initialize!(state)

# Clifford gates are cheap - just updates the tableau
apply_gate!(state, HGate(1))
apply_gate!(state, CNOTGate(1, 2))

# T gates require actual MPS manipulation
apply_gate!(state, TGate(1))

# Get amplitudes, sample, etc
amp = amplitude(state, [0, 0, 0, 0, 0])
samples = sample_state(state; num_samples=100)
```

## How it works

When you apply a T gate (or any Rz(θ) rotation), we need to figure out what Pauli rotation to actually apply to the MPS. If you've accumulated Clifford C, then applying Rz(θ) on qubit k is equivalent to applying exp(-iθ/2 · P̃) to the MPS, where P̃ = C† Z_k C is the "twisted Pauli".

The cool part is disentangling. If the twisted Pauli has an X or Y on some qubit that's still in |0⟩, we can use that qubit as an ancilla to absorb the rotation without increasing bond dimension. This is OFD (optimization-free disentangling). When OFD doesn't work, we fall back to OBD which does a variational search over two-qubit Cliffords.

### Bond dimension prediction

Given t twisted Paulis, construct a matrix M over GF(2) where M[k,j] = 1 if the k-th Pauli has X or Y on qubit j. The bond dimension scales as 2^(t - rank(M)). So if your Paulis are "independent" (full rank), you stay at bond dim 1.

## Disentangling strategies

```julia
# Try OFD only
simulate_circuit(circuit, n; strategy=OFDStrategy())

# Try OFD, fall back to OBD
simulate_circuit(circuit, n; strategy=HybridStrategy())

# Just apply rotations directly (for comparison)
simulate_circuit(circuit, n; strategy=NoDisentangling())
```

## Available gates

Non-Clifford (modify MPS):
- `TGate(q)`, `TdagGate(q)`
- `RzGate(q, θ)`, `RxGate(q, θ)`, `RyGate(q, θ)`

Clifford (just update tableau):
- `HGate(q)`, `SGate(q)`, `SdagGate(q)`
- `XGate(q)`, `YGate(q)`, `ZGate(q)`
- `CNOTGate(c, t)`, `CZGate(q1, q2)`, `SWAPGate(q1, q2)`

## Running tests

```julia
julia --project=. test/runtests.jl
```

or

```julia
using Pkg
Pkg.test("CAMPS")
```

## References

- Liu & Clark, "Simulating noisy variational quantum algorithms: A polynomial approach" [arXiv:2412.17209](https://arxiv.org/abs/2412.17209)
- Qian et al., "Augmenting Qubits with Cliffords for Quantum Ground State Preparation" [arXiv:2405.09217](https://arxiv.org/abs/2405.09217)
