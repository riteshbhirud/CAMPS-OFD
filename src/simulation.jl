
"""
    apply_gate!(state::CAMPSState, gate::Gate;
                strategy::DisentanglingStrategy=HybridStrategy()) -> CAMPSState

Apply a gate to a CAMPS state.

This is the main entry point for gate application. It dispatches to the
appropriate handler based on gate type:
- CliffordGate: Updates Clifford tableau only (free)
- RotationGate: Uses the specified disentangling strategy

# Arguments
- `state::CAMPSState`: CAMPS state (modified in-place)
- `gate::Gate`: Gate to apply
- `strategy::DisentanglingStrategy`: Strategy for non-Clifford gates

# Returns
- `CAMPSState`: Modified state

# Example
```julia
state = CAMPSState(5)
initialize!(state)

apply_gate!(state, HGate(1))           # Clifford - free
apply_gate!(state, CNOTGate(1, 2))     # Clifford - free
apply_gate!(state, TGate(1))           # Non-Clifford - uses disentangling
apply_gate!(state, RzGate(2, π/3))     # Non-Clifford - uses disentangling
```
"""
function apply_gate!(state::CAMPSState, gate::CliffordGate;
                     strategy::DisentanglingStrategy=HybridStrategy())::CAMPSState
    ensure_initialized!(state)
    apply_clifford_gate_to_state!(state, gate)
    return state
end

function apply_gate!(state::CAMPSState, gate::RotationGate;
                     strategy::DisentanglingStrategy=HybridStrategy())::CAMPSState
    ensure_initialized!(state)

    if is_clifford_angle(gate.angle)
        clifford_gate = rotation_to_clifford(gate)
        if clifford_gate !== nothing
            apply_clifford_gate_to_state!(state, clifford_gate)
            return state
        end
    end

    apply_rotation_hybrid!(state, gate.axis, gate.qubit, gate.angle; strategy=strategy)
    return state
end

"""
    rotation_to_clifford(gate::RotationGate) -> Union{CliffordGate, Nothing}

Convert a rotation gate with Clifford angle to a CliffordGate.

Returns nothing if the angle is not a Clifford angle.

# Arguments
- `gate::RotationGate`: Rotation gate

# Returns
- `Union{CliffordGate, Nothing}`: Equivalent CliffordGate or nothing

# Clifford angles
- θ = 0: Identity
- θ = π/2: S or √X/√Y (depending on axis)
- θ = π: Pauli X/Y/Z
- θ = 3π/2: S† or √X†/√Y†
"""
function rotation_to_clifford(gate::RotationGate)::Union{CliffordGate, Nothing}
    θ = normalize_angle(gate.angle)
    q = gate.qubit

    if abs(θ) < 1e-10 || abs(θ - 2π) < 1e-10
        return CliffordGate([(:Z, q), (:Z, q)], [q])
    elseif abs(θ - π/2) < 1e-10
        if gate.axis == :Z
            return SGate(q)
        elseif gate.axis == :X
            return SqrtXGate(q)
        elseif gate.axis == :Y
            return SqrtYGate(q)
        end
    elseif abs(θ - π) < 1e-10
        if gate.axis == :X
            return XGate(q)
        elseif gate.axis == :Y
            return YGate(q)
        elseif gate.axis == :Z
            return ZGate(q)
        end
    elseif abs(θ - 3π/2) < 1e-10
        if gate.axis == :Z
            return SdagGate(q)
        elseif gate.axis == :X
            return SqrtXdagGate(q)
        elseif gate.axis == :Y
            return SqrtYdagGate(q)
        end
    end

    return nothing
end

#==============================================================================#
# CIRCUIT SIMULATION
#==============================================================================#

"""
    SimulationResult

Result of simulating a quantum circuit.

# Fields
- `n_qubits::Int`: Number of qubits
- `n_gates::Int`: Total number of gates
- `n_clifford::Int`: Number of Clifford gates
- `n_non_clifford::Int`: Number of non-Clifford gates
- `n_ofd_applied::Int`: Number of gates handled by OFD
- `n_obd_applied::Int`: Number of gates handled by OBD (fallback)
- `final_bond_dim::Int`: Final maximum bond dimension
- `predicted_bond_dim::Int`: GF(2) predicted bond dimension
- `final_entropy::Float64`: Final maximum entanglement entropy
- `state::CAMPSState`: Final CAMPS state
"""
struct SimulationResult
    n_qubits::Int
    n_gates::Int
    n_clifford::Int
    n_non_clifford::Int
    n_ofd_applied::Int
    n_obd_applied::Int
    final_bond_dim::Int
    predicted_bond_dim::Int
    final_entropy::Float64
    state::CAMPSState
end

"""
    simulate_circuit(circuit::Vector{<:Gate}, n_qubits::Int;
                     strategy::DisentanglingStrategy=HybridStrategy(),
                     max_bond::Int=1024,
                     cutoff::Float64=1e-15,
                     verbose::Bool=false) -> SimulationResult

Simulate a quantum circuit using CAMPS.

# Arguments
- `circuit::Vector{<:Gate}`: Circuit as a sequence of gates
- `n_qubits::Int`: Number of qubits
- `strategy::DisentanglingStrategy`: Disentangling strategy
- `max_bond::Int`: Maximum bond dimension
- `cutoff::Float64`: SVD truncation cutoff
- `verbose::Bool`: Print progress information

# Returns
- `SimulationResult`: Simulation results including final state

# Example
```julia
# Build a circuit
circuit = Gate[
    HGate(1), HGate(2), HGate(3),
    TGate(1), CNOTGate(1, 2), TGate(2),
    CNOTGate(2, 3), TGate(3)
]

# Simulate
result = simulate_circuit(circuit, 3)
println("Final bond dimension: \$(result.final_bond_dim)")
println("Predicted by GF(2): \$(result.predicted_bond_dim)")

# Sample from the final state
samples = sample_mps_multiple(result.state.mps, 1000)
```
"""
function simulate_circuit(circuit::Vector{<:Gate}, n_qubits::Int;
                          strategy::DisentanglingStrategy=HybridStrategy(),
                          max_bond::Int=1024,
                          cutoff::Float64=1e-15,
                          verbose::Bool=false)::SimulationResult

    validate_circuit(circuit, n_qubits)

    state = CAMPSState(n_qubits; max_bond=max_bond, cutoff=cutoff)
    initialize!(state)

    n_clifford = 0
    n_non_clifford = 0
    n_ofd = 0
    n_obd = 0

    for (i, gate) in enumerate(circuit)
        if verbose && i % 100 == 0
            println("Applying gate $i/$(length(circuit))")
        end

        if gate isa CliffordGate
            apply_gate!(state, gate; strategy=strategy)
            n_clifford += 1
        elseif gate isa RotationGate
            if is_clifford_angle(gate.angle)
                apply_gate!(state, gate; strategy=strategy)
                n_clifford += 1
            else
                P_twisted = compute_twisted_pauli(state, gate.axis, gate.qubit)
                control = find_disentangling_qubit(P_twisted, state.free_qubits)

                if control !== nothing && (strategy isa OFDStrategy || strategy isa HybridStrategy)
                    n_ofd += 1
                else
                    n_obd += 1
                end

                apply_gate!(state, gate; strategy=strategy)
                n_non_clifford += 1
            end
        end
    end

    final_bond = get_bond_dimension(state)
    predicted_bond = get_predicted_bond_dimension(state)
    final_entropy = max_entanglement_entropy(state.mps)

    return SimulationResult(
        n_qubits,
        length(circuit),
        n_clifford,
        n_non_clifford,
        n_ofd,
        n_obd,
        final_bond,
        predicted_bond,
        final_entropy,
        state
    )
end

#==============================================================================#
# OBSERVABLE EXTRACTION
#==============================================================================#

"""
    sample_state(state::CAMPSState; num_samples::Int=1) -> Vector{Vector{Int}}

Sample computational basis states from a CAMPS state |ψ⟩ = C|mps⟩.

This properly accounts for the Clifford transformation:
1. Sample from |mps⟩ to get basis state |m⟩
2. Apply C to get the actual sampled state C|m⟩

For computational basis measurements, sampling from |mps⟩ gives the correct
distribution because |⟨b|C|mps⟩|² sums correctly over the unitary C.

# Arguments
- `state::CAMPSState`: CAMPS state
- `num_samples::Int`: Number of samples

# Returns
- `Vector{Vector{Int}}`: Sampled bitstrings (computational basis outcomes)

# Note
For CAMPS, we sample from the MPS directly. The Clifford C transforms the
measurement basis but doesn't change the distribution of outcomes when
measuring in the computational basis (since C is unitary).
"""
function sample_state(state::CAMPSState; num_samples::Int=1)::Vector{Vector{Int}}
    ensure_initialized!(state)
    return sample_mps_multiple(state.mps, num_samples)
end

"""
    apply_clifford_to_basis_state(C::Destabilizer, bitstring::Vector{Int}) -> Tuple{Vector{Int}, ComplexF64}

Apply the inverse Clifford C† to a computational basis state |b⟩.

For a Clifford C and computational basis state |b⟩, C†|b⟩ is always
another computational basis state (up to a global phase).

This uses the stabilizer formalism:
- |b⟩ is stabilized by Z_j^(1-2b_j) for each j
- After C†, the stabilizers become C† Z_j^(1-2b_j) C
- The resulting state is the unique +1 eigenstate of these operators

# Arguments
- `C::Destabilizer`: Clifford operator
- `bitstring::Vector{Int}`: Input basis state (vector of 0s and 1s)

# Returns
- `Tuple{Vector{Int}, ComplexF64}`: (transformed bitstring, phase)
  The phase is ±1 or ±i (always a Pauli phase).

# Theory
The computational basis state |b⟩ = |b₁b₂...bₙ⟩ is the unique state satisfying:
  (-1)^(bⱼ) Zⱼ |b⟩ = |b⟩  for all j

Applying C†:
  C†|b⟩ is stabilized by C† [(-1)^(bⱼ) Zⱼ] C = (-1)^(bⱼ) (C† Zⱼ C)

The transformed stabilizers determine a unique computational basis state.
"""
function apply_clifford_to_basis_state(C::Destabilizer, bitstring::Vector{Int})::Tuple{Vector{Int}, ComplexF64}
    n = nqubits(C)
    length(bitstring) == n || throw(ArgumentError("Bitstring length must match Clifford size"))

    C_op = CliffordOperator(C)
    C_inv = inv(C_op)

    output_bits = zeros(Int, n)
    total_phase = ComplexF64(1.0)

    for j in 1:n
        Z_j = single_z(n, j)

        if bitstring[j] == 1
            Z_j = PauliOperator(0x02, xbit(Z_j), zbit(Z_j))
        end

        stab = Stabilizer([Z_j])
        apply!(stab, C_inv)
        P_transformed = stab[1]

    end

    output_bits, total_phase = compute_clifford_action_on_basis_state(C, bitstring)

    return (output_bits, total_phase)
end

"""
    compute_clifford_action_on_basis_state(C::Destabilizer, bitstring::Vector{Int}) -> Tuple{Vector{Int}, ComplexF64}

Compute C†|b⟩ for a Clifford C and computational basis state |b⟩.

For CAMPS amplitude computation, we need to compute ⟨b|C|mps⟩ = (⟨b|C)·|mps⟩.
Since C is Clifford and |b⟩ is a computational basis state, C†|b⟩ is another
computational basis state times a Pauli phase.

**Key insight**: Rather than trying to compute C†|b⟩ symbolically (which is
complex for general Cliffords), we use the Clifford matrix representation
for small systems. For larger systems, we note that in CAMPS the amplitude
computation can be done by going back to the MPS directly since the MPS
already has the non-trivial state information.

# Method
For n ≤ 10: Use direct matrix computation via `clifford_to_matrix`
For n > 10: Not currently supported (would need stabilizer-based algorithm)
"""
function compute_clifford_action_on_basis_state(C::Destabilizer, bitstring::Vector{Int})::Tuple{Vector{Int}, ComplexF64}
    n = nqubits(C)

    if n > 12
        throw(ArgumentError("compute_clifford_action_on_basis_state only supported for n ≤ 12 (got n=$n)"))
    end

    C_op = CliffordOperator(C)
    C_mat = clifford_to_matrix(C_op)
    C_inv_mat = C_mat'

    idx_in = sum(bitstring[j] * 2^(j-1) for j in 1:n) + 1

    dim = 2^n
    output_idx = 0
    output_phase = ComplexF64(0.0)

    for k in 1:dim
        amp = C_inv_mat[k, idx_in]
        if abs(amp) > 1e-10
            if output_idx != 0
                throw(ErrorException("Clifford action on basis state gave non-basis state"))
            end
            output_idx = k
            output_phase = amp
        end
    end

    if output_idx == 0
        throw(ErrorException("Clifford action gave zero state"))
    end

    output_bits = [(output_idx - 1) >> j & 1 for j in 0:(n-1)]

    phase_normalized = output_phase
    phase_mag = abs(output_phase)
    if phase_mag > 1e-10
        phase_normalized = output_phase / phase_mag
    end

    return (output_bits, phase_normalized)
end

"""
    pauli_eigenvalue_on_computational_basis(P::PauliOperator, bitstring::Vector{Int}) -> Int

Compute the eigenvalue ⟨b|P|b⟩ for Pauli P on computational basis state |b⟩.

Returns:
- +1 if P|b⟩ = +|b⟩
- -1 if P|b⟩ = -|b⟩
- 0 if P has off-diagonal elements (X or Y) making ⟨b|P|b⟩ = 0

For diagonal Paulis (tensor products of I and Z only):
⟨b|Z₁^{a₁} Z₂^{a₂} ... Zₙ^{aₙ}|b⟩ = ∏ⱼ (-1)^{aⱼ·bⱼ}
"""
function pauli_eigenvalue_on_computational_basis(P::PauliOperator, bitstring::Vector{Int})::Int
    n = nqubits(P)

    global_phase = get_pauli_phase(P)

    for j in 1:n
        x, z = P[j]
        if x
            return 0
        end
    end

    sign_factor = 1
    for j in 1:n
        x, z = P[j]
        if z && bitstring[j] == 1
            sign_factor *= -1
        end
    end

    eigenvalue = real(global_phase) * sign_factor

    return Int(round(eigenvalue))
end

"""
    compute_clifford_phase_on_basis(C::Destabilizer, input::Vector{Int}, output::Vector{Int}) -> ComplexF64

Compute the phase factor in C†|input⟩ = phase × |output⟩.

This computes the Pauli phase accumulated when the Clifford C† maps
the computational basis state |input⟩ to |output⟩.

The phase is always ±1 or ±i (a Pauli phase: i^k for k ∈ {0,1,2,3}).
"""
function compute_clifford_phase_on_basis(C::Destabilizer, input::Vector{Int}, output::Vector{Int})::ComplexF64
    n = nqubits(C)

    if n <= 6
        C_mat = clifford_to_matrix(CliffordOperator(C))
        C_inv_mat = C_mat'

        idx_in = sum(input[j] * 2^(j-1) for j in 1:n) + 1
        idx_out = sum(output[j] * 2^(j-1) for j in 1:n) + 1

        phase = C_inv_mat[idx_out, idx_in]

        return phase
    end

    C_op = CliffordOperator(C)
    C_inv = inv(C_op)

    accumulated_phase = ComplexF64(1.0)

    for j in 1:n
        if input[j] == 1
            X_j = single_x(n, j)
            stab = Stabilizer([X_j])
            apply!(stab, C_inv)
            P_transformed = stab[1]

            accumulated_phase *= get_pauli_phase(P_transformed)
        end
    end

    return accumulated_phase
end

"""
    amplitude(state::CAMPSState, bitstring::Vector{Int}) -> ComplexF64

Compute the amplitude ⟨bitstring|ψ⟩ for a CAMPS state |ψ⟩ = C|mps⟩.

# Theory
For CAMPS state |ψ⟩ = C|mps⟩:
    ⟨b|ψ⟩ = ⟨b|C|mps⟩ = Σ_j C_{bj} ⟨j|mps⟩

where C_{bj} = ⟨b|C|j⟩ is the matrix element of the Clifford unitary.

For small systems (n ≤ 12), we compute this using the Clifford matrix.
For larger systems, a stabilizer-based algorithm would be needed.

# Arguments
- `state::CAMPSState`: CAMPS state |ψ⟩ = C|mps⟩
- `bitstring::Vector{Int}`: Computational basis state (vector of 0s and 1s)

# Returns
- `ComplexF64`: Amplitude ⟨bitstring|ψ⟩

# Example
```julia
state = CAMPSState(3)
initialize!(state)
apply_gate!(state, HGate(1))
apply_gate!(state, TGate(1))

amp = amplitude(state, [0, 0, 0])  # Amplitude of |000⟩
```
"""
function amplitude(state::CAMPSState, bitstring::Vector{Int})::ComplexF64
    ensure_initialized!(state)

    n = state.n_qubits
    length(bitstring) == n || throw(ArgumentError("Bitstring length must match n_qubits"))
    all(b -> b in (0, 1), bitstring) || throw(ArgumentError("Bitstring must contain only 0s and 1s"))

    if n > 12
        throw(ArgumentError("amplitude computation only supported for n ≤ 12 (got n=$n)"))
    end

    C_op = CliffordOperator(state.clifford)
    C_mat = clifford_to_matrix(C_op)

    b_idx = 1
    for k in 1:n
        b_idx += bitstring[k] * (1 << (k - 1))
    end

    dim = 2^n
    result = ComplexF64(0.0)

    for j_idx in 1:dim
        C_bj = C_mat[b_idx, j_idx]

        if abs(C_bj) > 1e-15
            j_bits = [(j_idx - 1) >> (k - 1) & 1 for k in 1:n]

            mps_amp = mps_amplitude(state.mps, j_bits, state.sites)

            result += C_bj * mps_amp
        end
    end

    return result
end

"""
    probability(state::CAMPSState, bitstring::Vector{Int}) -> Float64

Compute the probability |⟨bitstring|ψ⟩|² for a CAMPS state.

# Arguments
- `state::CAMPSState`: CAMPS state
- `bitstring::Vector{Int}`: Computational basis state

# Returns
- `Float64`: Probability (non-negative real number)
"""
function probability(state::CAMPSState, bitstring::Vector{Int})::Float64
    return abs2(amplitude(state, bitstring))
end

#==============================================================================#
# STATE VECTOR EXTRACTION (for small systems)
#==============================================================================#

"""
    state_vector(state::CAMPSState) -> Vector{ComplexF64}

Extract the full state vector from a CAMPS state.

**Warning**: This has exponential complexity O(2^n) and should only be used
for small systems (n ≤ 12 qubits recommended, hard limit at n ≤ 20).

The state vector is returned in computational basis order:
|ψ⟩ = Σᵢ ψᵢ |i⟩ where i is interpreted as a binary number (little-endian).

# Arguments
- `state::CAMPSState`: CAMPS state

# Returns
- `Vector{ComplexF64}`: State vector of length 2^n

# Example
```julia
state = CAMPSState(3)
initialize!(state)
apply_gate!(state, HGate(1))

psi = state_vector(state)
# psi[1] = ⟨000|ψ⟩ = 1/√2
# psi[2] = ⟨001|ψ⟩ = 1/√2 (qubit 1 in superposition)
```
"""
function state_vector(state::CAMPSState)::Vector{ComplexF64}
    ensure_initialized!(state)
    n = state.n_qubits

    if n > 20
        throw(ArgumentError("state_vector only supported for n ≤ 20 qubits (got n=$n)"))
    end

    dim = 2^n
    psi = Vector{ComplexF64}(undef, dim)

    for i in 0:(dim-1)
        bitstring = [(i >> j) & 1 for j in 0:(n-1)]
        psi[i+1] = amplitude(state, bitstring)
    end

    return psi
end

"""
    state_vector_sparse(state::CAMPSState; threshold::Float64=1e-14) -> Dict{Vector{Int}, ComplexF64}

Extract non-zero amplitudes from a CAMPS state as a sparse dictionary.

This is more memory-efficient than `state_vector` when the state has few
non-zero amplitudes in the computational basis.

# Arguments
- `state::CAMPSState`: CAMPS state
- `threshold::Float64`: Amplitudes with |ψᵢ| < threshold are considered zero

# Returns
- `Dict{Vector{Int}, ComplexF64}`: Map from bitstrings to amplitudes

# Example
```julia
state = CAMPSState(4)
initialize!(state)  # |0000⟩
# Only one non-zero amplitude
sparse_psi = state_vector_sparse(state)
# sparse_psi[[0,0,0,0]] ≈ 1.0
```
"""
function state_vector_sparse(state::CAMPSState; threshold::Float64=1e-14)::Dict{Vector{Int}, ComplexF64}
    ensure_initialized!(state)
    n = state.n_qubits

    if n > 20
        throw(ArgumentError("state_vector_sparse only supported for n ≤ 20 qubits (got n=$n)"))
    end

    result = Dict{Vector{Int}, ComplexF64}()
    dim = 2^n

    for i in 0:(dim-1)
        bitstring = [(i >> j) & 1 for j in 0:(n-1)]
        amp = amplitude(state, bitstring)
        if abs(amp) >= threshold
            result[bitstring] = amp
        end
    end

    return result
end

#==============================================================================#
# FIDELITY AND STATE COMPARISON
#==============================================================================#

"""
    fidelity(state1::CAMPSState, state2::CAMPSState) -> Float64

Compute the fidelity |⟨ψ₁|ψ₂⟩|² between two CAMPS states.

The fidelity measures how similar two quantum states are:
- F = 1: identical states (up to global phase)
- F = 0: orthogonal states

**Note**: For small systems (n ≤ 12), this uses state vector extraction.
For larger systems, it uses sampling-based estimation.

# Arguments
- `state1::CAMPSState`: First CAMPS state
- `state2::CAMPSState`: Second CAMPS state

# Returns
- `Float64`: Fidelity in [0, 1]

# Example
```julia
state1 = CAMPSState(3)
initialize!(state1)
apply_gate!(state1, HGate(1))

state2 = CAMPSState(3)
initialize!(state2)
apply_gate!(state2, HGate(1))

F = fidelity(state1, state2)  # Should be 1.0
```
"""
function fidelity(state1::CAMPSState, state2::CAMPSState)::Float64
    ensure_initialized!(state1)
    ensure_initialized!(state2)

    state1.n_qubits == state2.n_qubits ||
        throw(ArgumentError("States must have same number of qubits"))

    n = state1.n_qubits

    if n <= 12
        psi1 = state_vector(state1)
        psi2 = state_vector(state2)
        overlap = dot(psi1, psi2)
        return abs2(overlap)
    end

    return fidelity_sampling(state1, state2)
end

"""
    overlap(state1::CAMPSState, state2::CAMPSState) -> ComplexF64

Compute the overlap ⟨ψ₁|ψ₂⟩ between two CAMPS states.

# Arguments
- `state1::CAMPSState`: First CAMPS state (bra)
- `state2::CAMPSState`: Second CAMPS state (ket)

# Returns
- `ComplexF64`: Overlap ⟨ψ₁|ψ₂⟩

# Note
For small systems (n ≤ 12), this uses exact state vector computation.
For larger systems, a sampling-based approximation is used.
"""
function overlap(state1::CAMPSState, state2::CAMPSState)::ComplexF64
    ensure_initialized!(state1)
    ensure_initialized!(state2)

    state1.n_qubits == state2.n_qubits ||
        throw(ArgumentError("States must have same number of qubits"))

    n = state1.n_qubits

    if n <= 12
        psi1 = state_vector(state1)
        psi2 = state_vector(state2)
        return dot(psi1, psi2)
    end

    F = fidelity_sampling(state1, state2)
    return ComplexF64(sqrt(F))
end

"""
    fidelity_sampling(state1::CAMPSState, state2::CAMPSState;
                      num_samples::Int=10000) -> Float64

Estimate fidelity using importance sampling.

Uses the SWAP test idea: sample from one state and evaluate probability
on the other state.

# Arguments
- `state1::CAMPSState`: First state
- `state2::CAMPSState`: Second state
- `num_samples::Int`: Number of samples for estimation

# Returns
- `Float64`: Estimated fidelity
"""
function fidelity_sampling(state1::CAMPSState, state2::CAMPSState;
                            num_samples::Int=10000)::Float64
    n = state1.n_qubits

    samples1 = sample_state(state1; num_samples=num_samples)

    total = 0.0
    for sample in samples1
        p2 = probability(state2, sample)
        total += sqrt(p2)
    end

    estimate = (total / num_samples)^2

    return clamp(estimate, 0.0, 1.0)
end

"""
    fidelity_with_target(state::CAMPSState, target_vector::Vector{ComplexF64}) -> Float64

Compute fidelity between a CAMPS state and an explicit target state vector.

# Arguments
- `state::CAMPSState`: CAMPS state
- `target_vector::Vector{ComplexF64}`: Target state vector (length 2^n)

# Returns
- `Float64`: Fidelity |⟨target|ψ⟩|²
"""
function fidelity_with_target(state::CAMPSState, target_vector::Vector{ComplexF64})::Float64
    ensure_initialized!(state)
    n = state.n_qubits

    expected_dim = 2^n
    length(target_vector) == expected_dim ||
        throw(ArgumentError("Target vector must have length 2^n = $expected_dim"))

    overlap_val = ComplexF64(0.0)

    for i in 0:(expected_dim-1)
        bitstring = [(i >> j) & 1 for j in 0:(n-1)]
        amp = amplitude(state, bitstring)
        overlap_val += conj(target_vector[i+1]) * amp
    end

    return abs2(overlap_val)
end

"""
    norm(state::CAMPSState) -> Float64

Compute the norm √⟨ψ|ψ⟩ of a CAMPS state.

For a properly normalized CAMPS state, this should return 1.0.
Deviations indicate numerical errors in the simulation.

# Arguments
- `state::CAMPSState`: CAMPS state

# Returns
- `Float64`: Norm of the state
"""
function LinearAlgebra.norm(state::CAMPSState)::Float64
    ensure_initialized!(state)
    return get_mps_norm(state.mps)
end

"""
    normalize!(state::CAMPSState) -> CAMPSState

Normalize a CAMPS state in-place.

# Arguments
- `state::CAMPSState`: CAMPS state (modified)

# Returns
- `CAMPSState`: Normalized state
"""
function normalize!(state::CAMPSState)::CAMPSState
    ensure_initialized!(state)
    normalize_mps!(state.mps)
    return state
end

#==============================================================================#
# STANDARD CIRCUIT GENERATORS
#==============================================================================#

"""
    qft_circuit(n::Int) -> Vector{Gate}

Generate a Quantum Fourier Transform circuit on n qubits.

The QFT transforms computational basis states to Fourier basis states:
QFT|j⟩ = (1/√N) Σₖ exp(2πijk/N) |k⟩

# Arguments
- `n::Int`: Number of qubits

# Returns
- `Vector{Gate}`: QFT circuit

# Note
This generates the standard QFT circuit with Hadamards and controlled phase
rotations. The controlled phase gates use non-Clifford angles (except CZ).
"""
function qft_circuit(n::Int)::Vector{Gate}
    n > 0 || throw(ArgumentError("Number of qubits must be positive"))

    circuit = Gate[]

    for j in 1:n
        push!(circuit, HGate(j))

        for k in (j+1):n
            m = k - j + 1
            angle = 2π / 2^m

            if m == 1
                push!(circuit, CZGate(j, k))
            else
                push!(circuit, RzGate(k, angle/2))
                push!(circuit, CNOTGate(j, k))
                push!(circuit, RzGate(k, -angle/2))
                push!(circuit, CNOTGate(j, k))
                push!(circuit, RzGate(j, angle/2))
            end
        end
    end

    for j in 1:(n÷2)
        push!(circuit, SWAPGate(j, n-j+1))
    end

    return circuit
end

"""
    inverse_qft_circuit(n::Int) -> Vector{Gate}

Generate an inverse Quantum Fourier Transform circuit.

QFT† transforms Fourier basis back to computational basis.

# Arguments
- `n::Int`: Number of qubits

# Returns
- `Vector{Gate}`: Inverse QFT circuit
"""
function inverse_qft_circuit(n::Int)::Vector{Gate}
    qft = qft_circuit(n)

    inv_circuit = Gate[]
    for gate in reverse(qft)
        if gate isa RotationGate
            push!(inv_circuit, RotationGate(gate.qubit, gate.axis, -gate.angle))
        else
            push!(inv_circuit, gate)
        end
    end

    return inv_circuit
end

"""
    ghz_circuit(n::Int) -> Vector{Gate}

Generate a circuit to create an n-qubit GHZ state.

GHZ state: |GHZ⟩ = (|00...0⟩ + |11...1⟩) / √2

# Arguments
- `n::Int`: Number of qubits

# Returns
- `Vector{Gate}`: GHZ preparation circuit (Clifford-only)
"""
function ghz_circuit(n::Int)::Vector{Gate}
    n > 0 || throw(ArgumentError("Number of qubits must be positive"))

    circuit = Gate[]

    push!(circuit, HGate(1))

    for j in 1:(n-1)
        push!(circuit, CNOTGate(j, j+1))
    end

    return circuit
end

"""
    w_state_circuit(n::Int) -> Vector{Gate}

Generate a circuit to create an n-qubit W state.

W state: |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |00...01⟩) / √n

# Arguments
- `n::Int`: Number of qubits

# Returns
- `Vector{Gate}`: W state preparation circuit

# Note
The W state preparation requires non-Clifford rotations for n > 2.
"""
function w_state_circuit(n::Int)::Vector{Gate}
    n > 0 || throw(ArgumentError("Number of qubits must be positive"))

    if n == 1
        return [XGate(1)]
    end

    circuit = Gate[]

    push!(circuit, XGate(1))

    for k in 1:(n-1)
        angle = 2 * acos(sqrt((n-k)/(n-k+1)))

        if abs(angle - π/2) < 1e-10
            push!(circuit, CNOTGate(k, k+1))
            push!(circuit, HGate(k+1))
            push!(circuit, CNOTGate(k, k+1))
        else
            push!(circuit, RyGate(k+1, angle/2))
            push!(circuit, CNOTGate(k, k+1))
            push!(circuit, RyGate(k+1, -angle/2))
            push!(circuit, CNOTGate(k, k+1))
        end
    end

    return circuit
end

"""
    random_t_depth_circuit(n::Int, t_depth::Int; seed::Union{Int,Nothing}=nothing) -> Vector{Gate}

Generate a random circuit with specified T-gate depth.

T-depth is the minimum number of T-gate layers in the circuit.
This is useful for benchmarking CAMPS against T-count/T-depth metrics.

# Arguments
- `n::Int`: Number of qubits
- `t_depth::Int`: Number of T-gate layers
- `seed::Union{Int,Nothing}`: Random seed

# Returns
- `Vector{Gate}`: Random circuit with specified T-depth
"""
function random_t_depth_circuit(n::Int, t_depth::Int; seed::Union{Int,Nothing}=nothing)::Vector{Gate}
    n > 0 || throw(ArgumentError("Number of qubits must be positive"))
    t_depth >= 0 || throw(ArgumentError("T-depth must be non-negative"))

    if seed !== nothing
        Random.seed!(seed)
    end

    circuit = Gate[]

    for q in 1:n
        if rand() < 0.5
            push!(circuit, HGate(q))
        end
        if rand() < 0.3
            push!(circuit, SGate(q))
        end
    end

    for layer in 1:t_depth
        t_qubits = randperm(n)[1:max(1, rand(1:n))]
        for q in t_qubits
            push!(circuit, TGate(q))
        end

        for q in 1:n
            r = rand()
            if r < 0.25
                push!(circuit, HGate(q))
            elseif r < 0.4
                push!(circuit, SGate(q))
            end
        end

        for i in 1:2:(n-1)
            if rand() < 0.5
                push!(circuit, CNOTGate(i, i+1))
            end
        end
    end

    return circuit
end

"""
    hardware_efficient_ansatz(n::Int, layers::Int; rotation_angles::Union{Vector{Float64},Nothing}=nothing) -> Vector{Gate}

Generate a hardware-efficient variational ansatz circuit.

This is commonly used in VQE and other variational algorithms.
Structure: [Ry-Rz layer] - [CNOT layer] - [Ry-Rz layer] - ...

# Arguments
- `n::Int`: Number of qubits
- `layers::Int`: Number of ansatz layers
- `rotation_angles::Union{Vector{Float64},Nothing}`: Rotation angles (random if nothing)

# Returns
- `Vector{Gate}`: Hardware-efficient ansatz circuit
"""
function hardware_efficient_ansatz(n::Int, layers::Int;
                                    rotation_angles::Union{Vector{Float64},Nothing}=nothing)::Vector{Gate}
    n > 0 || throw(ArgumentError("Number of qubits must be positive"))
    layers >= 0 || throw(ArgumentError("Number of layers must be non-negative"))

    total_angles = 2 * n * layers

    if rotation_angles === nothing
        rotation_angles = 2π * rand(total_angles)
    else
        length(rotation_angles) == total_angles ||
            throw(ArgumentError("Expected $total_angles rotation angles, got $(length(rotation_angles))"))
    end

    circuit = Gate[]
    angle_idx = 1

    for layer in 1:layers
        for q in 1:n
            push!(circuit, RyGate(q, rotation_angles[angle_idx]))
            angle_idx += 1
            push!(circuit, RzGate(q, rotation_angles[angle_idx]))
            angle_idx += 1
        end

        for q in 1:(n-1)
            push!(circuit, CNOTGate(q, q+1))
        end
    end

    return circuit
end

"""
    expectation_value(state::CAMPSState, P::PauliOperator) -> ComplexF64

Compute the expectation value ⟨ψ|P|ψ⟩ for a Pauli operator.

# Arguments
- `state::CAMPSState`: CAMPS state
- `P::PauliOperator`: Pauli operator

# Returns
- `ComplexF64`: Expectation value

# Theory
⟨ψ|P|ψ⟩ = ⟨mps|C†PC|mps⟩ = ⟨mps|P̃|mps⟩

where P̃ = C†PC is the twisted Pauli.
"""
function expectation_value(state::CAMPSState, P::PauliOperator)::ComplexF64
    ensure_initialized!(state)

    P_twisted = commute_pauli_through_clifford(P, state.clifford)

    mps_copy = copy(state.mps)
    apply_pauli_string!(mps_copy, P_twisted, state.sites)

    return mps_overlap(state.mps, mps_copy)
end

"""
    expectation_value_z(state::CAMPSState, qubit::Int) -> Float64

Compute ⟨Z_qubit⟩ efficiently.

# Arguments
- `state::CAMPSState`: CAMPS state
- `qubit::Int`: Target qubit

# Returns
- `Float64`: Expectation value (real for Z)
"""
function expectation_value_z(state::CAMPSState, qubit::Int)::Float64
    P = single_z(state.n_qubits, qubit)
    return real(expectation_value(state, P))
end

"""
    expectation_value_x(state::CAMPSState, qubit::Int) -> Float64

Compute ⟨X_qubit⟩ efficiently.

# Arguments
- `state::CAMPSState`: CAMPS state
- `qubit::Int`: Target qubit

# Returns
- `Float64`: Expectation value (real for X)
"""
function expectation_value_x(state::CAMPSState, qubit::Int)::Float64
    P = single_x(state.n_qubits, qubit)
    return real(expectation_value(state, P))
end

"""
    expectation_value_y(state::CAMPSState, qubit::Int) -> Float64

Compute ⟨Y_qubit⟩ efficiently.

# Arguments
- `state::CAMPSState`: CAMPS state
- `qubit::Int`: Target qubit

# Returns
- `Float64`: Expectation value (real for Y)
"""
function expectation_value_y(state::CAMPSState, qubit::Int)::Float64
    P = single_y(state.n_qubits, qubit)
    return real(expectation_value(state, P))
end

#==============================================================================#
# RANDOM CIRCUIT GENERATION
#==============================================================================#

"""
    random_clifford_t_circuit(n_qubits::Int, n_t_gates::Int;
                               seed::Union{Int, Nothing}=nothing,
                               include_measurements::Bool=false) -> Vector{Gate}

Generate a random Clifford+T circuit.

# Arguments
- `n_qubits::Int`: Number of qubits
- `n_t_gates::Int`: Number of T gates to include
- `seed::Union{Int, Nothing}`: Random seed for reproducibility
- `include_measurements::Bool`: Whether to end with computational basis measurement

# Returns
- `Vector{Gate}`: Random circuit

# Structure
The circuit has the following structure:
1. Random single-qubit Cliffords on all qubits
2. Random CNOT/CZ layer
3. T gates distributed randomly
4. More random Cliffords
5. (Optional) Computational basis measurements
"""
function random_clifford_t_circuit(n_qubits::Int, n_t_gates::Int;
                                    seed::Union{Int, Nothing}=nothing,
                                    include_measurements::Bool=false)::Vector{Gate}
    if seed !== nothing
        Random.seed!(seed)
    end

    circuit = Gate[]

    single_qubit_cliffords = [HGate, SGate, q -> CliffordGate([(:Z, q), (:Z, q)], [q])]
    for q in 1:n_qubits
        gate_type = rand(single_qubit_cliffords)
        push!(circuit, gate_type(q))
    end

    n_entangling = n_qubits ÷ 2
    for _ in 1:n_entangling
        q1, q2 = randperm(n_qubits)[1:2]
        push!(circuit, CNOTGate(q1, q2))
    end

    for q in 1:n_qubits
        gate_type = rand(single_qubit_cliffords)
        push!(circuit, gate_type(q))
    end

    t_gate_qubits = rand(1:n_qubits, n_t_gates)
    for q in t_gate_qubits
        push!(circuit, TGate(q))
    end

    for q in 1:n_qubits
        gate_type = rand(single_qubit_cliffords)
        push!(circuit, gate_type(q))
    end

    for _ in 1:n_entangling
        q1, q2 = randperm(n_qubits)[1:2]
        push!(circuit, CNOTGate(q1, q2))
    end

    return circuit
end

"""
    random_brickwork_circuit(n_qubits::Int, n_layers::Int, t_probability::Float64;
                              seed::Union{Int, Nothing}=nothing) -> Vector{Gate}

Generate a random brickwork circuit with T gates at specified probability.

# Arguments
- `n_qubits::Int`: Number of qubits
- `n_layers::Int`: Number of brickwork layers
- `t_probability::Float64`: Probability of T gate after each single-qubit Clifford
- `seed::Union{Int, Nothing}`: Random seed

# Returns
- `Vector{Gate}`: Random brickwork circuit

# Structure
Each layer consists of:
1. Single-qubit gates (Clifford + optional T)
2. Two-qubit gates (CNOT or CZ)
"""
function random_brickwork_circuit(n_qubits::Int, n_layers::Int, t_probability::Float64;
                                   seed::Union{Int, Nothing}=nothing)::Vector{Gate}
    if seed !== nothing
        Random.seed!(seed)
    end

    circuit = Gate[]

    for layer in 1:n_layers
        for q in 1:n_qubits
            r = rand()
            if r < 0.33
                push!(circuit, HGate(q))
            elseif r < 0.67
                push!(circuit, SGate(q))
            end

            if rand() < t_probability
                push!(circuit, TGate(q))
            end
        end

        offset = (layer - 1) % 2
        for i in (1 + offset):2:(n_qubits - 1)
            if rand() < 0.5
                push!(circuit, CNOTGate(i, i + 1))
            else
                push!(circuit, CZGate(i, i + 1))
            end
        end
    end

    return circuit
end

#==============================================================================#
# CIRCUIT ANALYSIS
#==============================================================================#

"""
    analyze_circuit(circuit::Vector{<:Gate}, n_qubits::Int) -> NamedTuple

Analyze a circuit without simulating it.

# Arguments
- `circuit::Vector{<:Gate}`: Circuit to analyze
- `n_qubits::Int`: Number of qubits

# Returns
- `NamedTuple` with fields:
  - `n_gates::Int`: Total gates
  - `n_clifford::Int`: Clifford gates
  - `n_t_gates::Int`: T gates
  - `n_other_rotation::Int`: Other rotation gates
  - `t_count::Int`: Total T-gate count
  - `depth::Int`: Circuit depth
  - `t_depth::Int`: T-gate depth (layers containing T gates)
"""
function analyze_circuit(circuit::Vector{<:Gate}, n_qubits::Int)
    n_clifford = 0
    n_t = 0
    n_other = 0

    for gate in circuit
        if gate isa CliffordGate
            n_clifford += 1
        elseif gate isa RotationGate
            if is_clifford_angle(gate.angle)
                n_clifford += 1
            elseif abs(gate.angle) ≈ π/4 && gate.axis == :Z
                n_t += 1
            else
                n_other += 1
            end
        end
    end

    d = gate_depth(circuit, n_qubits)

    return (
        n_gates = length(circuit),
        n_clifford = n_clifford,
        n_t_gates = n_t,
        n_other_rotation = n_other,
        t_count = n_t,
        depth = d,
        t_depth = 0
    )
end

"""
    predict_bond_dimension_for_circuit(circuit::Vector{<:Gate}, n_qubits::Int;
                                        strategy::DisentanglingStrategy=OFDStrategy())
        -> NamedTuple

Predict the bond dimension that would result from simulating a circuit.

Uses the GF(2) theory to predict bond dimension without full simulation.

# Arguments
- `circuit::Vector{<:Gate}`: Circuit
- `n_qubits::Int`: Number of qubits
- `strategy::DisentanglingStrategy`: Disentangling strategy

# Returns
- `NamedTuple` with fields:
  - `predicted_chi::Int`: Predicted maximum bond dimension
  - `n_t_gates::Int`: Number of T gates
  - `gf2_rank::Int`: Rank of the GF(2) matrix
  - `nullity::Int`: Nullity (t - rank)
"""
function predict_bond_dimension_for_circuit(circuit::Vector{<:Gate}, n_qubits::Int;
                                             strategy::DisentanglingStrategy=OFDStrategy())
    state = CAMPSState(n_qubits)
    initialize!(state)

    t_gate_twisted_paulis = PauliOperator[]

    for gate in circuit
        if gate isa CliffordGate
            apply_clifford_gate_to_state!(state, gate)
        elseif gate isa RotationGate
            if !is_clifford_angle(gate.angle)
                P_twisted = compute_twisted_pauli(state, gate.axis, gate.qubit)
                push!(t_gate_twisted_paulis, P_twisted)

                if strategy isa OFDStrategy || strategy isa HybridStrategy
                    control = find_disentangling_qubit(P_twisted, state.free_qubits)
                    if control !== nothing
                        D_gates = build_disentangler_gates(P_twisted, control)
                        D_flat = flatten_gate_sequence(D_gates)
                        apply_inverse_gates!(state.clifford, D_flat)
                        state.free_qubits[control] = false
                    end
                end
            end
        end
    end

    if isempty(t_gate_twisted_paulis)
        return (predicted_chi=1, n_t_gates=0, gf2_rank=0, nullity=0)
    end

    analysis = analyze_gf2_structure(t_gate_twisted_paulis)

    return (
        predicted_chi = analysis.predicted_chi,
        n_t_gates = analysis.t,
        gf2_rank = analysis.rank,
        nullity = analysis.nullity
    )
end

#==============================================================================#
# DISPLAY UTILITIES
#==============================================================================#

function Base.show(io::IO, result::SimulationResult)
    print(io, "SimulationResult(")
    print(io, "n=$(result.n_qubits), ")
    print(io, "gates=$(result.n_gates), ")
    print(io, "χ=$(result.final_bond_dim), ")
    print(io, "χ_pred=$(result.predicted_bond_dim))")
end

function Base.show(io::IO, ::MIME"text/plain", result::SimulationResult)
    println(io, "SimulationResult")
    println(io, "  Qubits: $(result.n_qubits)")
    println(io, "  Total gates: $(result.n_gates)")
    println(io, "    Clifford: $(result.n_clifford)")
    println(io, "    Non-Clifford: $(result.n_non_clifford)")
    println(io, "  Disentangling:")
    println(io, "    OFD applied: $(result.n_ofd_applied)")
    println(io, "    OBD applied: $(result.n_obd_applied)")
    println(io, "  Bond dimension:")
    println(io, "    Final: $(result.final_bond_dim)")
    println(io, "    GF(2) predicted: $(result.predicted_bond_dim)")
    print(io, "  Max entanglement entropy: $(round(result.final_entropy, digits=4))")
end

function Base.show(io::IO, result::OFDSResult)
    print(io, "OFDSResult(applied=$(result.num_applied), failed=$(result.num_failed), χ=$(result.actual_chi))")
end

function Base.show(io::IO, result::OBDResult)
    print(io, "OBDResult(sweeps=$(result.num_sweeps), converged=$(result.converged), ")
    print(io, "S: $(round(result.initial_max_entropy, digits=3)) → $(round(result.final_max_entropy, digits=3)))")
end
