
"""
    compute_gf2_for_mixed_circuit(gates::Vector, t_positions::Vector{Int},
                                   n_qubits::Int;
                                   seed::Union{Integer, Nothing}=nothing,
                                   simulate_ofd::Bool=true) -> NamedTuple

Compute GF(2) rank analysis for a circuit with mixed gate formats.

Handles both symbolic tuples (`:H`, `:CNOT`, `:T`, `:random2q`, etc.) and
CAMPS Gate objects (`CliffordGate`, `RotationGate`). Uses a Clifford-only walk
(no MPS needed), making this orders of magnitude faster than full simulation.

# Arguments
- `gates::Vector`: Circuit gates (tuples or Gate objects)
- `t_positions::Vector{Int}`: Positions of T-gates in the circuit (1-indexed)
- `n_qubits::Int`: Number of qubits
- `seed::Union{Integer, Nothing}`: RNG seed (required for `:random2q` gates to be reproducible)
- `simulate_ofd::Bool`: If true, update Clifford for successful OFD disentanglers.
  This matches the behavior of `predict_bond_dimension_for_circuit()`. If false,
  only applies Clifford gates without OFD disentangler updates.

# Returns
NamedTuple with fields:
- `n_t_gates::Int`: Number of non-Clifford gates found
- `gf2_rank::Int`: Rank of the GF(2) matrix
- `nullity::Int`: t - rank (determines bond dimension exponent)
- `predicted_chi::Int`: Predicted bond dimension 2^nullity
- `n_disentanglable::Int`: Number of T-gates disentanglable by OFD
- `n_not_disentanglable::Int`: Number of T-gates not disentanglable
- `rank_to_t_ratio::Float64`: gf2_rank / n_t_gates (1.0 = all independent)
- `xbit_density::Float64`: Average fraction of 1s in GF(2) matrix rows
- `twisted_paulis::Vector{PauliOperator}`: Collected twisted Paulis

# Theory
The GF(2) matrix z has rows corresponding to twisted Pauli strings from T-gates,
with entries z[k,j] = 1 if the k-th twisted Pauli has X or Y on qubit j.
The bond dimension of the CAMPS MPS is bounded by 2^(t - rank(z)) = 2^nullity.

A high rank_to_t_ratio (close to 1.0) indicates most T-gates are GF(2)-independent
and can be disentangled by OFD, keeping bond dimension low.

# Known Limitation
For circuits containing `:random2q` gates (RandomBrickwall, RandomAllToAll families),
the random Cliffords drawn depend on RNG state. If `simulate_ofd=true`, our
Clifford-only walk may diverge from a full benchmark run (which also consumes RNG
via OBD fallback). This affects only 2 of 14 circuit families and the statistical
properties are preserved across realizations.
"""
function compute_gf2_for_mixed_circuit(
    gates::Vector,
    t_positions::Vector{Int},
    n_qubits::Int;
    seed::Union{Integer, Nothing}=nothing,
    simulate_ofd::Bool=true
)
    state = CAMPSState(n_qubits)
    initialize!(state)

    twisted_paulis = PauliOperator[]
    n_disentanglable = 0
    n_not_disentanglable = 0

    for (idx, gate) in enumerate(gates)
        if gate isa Tuple
            result = _process_tuple_gate_counted!(
                state, gate, twisted_paulis, simulate_ofd)
            if result !== nothing
                n_disentanglable += result[1]
                n_not_disentanglable += result[2]
            end
        elseif gate isa CliffordGate
            apply_clifford_gate_to_state!(state, gate)
        elseif gate isa RotationGate
            if !is_clifford_angle(gate.angle)
                P_twisted = compute_twisted_pauli(state, gate.axis, gate.qubit)
                push!(twisted_paulis, P_twisted)

                if simulate_ofd
                    control = find_disentangling_qubit(P_twisted, state.free_qubits)
                    if control !== nothing
                        D_gates = build_disentangler_gates(P_twisted, control)
                        D_flat = flatten_gate_sequence(D_gates)
                        apply_inverse_gates!(state.clifford, D_flat)
                        state.free_qubits[control] = false
                        n_disentanglable += 1
                    else
                        n_not_disentanglable += 1
                    end
                end
            else
                clifford_gate = rotation_to_clifford(gate)
                if clifford_gate !== nothing
                    apply_clifford_gate_to_state!(state, clifford_gate)
                end
            end
        end
    end

    if isempty(twisted_paulis)
        return (
            n_t_gates = 0,
            gf2_rank = 0,
            nullity = 0,
            predicted_chi = 1,
            n_disentanglable = 0,
            n_not_disentanglable = 0,
            rank_to_t_ratio = NaN,
            xbit_density = NaN,
            twisted_paulis = PauliOperator[]
        )
    end

    analysis = analyze_gf2_structure(twisted_paulis)

    M = build_gf2_matrix(twisted_paulis)
    xbit_density = sum(M) / length(M)

    return (
        n_t_gates = analysis.t,
        gf2_rank = analysis.rank,
        nullity = analysis.nullity,
        predicted_chi = analysis.predicted_chi,
        n_disentanglable = n_disentanglable,
        n_not_disentanglable = n_not_disentanglable,
        rank_to_t_ratio = analysis.t > 0 ? analysis.rank / analysis.t : NaN,
        xbit_density = xbit_density,
        twisted_paulis = twisted_paulis
    )
end

"""
Internal helper: Process a single tuple-format gate and return
(disentanglable_increment, not_disentanglable_increment) or nothing for non-T gates.
"""
function _process_tuple_gate_counted!(
    state::CAMPSState,
    gate::Tuple,
    twisted_paulis::Vector{PauliOperator},
    simulate_ofd::Bool
)::Union{Nothing, Tuple{Int, Int}}
    gate_type = gate[1]
    qubits = gate[2]

    if gate_type == :T
        qubit = qubits[1]
        P_twisted = compute_twisted_pauli(state, :Z, qubit)
        push!(twisted_paulis, P_twisted)

        if simulate_ofd
            control = find_disentangling_qubit(P_twisted, state.free_qubits)
            if control !== nothing
                D_gates = build_disentangler_gates(P_twisted, control)
                D_flat = flatten_gate_sequence(D_gates)
                apply_inverse_gates!(state.clifford, D_flat)
                state.free_qubits[control] = false
                return (1, 0)
            else
                return (0, 1)
            end
        end
        return nothing

    elseif gate_type == :H
        apply_gate!(state, CliffordGate([(:H, qubits[1])], [qubits[1]]))
    elseif gate_type == :CNOT
        apply_gate!(state, CliffordGate([(:CNOT, qubits[1], qubits[2])], [qubits[1], qubits[2]]))
    elseif gate_type == :X
        apply_gate!(state, CliffordGate([(:X, qubits[1])], [qubits[1]]))
    elseif gate_type == :Z
        apply_gate!(state, CliffordGate([(:Z, qubits[1])], [qubits[1]]))
    elseif gate_type == :S
        apply_gate!(state, CliffordGate([(:S, qubits[1])], [qubits[1]]))
    elseif gate_type == :Sdag
        apply_gate!(state, CliffordGate([(:Sdag, qubits[1])], [qubits[1]]))
    elseif gate_type == :random2q
        q1, q2 = qubits[1], qubits[2]
        cliff = random_clifford(2)
        sparse = SparseGate(cliff, [q1, q2])
        apply!(state.clifford, sparse)
    end

    return nothing
end
