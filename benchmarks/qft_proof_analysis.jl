QFT rank = N-1 Proof Support Analysis
=======================================

Two analyses requested by PI to support the analytic proof:

Part 1: At N=8, for each of the 7 pivot rows in the RREF, identify which
         QFT iteration j (which qubit's H-block) produced the FIRST x-bit
         row that established that pivot. Confirm pivot in column j comes
         from iteration j's Rz gates.

Part 2: At N=6, print the x-bit portion of the Clifford tableau's
         representation of Z_N (i.e., the twisted Pauli that Z_N maps to
         under C†·Z_N·C) after EVERY gate in the circuit. Verify that
         qubit N never acquires an X/Y component until the final H on qubit N.

         Specifically: the claim is that CNOT(N, target) with qubit N as
         control does NOT give Z_N an X component. This is verified by tracking
         what C†·Z_N·C looks like after each gate.

Usage:
    julia benchmarks/qft_proof_analysis.jl
=

camps_dir = dirname(dirname(@__FILE__))
println("Activating CAMPS.jl environment: $camps_dir")
using Pkg
Pkg.activate(camps_dir)

using CAMPS
using QuantumClifford
using Random
using Printf

"""
    track_pivot_origins(n_qubits::Int) -> NamedTuple

For each non-Clifford rotation in qft_circuit(N), compute the twisted Pauli
x-bit row and try to insert it into an online RREF. When insertion succeeds
(new independent row), record which QFT iteration j produced it and which
column it pivots on.

The QFT circuit has a clear block structure:
  Iteration j=1: H(1), then controlled-R_m with qubits 2..N
  Iteration j=2: H(2), then controlled-R_m with qubits 3..N
  ...
  Iteration j=N: H(N)  (no controlled rotations)

Tracks which iteration j each gate belongs to.
"""
function track_pivot_origins(n_qubits::Int)
    circuit = qft_circuit(n_qubits)

    state = CAMPSState(n_qubits)
    initialize!(state)

    current_j = 0

    rref_matrix = zeros(Bool, n_qubits, n_qubits)
    pivot_col = zeros(Int, n_qubits)
    rank = 0

    pivot_origins = NamedTuple[]
    t_count = 0
    all_rows = NamedTuple[]

    for gate in circuit
        if gate isa CliffordGate
            for spec in gate.gates
                if spec isa Tuple && spec[1] == :H
                    current_j = spec[2]
                end
            end
            apply_clifford_gate_to_state!(state, gate)

        elseif gate isa RotationGate
            if is_clifford_angle(gate.angle)
                clifford_gate = rotation_to_clifford(gate)
                if clifford_gate !== nothing
                    apply_clifford_gate_to_state!(state, clifford_gate)
                end
            else
                t_count += 1

                P_twisted = compute_twisted_pauli(state, gate.axis, gate.qubit)
                xb = BitVector(xbit(P_twisted))

                push!(all_rows, (
                    t_index = t_count,
                    iteration_j = current_j,
                    qubit = gate.qubit,
                    angle = gate.angle,
                    xbit = copy(xb)
                ))

                r = copy(xb)
                for i in 1:rank
                    pc = pivot_col[i]
                    if r[pc]
                        for col in 1:n_qubits
                            r[col] = r[col] ⊻ rref_matrix[i, col]
                        end
                    end
                end

                new_pivot = findfirst(r)
                if new_pivot !== nothing
                    rank += 1
                    for col in 1:n_qubits
                        rref_matrix[rank, col] = r[col]
                    end
                    pivot_col[rank] = new_pivot

                    push!(pivot_origins, (
                        pivot_column = new_pivot,
                        iteration_j = current_j,
                        t_index = t_count,
                        qubit = gate.qubit,
                        angle = gate.angle,
                        xbit_row = copy(xb),
                        reduced_row = copy(r)
                    ))
                end

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

    return (
        n_qubits = n_qubits,
        rank = rank,
        t_count = t_count,
        pivot_origins = pivot_origins,
        all_rows = all_rows,
        pivot_cols = pivot_col[1:rank]
    )
end

"""
    track_zN_evolution(n_qubits::Int) -> Vector{NamedTuple}

After each gate in qft_circuit(N), compute the twisted Pauli P̃ = C†·Z_N·C
and record its x-bit vector. The claim is that x-bit of P̃ is all-zeros
(meaning Z_N stays in the Z-subspace) until the final H on qubit N.

This directly tests whether CNOT(N, target) modifies the X-component of Z_N.

Note: OFD is NOT applied here — the raw Clifford evolution of Z_N is tracked.
OFD modifies the Clifford tableau, which would change the tracking. For the
proof, the behavior of Z_N under the *circuit* Cliffords alone is needed,
without disentangler modifications.

But note — the actual GF(2) analysis DOES apply OFD (it modifies the Clifford
state). So BOTH views are needed:
  (a) Raw circuit evolution (no OFD) — for understanding the proof argument
  (b) With OFD applied — matching the actual GF(2) computation

We'll run both and compare.
"""
function track_zN_evolution(n_qubits::Int; apply_ofd::Bool=false)
    circuit = qft_circuit(n_qubits)
    N = n_qubits

    state = CAMPSState(n_qubits)
    initialize!(state)

    Z_N = axis_to_pauli(:Z, N, N)

    snapshots = NamedTuple[]
    gate_idx = 0
    current_j = 0

    for gate in circuit
        gate_idx += 1

        gate_label = ""
        if gate isa CliffordGate
            for spec in gate.gates
                if spec isa Tuple
                    if spec[1] == :H
                        current_j = spec[2]
                        gate_label = "H($(spec[2]))"
                    elseif spec[1] == :CNOT
                        gate_label = "CNOT($(spec[2]),$(spec[3]))"
                    elseif spec[1] == :S
                        gate_label = "S($(spec[2]))"
                    elseif spec[1] == :Sdag
                        gate_label = "S†($(spec[2]))"
                    else
                        gate_label = "$(spec[1])($(spec[2:end]...))"
                    end
                end
            end
            apply_clifford_gate_to_state!(state, gate)

        elseif gate isa RotationGate
            is_cliff = is_clifford_angle(gate.angle)
            if is_cliff
                gate_label = "Rz_cliff(q=$(gate.qubit), θ=$(round(gate.angle, digits=4)))"
                clifford_gate = rotation_to_clifford(gate)
                if clifford_gate !== nothing
                    apply_clifford_gate_to_state!(state, clifford_gate)
                end
            else
                gate_label = "Rz(q=$(gate.qubit), θ=$(round(gate.angle, digits=4)))"
                if apply_ofd
                    P_twisted = compute_twisted_pauli(state, gate.axis, gate.qubit)
                    control = find_disentangling_qubit(P_twisted, state.free_qubits)
                    if control !== nothing
                        D_gates = build_disentangler_gates(P_twisted, control)
                        D_flat = flatten_gate_sequence(D_gates)
                        apply_inverse_gates!(state.clifford, D_flat)
                        state.free_qubits[control] = false
                        gate_label *= " [OFD→q$control]"
                    end
                end
            end

        elseif gate isa SWAPGate
            gate_label = "SWAP($(gate.qubit1),$(gate.qubit2))"
            apply_clifford_gate_to_state!(state, gate)

        elseif gate isa CZGate
            gate_label = "CZ($(gate.control),$(gate.target))"
            apply_clifford_gate_to_state!(state, gate)
        else
            gate_label = "other"
            apply_clifford_gate_to_state!(state, gate)
        end

        P_twisted_ZN = commute_pauli_through_clifford(Z_N, state.clifford)
        xb = BitVector(xbit(P_twisted_ZN))
        zb = BitVector(zbit(P_twisted_ZN))

        push!(snapshots, (
            gate_idx = gate_idx,
            iteration_j = current_j,
            gate_label = gate_label,
            xbit_ZN = copy(xb),
            zbit_ZN = copy(zb),
            xN_nonzero = any(xb),
            pauli_str = pauli_to_string(P_twisted_ZN)
        ))
    end

    return snapshots
end

function main()
    println("="^90)
    println("PART 1: PIVOT ORIGIN TRACKING — N=8")
    println("="^90)
    println()
    println("For each pivot column in the RREF, which QFT iteration j produced")
    println("the first linearly independent row that established that pivot?")
    println()

    result = track_pivot_origins(8)

    println("-"^90)
    @printf("%8s %12s %8s %8s %12s   %-30s\n",
            "Pivot", "Iteration j", "t-index", "qubit", "angle", "x-bit row (original)")
    println("-"^90)

    for po in result.pivot_origins
        xb_str = "[" * join([Int(po.xbit_row[j]) for j in 1:8], ",") * "]"
        @printf("%8d %12d %8d %8d %12.4f   %s\n",
                po.pivot_column, po.iteration_j, po.t_index, po.qubit, po.angle, xb_str)
    end
    println("-"^90)
    println()

    println("VERIFICATION: Does pivot column j come from iteration j?")
    all_match = true
    for po in result.pivot_origins
        match = po.pivot_column == po.iteration_j
        marker = match ? "YES" : "NO"
        if !match
            all_match = false
        end
        @printf("  Pivot col %d ← iteration %d: %s\n",
                po.pivot_column, po.iteration_j, marker)
    end
    println()
    if all_match
        println("RESULT: YES — pivot in column j is always established by iteration j.")
    else
        println("RESULT: NO — some pivots come from different iterations.")
        println("         Check which iteration actually provides each pivot.")
    end

    println()
    println("GATES PER ITERATION:")
    println("-"^60)
    @printf("%12s %10s %12s\n", "Iteration j", "# non-Cliff", "# independent")
    println("-"^60)
    for j in 1:8
        n_gates = count(r -> r.iteration_j == j, result.all_rows)
        n_indep = count(po -> po.iteration_j == j, result.pivot_origins)
        @printf("%12d %10d %12d\n", j, n_gates, n_indep)
    end
    println("-"^60)

    println()
    println("="^90)
    println("PIVOT ORIGIN TRACKING — N=4 (for manual verification)")
    println("="^90)
    println()

    result4 = track_pivot_origins(4)

    println("-"^80)
    @printf("%8s %12s %8s %8s %12s   %-20s\n",
            "Pivot", "Iteration j", "t-index", "qubit", "angle", "x-bit row")
    println("-"^80)

    for po in result4.pivot_origins
        xb_str = "[" * join([Int(po.xbit_row[j]) for j in 1:4], ",") * "]"
        @printf("%8d %12d %8d %8d %12.4f   %s\n",
                po.pivot_column, po.iteration_j, po.t_index, po.qubit, po.angle, xb_str)
    end
    println("-"^80)

    println()
    println("ALL non-Clifford rows for N=4:")
    println("-"^70)
    for r in result4.all_rows
        xb_str = "[" * join([Int(r.xbit[j]) for j in 1:4], ",") * "]"
        @printf("  t=%2d  iter_j=%d  qubit=%d  angle=%8.4f  xbit=%s\n",
                r.t_index, r.iteration_j, r.qubit, r.angle, xb_str)
    end
    println("-"^70)

    println()
    println()
    println("="^90)
    println("PART 2: Z_N EVOLUTION — N=6 (without OFD)")
    println("="^90)
    println()
    println("Tracking C†·Z_6·C after each gate. The claim is that the x-bit")
    println("of P̃ = C†·Z_6·C is all-zeros (no X/Y component on ANY qubit)")
    println("until H(6) is applied.")
    println()

    snapshots_no_ofd = track_zN_evolution(6; apply_ofd=false)

    println("-"^90)
    @printf("%4s %8s %-35s %12s   %s\n",
            "gate", "iter_j", "gate", "x(Z₆)≠0?", "P̃ = C†Z₆C")
    println("-"^90)

    prev_xb = nothing
    for s in snapshots_no_ofd
        xb = s.xbit_ZN
        changed = (prev_xb === nothing) || (xb != prev_xb)
        is_non_cliff_rz = startswith(s.gate_label, "Rz(") && !contains(s.gate_label, "cliff")
        is_interesting = changed || is_non_cliff_rz || contains(s.gate_label, "H(6)")

        if is_interesting
            xb_str = "[" * join([Int(xb[j]) for j in 1:6], ",") * "]"
            marker = s.xN_nonzero ? "YES ←" : "no"
            @printf("%4d %8d %-35s %12s   %s  xbit=%s\n",
                    s.gate_idx, s.iteration_j, s.gate_label, marker,
                    s.pauli_str, xb_str)
        end
        prev_xb = copy(xb)
    end
    println("-"^90)

    first_nonzero = findfirst(s -> s.xN_nonzero, snapshots_no_ofd)
    if first_nonzero !== nothing
        s = snapshots_no_ofd[first_nonzero]
        println()
        @printf("FIRST gate where x-bit of C†·Z₆·C becomes nonzero:\n")
        @printf("  gate %d: %s (iteration j=%d)\n", s.gate_idx, s.gate_label, s.iteration_j)
        @printf("  P̃ = %s\n", s.pauli_str)
    else
        println()
        println("x-bit of C†·Z₆·C is NEVER nonzero throughout the entire circuit!")
    end

    n_gates_before = first_nonzero !== nothing ? first_nonzero - 1 : length(snapshots_no_ofd)
    n_total = length(snapshots_no_ofd)
    @printf("Gates with x-bit all-zero: %d/%d\n", n_gates_before, n_total)

    println()
    println("="^90)
    println("PART 2b: Z_N EVOLUTION — N=6 (with OFD applied)")
    println("="^90)
    println()
    println("Same tracking but with OFD modifying the Clifford state (as in actual")
    println("GF(2) analysis). The key question: does OFD ever cause Z_N to acquire")
    println("X/Y components earlier than expected?")
    println()

    snapshots_ofd = track_zN_evolution(6; apply_ofd=true)

    println("-"^90)
    @printf("%4s %8s %-45s %12s   %s\n",
            "gate", "iter_j", "gate", "x(Z₆)≠0?", "P̃ = C†Z₆C")
    println("-"^90)

    prev_xb = nothing
    for s in snapshots_ofd
        xb = s.xbit_ZN
        changed = (prev_xb === nothing) || (xb != prev_xb)
        is_non_cliff_rz = startswith(s.gate_label, "Rz(") && !contains(s.gate_label, "cliff")
        is_interesting = changed || is_non_cliff_rz || contains(s.gate_label, "H(6)")

        if is_interesting
            xb_str = "[" * join([Int(xb[j]) for j in 1:6], ",") * "]"
            marker = s.xN_nonzero ? "YES ←" : "no"
            @printf("%4d %8d %-45s %12s   %s  xbit=%s\n",
                    s.gate_idx, s.iteration_j, s.gate_label, marker,
                    s.pauli_str, xb_str)
        end
        prev_xb = copy(xb)
    end
    println("-"^90)

    first_nonzero_ofd = findfirst(s -> s.xN_nonzero, snapshots_ofd)
    if first_nonzero_ofd !== nothing
        s = snapshots_ofd[first_nonzero_ofd]
        @printf("\nFIRST gate where x-bit of C†·Z₆·C becomes nonzero (with OFD):\n")
        @printf("  gate %d: %s (iteration j=%d)\n", s.gate_idx, s.gate_label, s.iteration_j)
    else
        println("\nx-bit of C†·Z₆·C is NEVER nonzero (with OFD)!")
    end

    println()
    println("="^90)
    println("PART 2c: Z_N x-bit ZERO VERIFICATION — N=4,6,8,16 (with OFD)")
    println("="^90)
    println()
    println("For each N, at which gate does x-bit of C†·Z_N·C first become nonzero?")
    println()

    for n in [4, 6, 8, 16]
        snapshots = track_zN_evolution(n; apply_ofd=true)
        first_nz = findfirst(s -> s.xN_nonzero, snapshots)
        total_gates = length(snapshots)

        if first_nz !== nothing
            s = snapshots[first_nz]
            @printf("  N=%2d: first nonzero at gate %d/%d — %s (iter j=%d)\n",
                    n, s.gate_idx, total_gates, s.gate_label, s.iteration_j)

            if contains(s.gate_label, "H($n)")
                println("         → This IS H(N), confirming Z_N stays Z-like until H(N)")
            else
                println("         → WARNING: This is NOT H(N)!")
            end
        else
            @printf("  N=%2d: NEVER nonzero (%d gates total)\n", n, total_gates)
        end
    end

    println()
    println()
    println("="^90)
    println("PART 3: CNOT SYMPLECTIC VERIFICATION — N=6")
    println("="^90)
    println()
    println("Detailed check: after each CNOT gate involving qubit 6,")
    println("what happens to C†·Z₆·C?")
    println()
    println("Symplectic rules for CNOT(control, target):")
    println("  Z_control → Z_control            (no X acquired)")
    println("  Z_target  → Z_control · Z_target  (no X acquired)")
    println("  X_control → X_control · X_target  (X spreads)")
    println("  X_target  → X_target              (no change)")
    println()
    println("So when qubit 6 is CONTROL: CNOT(6,t) maps Z₆ → Z₆ (unchanged)")
    println("When qubit 6 is TARGET:     CNOT(c,6) maps Z₆ → Z_c·Z₆")
    println("Either way, Z₆ stays in the Z-subspace (no X/Y acquired).")
    println()

    for s in snapshots_no_ofd
        if contains(s.gate_label, "CNOT") &&
           (contains(s.gate_label, "6,") || contains(s.gate_label, ",6)"))
            xb_str = "[" * join([Int(s.xbit_ZN[j]) for j in 1:6], ",") * "]"
            zb_str = "[" * join([Int(s.zbit_ZN[j]) for j in 1:6], ",") * "]"
            @printf("  gate %3d: %-25s → Z₆ maps to: %s  (xbit=%s, zbit=%s)\n",
                    s.gate_idx, s.gate_label, s.pauli_str, xb_str, zb_str)
        end
    end

    println()
    println("="^90)
    println("ANALYSIS COMPLETE")
    println("="^90)
end

main()
