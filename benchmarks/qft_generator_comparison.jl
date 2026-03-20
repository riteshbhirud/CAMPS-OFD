QFT Generator Comparison: GF(2) Rank from All Three Generators
================================================================

Compares T-counts and GF(2) ranks from:
  A. QFTFamily (circuit_families_complete.jl) — benchmark/validation generator
  B. generate_qft_circuit_large (gf2_scaling_analysis.jl) — scaling experiment generator
  C. qft_circuit (simulation.jl) — standard QFT generator with exact angles

Output: prints comparison table to stdout.
=

camps_dir = dirname(dirname(@__FILE__))
println("Activating CAMPS.jl environment: $camps_dir")
using Pkg
Pkg.activate(camps_dir)

using CAMPS
using QuantumClifford
using QuantumClifford.ECC
using Random
using Printf

include(joinpath(camps_dir, "benchmarks", "circuit_families_complete.jl"))

function generate_qft_circuit_large_local(n_qubits::Int; density::Symbol=:medium, seed::Int=42)
    density in (:low, :medium, :high) || throw(ArgumentError("density must be :low, :medium, or :high"))
    rng = Random.MersenneTwister(seed)
    k_max = density == :low ? n_qubits + 1 : (density == :medium ? 6 : 4)

    gates = Gate[]
    t_positions = Int[]

    for j in 1:n_qubits
        push!(gates, CliffordGate([(:H, j)], [j]))
        for k in 2:(n_qubits - j + 1)
            k > k_max && break
            control_qubit = j + k - 1
            target_qubit = j
            if k == 2
                push!(gates, CliffordGate([(:CNOT, control_qubit, target_qubit)], [control_qubit, target_qubit]))
                push!(gates, CliffordGate([(:S, target_qubit)], [target_qubit]))
                push!(gates, CliffordGate([(:CNOT, control_qubit, target_qubit)], [control_qubit, target_qubit]))
                push!(gates, CliffordGate([(:Sdag, target_qubit)], [target_qubit]))
            elseif k == 3
                push!(gates, CliffordGate([(:CNOT, control_qubit, target_qubit)], [control_qubit, target_qubit]))
                push!(gates, RotationGate(target_qubit, :Z, -π/4))
                push!(t_positions, length(gates))
                push!(gates, CliffordGate([(:CNOT, control_qubit, target_qubit)], [control_qubit, target_qubit]))
                push!(gates, RotationGate(control_qubit, :Z, π/4))
                push!(t_positions, length(gates))
                push!(gates, RotationGate(target_qubit, :Z, π/4))
                push!(t_positions, length(gates))
                push!(gates, CliffordGate([(:CNOT, control_qubit, target_qubit)], [control_qubit, target_qubit]))
                push!(gates, RotationGate(target_qubit, :Z, -π/4))
                push!(t_positions, length(gates))
            else
                n_t_this_gate = 4 * (k - 2)
                push!(gates, CliffordGate([(:CNOT, control_qubit, target_qubit)], [control_qubit, target_qubit]))
                for _ in 1:n_t_this_gate
                    push!(gates, RotationGate(target_qubit, :Z, π/4))
                    push!(t_positions, length(gates))
                end
                push!(gates, CliffordGate([(:CNOT, control_qubit, target_qubit)], [control_qubit, target_qubit]))
                push!(gates, CliffordGate([(:H, target_qubit)], [target_qubit]))
            end
        end
    end

    n_swaps = div(n_qubits, 2)
    for i in 1:n_swaps
        qubit1 = i
        qubit2 = n_qubits - i + 1
        push!(gates, CliffordGate([(:CNOT, qubit1, qubit2)], [qubit1, qubit2]))
        push!(gates, CliffordGate([(:CNOT, qubit2, qubit1)], [qubit2, qubit1]))
        push!(gates, CliffordGate([(:CNOT, qubit1, qubit2)], [qubit1, qubit2]))
    end

    return (n_qubits=n_qubits, gates=gates, t_positions=t_positions,
            metadata=Dict{String,Any}("family"=>"QFT", "density"=>string(density),
                                       "k_max"=>k_max, "n_t_gates"=>length(t_positions)))
end

function gf2_rank_from_gate_circuit(circuit::Vector{<:Gate}, n_qubits::Int)
    state = CAMPSState(n_qubits)
    initialize!(state)

    twisted_paulis = PauliOperator[]
    n_ofd = 0
    n_non_ofd = 0

    for gate in circuit
        if gate isa CliffordGate
            apply_clifford_gate_to_state!(state, gate)
        elseif gate isa RotationGate
            if is_clifford_angle(gate.angle)
                clifford_gate = rotation_to_clifford(gate)
                if clifford_gate !== nothing
                    apply_clifford_gate_to_state!(state, clifford_gate)
                end
            else
                P_twisted = compute_twisted_pauli(state, gate.axis, gate.qubit)
                push!(twisted_paulis, P_twisted)

                control = find_disentangling_qubit(P_twisted, state.free_qubits)
                if control !== nothing
                    D_gates = build_disentangler_gates(P_twisted, control)
                    D_flat = flatten_gate_sequence(D_gates)
                    apply_inverse_gates!(state.clifford, D_flat)
                    state.free_qubits[control] = false
                    n_ofd += 1
                else
                    n_non_ofd += 1
                end
            end
        end
    end

    t = length(twisted_paulis)
    if t == 0
        return (t=0, rank=0, nullity=0, n_ofd=0, n_non_ofd=0)
    end

    M = build_gf2_matrix(twisted_paulis)
    r = gf2_rank(M)

    return (t=t, rank=r, nullity=t - r, n_ofd=n_ofd, n_non_ofd=n_non_ofd)
end

function main()
    println("="^100)
    println("QFT GENERATOR COMPARISON: T-Counts and GF(2) Ranks")
    println("="^100)
    println()

    test_ns = [4, 6, 8, 12, 16]
    seed = 42

    println("-"^100)
    @printf("%4s | %-20s | %6s %6s %6s %6s | %-20s | %6s %6s %6s %6s\n",
            "N", "Gen A/B (large)", "t", "rank", "ν", "N-2",
            "Gen C (qft_circuit)", "t", "rank", "ν", "N-2")
    println("-"^100)

    for n in test_ns
        circuit_ab = generate_qft_circuit_large_local(n; density=:low, seed=seed)
        gf2_ab = gf2_rank_from_gate_circuit(circuit_ab.gates, n)

        circuit_c = qft_circuit(n)
        gf2_c = gf2_rank_from_gate_circuit(circuit_c, n)

        match_ab = gf2_ab.rank == n - 2 ? "YES" : "NO"
        match_c = gf2_c.rank == n - 2 ? "YES" : "NO"

        @printf("%4d | %-20s | %6d %6d %6d %6s | %-20s | %6d %6d %6d %6s\n",
                n, "density=:low", gf2_ab.t, gf2_ab.rank, gf2_ab.nullity, match_ab,
                "exact angles", gf2_c.t, gf2_c.rank, gf2_c.nullity, match_c)
    end
    println("-"^100)

    println("\n")
    println("="^100)
    println("QFTFamily (benchmark validation) — Multiple Densities")
    println("="^100)
    println("-"^80)
    @printf("%4s %10s %6s %6s %6s %6s %6s\n", "N", "density", "t", "rank", "ν", "N-2?", "OFD%")
    println("-"^80)

    for n in [3, 4, 5, 6, 7, 8]
        for density in [:low, :medium, :high]
            try
                circuit_fam = generate_circuit(QFTFamily(); n_qubits=n, density=density, seed=seed)
                gf2_fam = gf2_rank_from_gate_circuit(circuit_fam.gates, n)

                match = gf2_fam.rank == n - 2 ? "YES" : "NO"
                ofd_pct = gf2_fam.t > 0 ? 100.0 * gf2_fam.n_ofd / gf2_fam.t : NaN

                @printf("%4d %10s %6d %6d %6d %6s %6.1f%%\n",
                        n, string(density), gf2_fam.t, gf2_fam.rank, gf2_fam.nullity, match,
                        ofd_pct)
            catch e
                @printf("%4d %10s   ERROR: %s\n", n, string(density), sprint(showerror, e))
            end
        end
    end
    println("-"^80)

    println("\n")
    println("="^100)
    println("Generator C (qft_circuit) — Rotation Angles Detail for N=8")
    println("="^100)

    circuit_c8 = qft_circuit(8)
    rot_count = 0
    cliff_rot_count = 0
    for gate in circuit_c8
        if gate isa RotationGate
            is_cliff = is_clifford_angle(gate.angle)
            if is_cliff
                cliff_rot_count += 1
            else
                rot_count += 1
            end
        end
    end
    println("Total RotationGate: $(rot_count + cliff_rot_count)")
    println("  Clifford angles (multiples of π/2): $cliff_rot_count")
    println("  Non-Clifford angles: $rot_count")
    println("  → These $rot_count non-Clifford gates generate the GF(2) matrix rows")

    angles = Set{Float64}()
    for gate in circuit_c8
        if gate isa RotationGate && !is_clifford_angle(gate.angle)
            push!(angles, abs(gate.angle))
        end
    end
    println("\n  Unique non-Clifford |angle| values:")
    for a in sort(collect(angles))
        @printf("    %.6f rad = π/%.1f\n", a, π/a)
    end

    println("\n" * "="^100)
    println("COMPARISON COMPLETE")
    println("="^100)
end

main()
