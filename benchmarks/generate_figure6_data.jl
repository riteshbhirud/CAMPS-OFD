Generate Figure 6 Data: QFT Incremental Rank Growth (Generator C)
==================================================================

Produces per-T-gate cumulative GF(2) rank for the Gen C QFT circuit
at N = 8, 16, 32. Uses the same pipeline as experiment_qft_large_n_genC.jl.

Output:
    results/figure6_qft_incremental_genC.csv

Usage:
    julia benchmarks/generate_figure6_data.jl
=

camps_dir = dirname(dirname(@__FILE__))
println("Activating CAMPS.jl environment: $camps_dir")
using Pkg
Pkg.activate(camps_dir)

using CAMPS
using QuantumClifford
using Printf

mutable struct OnlineRREF
    n::Int
    matrix::Matrix{Bool}
    pivot_col::Vector{Int}
    rank::Int
end

function OnlineRREF(n::Int)
    return OnlineRREF(n, zeros(Bool, n, n), zeros(Int, n), 0)
end

function try_insert!(rref::OnlineRREF, row::BitVector)::Bool
    @assert length(row) == rref.n

    r = copy(row)
    for i in 1:rref.rank
        pc = rref.pivot_col[i]
        if r[pc]
            for j in 1:rref.n
                r[j] = r[j] ⊻ rref.matrix[i, j]
            end
        end
    end

    new_pivot = findfirst(r)
    if new_pivot === nothing
        return false
    end

    rref.rank += 1
    for j in 1:rref.n
        rref.matrix[rref.rank, j] = r[j]
    end
    rref.pivot_col[rref.rank] = new_pivot
    return true
end

function qft_incremental_rank_genC(n_qubits::Int)
    circuit = qft_circuit(n_qubits)

    state = CAMPSState(n_qubits)
    initialize!(state)

    rref = OnlineRREF(n_qubits)
    t_count = 0

    records = NamedTuple[]

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
                t_count += 1
                prev_rank = rref.rank

                P_twisted = compute_twisted_pauli(state, gate.axis, gate.qubit)
                xb = BitVector(xbit(P_twisted))
                try_insert!(rref, xb)

                rank_delta = rref.rank - prev_rank

                push!(records, (
                    t_index = t_count,
                    qubit = gate.qubit,
                    angle = gate.angle,
                    cumulative_rank = rref.rank,
                    rank_delta = rank_delta
                ))

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

    return records, t_count, rref.rank
end

function main()
    println("="^80)
    println("GENERATING FIGURE 6 DATA: QFT INCREMENTAL RANK (GENERATOR C)")
    println("="^80)
    println()

    output_dir = joinpath(camps_dir, "results")
    mkpath(output_dir)
    output_csv = joinpath(output_dir, "figure6_qft_incremental_genC.csv")

    open(output_csv, "w") do io
        println(io, "n_qubits,t_index,qubit,angle,cumulative_rank,rank_delta")

        for n in [8, 16, 32]
            t_start = time()
            records, t_total, final_rank = qft_incremental_rank_genC(n)
            elapsed = time() - t_start

            @printf("N=%2d: t=%d, rank=%d (N-1=%d), time=%.1fs\n",
                    n, t_total, final_rank, n-1, elapsed)

            n_jumps = count(r -> r.rank_delta > 0, records)
            @printf("      rank jumps: %d (expected N-1=%d)\n", n_jumps, n-1)

            for r in records
                @printf(io, "%d,%d,%d,%.10f,%d,%d\n",
                        n, r.t_index, r.qubit, r.angle,
                        r.cumulative_rank, r.rank_delta)
            end
        end
    end

    println()
    println("Saved: $output_csv")
    println("="^80)
end

main()
