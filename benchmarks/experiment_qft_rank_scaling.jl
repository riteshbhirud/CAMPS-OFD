Experiment: QFT Rank Scaling — Power Law Fit and Per-R_k Analysis
==================================================================

Part A: Overall GF(2) rank scaling with qubit count N.
  Fits rank = a·N^b across N ∈ {4..64}.

Part B: Per-controlled-R_k incremental rank analysis.
  Shows how each controlled rotation contributes to GF(2) rank,
  revealing that each R_k adds approximately constant independent rows.

Both parts are pure GF(2) analysis — no MPS simulation needed.

Usage:
    julia benchmarks/experiment_qft_rank_scaling.jl [mode]

Modes:
    "test"   - Quick validation (small n range)
    "medium" - Standard sweep

Output:
    results/experiment_qft_rank_scaling.csv
    results/experiment_qft_rank_incremental.csv
=

camps_dir = dirname(dirname(@__FILE__))
println("Activating CAMPS.jl environment: $camps_dir")
using Pkg
Pkg.activate(camps_dir)

using CAMPS
using QuantumClifford
using QuantumClifford.ECC
using Random
using Statistics
using Printf
using Dates

include(joinpath(camps_dir, "benchmarks", "circuit_families_complete.jl"))
include(joinpath(camps_dir, "benchmarks", "gf2_scaling_analysis.jl"))

function run_scaling_experiments(mode::String)
    n_range = mode == "test" ? [4, 8, 16] : [4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]
    n_realizations = mode == "test" ? 2 : 4
    results = []

    for n in n_range
        for real in 1:n_realizations
            seed = Int(hash(("qft_scaling", n, real)) % UInt32)

            try
                circuit = generate_qft_circuit_large(n; density=:low, seed=seed)

                Random.seed!(seed)
                gf2 = compute_gf2_for_mixed_circuit(
                    circuit.gates, circuit.t_positions, circuit.n_qubits;
                    seed=seed, simulate_ofd=true)

                push!(results, (
                    success = true,
                    n_qubits = n,
                    seed = seed,
                    t = gf2.n_t_gates,
                    rank = gf2.gf2_rank,
                    nullity = gf2.nullity,
                    predicted_chi = gf2.predicted_chi,
                    nullity_ratio = gf2.n_t_gates > 0 ? gf2.nullity / gf2.n_t_gates : NaN
                ))
            catch e
                push!(results, (
                    success = false,
                    n_qubits = n,
                    seed = seed,
                    t = 0,
                    rank = 0,
                    nullity = 0,
                    predicted_chi = 0,
                    nullity_ratio = NaN,
                    error_msg = sprint(showerror, e)
                ))
            end
        end
    end

    return results
end

"""
    run_qft_incremental_analysis(n_qubits; seed=42)

Walk through a QFT circuit gate by gate, tracking GF(2) rank after each
non-Clifford gate (T-gate). Records which controlled-R_k rotation each
T-gate belongs to.

Returns a vector of records: (n_qubits, seed, t_index, control_j, rotation_k,
    cumulative_t, cumulative_rank, rank_delta)
"""
function run_qft_incremental_analysis(n_qubits::Int; seed::Int=42)
    circuit = generate_qft_circuit_large(n_qubits; density=:low, seed=seed)
    gates = circuit.gates

    state = CAMPSState(n_qubits)
    initialize!(state)

    twisted_paulis = PauliOperator[]
    records = NamedTuple[]
    t_idx = 0

    for gate in gates
        if gate isa CliffordGate
            apply_clifford_gate_to_state!(state, gate)

        elseif gate isa RotationGate
            if is_clifford_angle(gate.angle)
                clifford_gate = rotation_to_clifford(gate)
                if clifford_gate !== nothing
                    apply_clifford_gate_to_state!(state, clifford_gate)
                end
            else
                t_idx += 1

                P_twisted = compute_twisted_pauli(state, gate.axis, gate.qubit)
                push!(twisted_paulis, P_twisted)

                M = build_gf2_matrix(twisted_paulis)
                r = gf2_rank(M)
                rank_delta = t_idx == 1 ? r : r - records[end].cumulative_rank

                push!(records, (
                    n_qubits = n_qubits,
                    seed = seed,
                    t_index = t_idx,
                    control_j = 0,
                    rotation_k = 0,
                    cumulative_t = t_idx,
                    cumulative_rank = r,
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

    records_annotated = annotate_qft_t_gates(n_qubits, records; seed=seed)

    return records_annotated
end

"""
Annotate QFT T-gate records with their (j, k) controlled-R_k origin.
"""
function annotate_qft_t_gates(n_qubits::Int, records::Vector; seed::Int=42)
    t_to_jk = Tuple{Int,Int}[]

    k_max = n_qubits + 1

    for j in 1:n_qubits
        for k in 2:(n_qubits - j + 1)
            k > k_max && break

            if k == 2
                continue
            elseif k == 3
                for _ in 1:4
                    push!(t_to_jk, (j, k))
                end
            else
                for _ in 1:4*(k-2)
                    push!(t_to_jk, (j, k))
                end
            end
        end
    end

    annotated = NamedTuple[]
    for (i, rec) in enumerate(records)
        if i <= length(t_to_jk)
            j, k = t_to_jk[i]
        else
            j, k = 0, 0
        end
        push!(annotated, (
            n_qubits = rec.n_qubits,
            seed = rec.seed,
            t_index = rec.t_index,
            control_j = j,
            rotation_k = k,
            cumulative_t = rec.cumulative_t,
            cumulative_rank = rec.cumulative_rank,
            rank_delta = rec.rank_delta
        ))
    end

    return annotated
end

function fit_power_law(ns::Vector, ranks::Vector)
    valid = [(n, r) for (n, r) in zip(ns, ranks) if r > 0 && n > 0]
    isempty(valid) && return (NaN, NaN, NaN)

    log_ns = [log(v[1]) for v in valid]
    log_ranks = [log(v[2]) for v in valid]

    n = length(log_ns)
    sum_x = sum(log_ns)
    sum_y = sum(log_ranks)
    sum_xy = sum(log_ns .* log_ranks)
    sum_x2 = sum(log_ns .^ 2)

    denom = n * sum_x2 - sum_x^2
    abs(denom) < 1e-12 && return (NaN, NaN, NaN)

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    a = exp(intercept)
    b = slope

    mean_y = sum_y / n
    ss_res = sum((log_ranks .- (intercept .+ slope .* log_ns)).^2)
    ss_tot = sum((log_ranks .- mean_y).^2)
    r_squared = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN

    return (a, b, r_squared)
end

function write_scaling_csv(results, output_path)
    open(output_path, "w") do io
        println(io, "success,n_qubits,seed,t,rank,nullity,predicted_chi,nullity_ratio")
        for r in results
            println(io, join([
                r.success,
                r.n_qubits,
                r.seed,
                r.t,
                r.rank,
                r.nullity,
                r.predicted_chi,
                isnan(r.nullity_ratio) ? "" : @sprintf("%.6f", r.nullity_ratio)
            ], ","))
        end
    end
end

function write_incremental_csv(records, output_path)
    open(output_path, "w") do io
        println(io, "n_qubits,seed,t_index,control_j,rotation_k,cumulative_t,cumulative_rank,rank_delta")
        for r in records
            println(io, join([
                r.n_qubits,
                r.seed,
                r.t_index,
                r.control_j,
                r.rotation_k,
                r.cumulative_t,
                r.cumulative_rank,
                r.rank_delta
            ], ","))
        end
    end
end

function main()
    mode = length(ARGS) >= 1 ? ARGS[1] : "medium"

    println("="^80)
    println("QFT RANK SCALING — POWER LAW FIT AND PER-R_k ANALYSIS")
    println("="^80)
    println("Mode: $mode")
    println("Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println()

    println("="^60)
    println("PART A: QFT Rank Scaling with Qubit Count")
    println("="^60)

    scaling_results = run_scaling_experiments(mode)
    successful = filter(r -> r.success, scaling_results)

    println("\nPer-N summary:")
    println("-"^60)
    @printf("%6s %8s %8s %8s %10s\n", "N", "T", "Rank", "Nullity", "ν/t")
    println("-"^60)

    n_vals_seen = sort(unique([r.n_qubits for r in successful]))
    mean_ranks = Float64[]
    for n in n_vals_seen
        nr = filter(r -> r.n_qubits == n, successful)
        mean_t = mean([r.t for r in nr])
        mean_rank = mean([r.rank for r in nr])
        mean_null = mean([r.nullity for r in nr])
        mean_nr = mean(filter(!isnan, [r.nullity_ratio for r in nr]))
        push!(mean_ranks, mean_rank)
        @printf("%6d %8.0f %8.1f %8.1f %10.4f\n", n, mean_t, mean_rank, mean_null, mean_nr)
    end
    println("-"^60)

    a, b, r_sq = fit_power_law(Float64.(n_vals_seen), mean_ranks)
    @printf("\nPower law fit: rank ≈ %.4f · N^%.4f  (R² = %.4f)\n", a, b, r_sq)

    println("\n" * "="^60)
    println("PART B: Per-R_k Incremental Rank Analysis")
    println("="^60)

    incr_n_range = mode == "test" ? [8] : [8, 16, 32]
    all_incremental = NamedTuple[]

    for n in incr_n_range
        seed = Int(hash(("qft_incr", n)) % UInt32)
        println("\nRunning incremental analysis for N=$n...")
        records = run_qft_incremental_analysis(n; seed=seed)
        append!(all_incremental, records)

        k_vals = sort(unique([r.rotation_k for r in records if r.rotation_k > 0]))
        @printf("  N=%d: %d T-gates, final rank=%d\n", n, length(records),
                isempty(records) ? 0 : records[end].cumulative_rank)
        @printf("  Per-R_k rank contributions:\n")
        for k in k_vals
            kr = filter(r -> r.rotation_k == k, records)
            total_delta = sum([r.rank_delta for r in kr])
            n_t = length(kr)
            @printf("    R_%d: %d T-gates, rank contribution = %d\n", k, n_t, total_delta)
        end
    end

    output_dir = joinpath(camps_dir, "results")
    mkpath(output_dir)

    csv_scaling = joinpath(output_dir, "experiment_qft_rank_scaling.csv")
    write_scaling_csv(scaling_results, csv_scaling)
    println("\nScaling results saved to: $csv_scaling")

    csv_incr = joinpath(output_dir, "experiment_qft_rank_incremental.csv")
    write_incremental_csv(all_incremental, csv_incr)
    println("Incremental results saved to: $csv_incr")

    println("\n" * "="^60)
    println("QFT RANK SCALING — COMPLETE")
    println("="^60)

    return scaling_results, all_incremental
end

results = main()
