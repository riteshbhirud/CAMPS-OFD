Experiment: Random Circuit GF(2) Rank Verification
====================================================

Part A: Absolute nullity convergence — verify that ν ≈ O(1) independent of N
for deep random Clifford+T circuits (t = N T-gates). This means ν/t → 0 as
N grows, confirming random circuits are OFD-friendly (near-full GF(2) rank).

Part B: Rank saturation — verify that GF(2) rank saturates at ~N as the
number of T-gates increases.

Both parts are pure GF(2) analysis — no MPS simulation needed.

Usage:
    julia benchmarks/experiment_random_rank.jl [mode]

Modes:
    "test"   - Quick validation (small n, few instances)
    "medium" - Standard sweep (100 instances per n)

Output:
    results/experiment_random_rank.csv
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

function run_nullity_convergence(mode::String)
    n_range = mode == "test" ? [4, 8, 16] : [4, 8, 12, 16, 24, 32, 48, 64]
    n_instances = mode == "test" ? 10 : 100
    results = []

    println("Part A: Absolute nullity convergence (ν ≈ O(1), ν/t → 0)")
    println("  n_range = $n_range, instances = $n_instances")
    println()

    for n in n_range
        t0 = time()
        for instance in 1:n_instances
            seed = Int(hash(("random_nullity", n, instance)) % UInt32)

            try
                params = Dict{Symbol, Any}(
                    :n_qubits => n,
                    :n_t_gates => n,
                    :clifford_layers => 2 * n,
                    :seed => seed
                )

                Random.seed!(seed)
                family = RandomAllToAllCliffordT()
                circuit = generate_circuit(family, params)

                gates = circuit.gates
                t_positions = circuit.t_gate_positions
                n_qubits = circuit.n_qubits

                Random.seed!(seed)
                gf2 = compute_gf2_for_mixed_circuit(
                    gates, t_positions, n_qubits;
                    seed=seed, simulate_ofd=true)

                t = gf2.n_t_gates
                nullity_ratio = t > 0 ? gf2.nullity / t : NaN

                push!(results, (
                    success = true,
                    part = "nullity_convergence",
                    n_qubits = n,
                    seed = seed,
                    t = t,
                    rank = gf2.gf2_rank,
                    nullity = gf2.nullity,
                    nullity_ratio = nullity_ratio,
                    t_density = 1.0
                ))
            catch e
                push!(results, (
                    success = false,
                    part = "nullity_convergence",
                    n_qubits = n,
                    seed = seed,
                    t = 0,
                    rank = 0,
                    nullity = 0,
                    nullity_ratio = NaN,
                    t_density = 1.0,
                    error_msg = sprint(showerror, e)
                ))
            end
        end
        dt = time() - t0
        nr = filter(r -> r.success && r.n_qubits == n, results)
        nullities = [r.nullity for r in nr]
        ratios = filter(!isnan, [r.nullity_ratio for r in nr])
        mean_null = isempty(nullities) ? NaN : mean(nullities)
        std_null = isempty(nullities) ? NaN : std(nullities)
        mean_ratio = isempty(ratios) ? NaN : mean(ratios)
        @printf("  N=%3d: mean(ν) = %.3f ± %.3f, mean(ν/t) = %.4f  (%d instances, %.1fs)\n",
                n, mean_null, std_null, mean_ratio, length(nullities), dt)
    end

    return results
end

function run_rank_saturation(mode::String)
    n_range = mode == "test" ? [8, 16] : [8, 16, 32]
    density_range = mode == "test" ? [0.5, 1.0, 2.0] :
                    [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    n_instances = mode == "test" ? 5 : 20
    results = []

    println("\nPart B: Rank saturation")
    println("  n_range = $n_range, densities = $density_range, instances = $n_instances")
    println()

    for n in n_range
        for td in density_range
            n_t = max(1, round(Int, td * n))

            for instance in 1:n_instances
                seed = Int(hash(("rank_saturation", n, td, instance)) % UInt32)

                try
                    params = Dict{Symbol, Any}(
                        :n_qubits => n,
                        :n_t_gates => n_t,
                        :clifford_layers => 2 * n,
                        :seed => seed
                    )

                    Random.seed!(seed)
                    family = RandomAllToAllCliffordT()
                    circuit = generate_circuit(family, params)

                    gates = circuit.gates
                    t_positions = circuit.t_gate_positions
                    n_qubits = circuit.n_qubits

                    Random.seed!(seed)
                    gf2 = compute_gf2_for_mixed_circuit(
                        gates, t_positions, n_qubits;
                        seed=seed, simulate_ofd=true)

                    t_actual = gf2.n_t_gates
                    nullity_ratio = t_actual > 0 ? gf2.nullity / t_actual : NaN

                    push!(results, (
                        success = true,
                        part = "rank_saturation",
                        n_qubits = n,
                        seed = seed,
                        t = t_actual,
                        rank = gf2.gf2_rank,
                        nullity = gf2.nullity,
                        nullity_ratio = nullity_ratio,
                        t_density = td
                    ))
                catch e
                    push!(results, (
                        success = false,
                        part = "rank_saturation",
                        n_qubits = n,
                        seed = seed,
                        t = 0,
                        rank = 0,
                        nullity = 0,
                        nullity_ratio = NaN,
                        t_density = td,
                        error_msg = sprint(showerror, e)
                    ))
                end
            end
        end

        nr = filter(r -> r.success && r.n_qubits == n && r.part == "rank_saturation", results)
        densities = sort(unique([r.t_density for r in nr]))
        @printf("  N=%3d saturation summary:\n", n)
        for td in densities
            dr = filter(r -> r.t_density == td, nr)
            mean_rank = mean([r.rank for r in dr])
            mean_t = mean([r.t for r in dr])
            @printf("    t/n=%.1f: mean_t=%.0f, mean_rank=%.1f (%.1f%% of N)\n",
                    td, mean_t, mean_rank, 100 * mean_rank / n)
        end
    end

    return results
end

function write_results_csv(results, output_path)
    open(output_path, "w") do io
        println(io, "success,part,n_qubits,seed,t,rank,nullity,nullity_ratio,t_density")

        for r in results
            println(io, join([
                r.success,
                "\"$(r.part)\"",
                r.n_qubits,
                r.seed,
                r.t,
                r.rank,
                r.nullity,
                isnan(r.nullity_ratio) ? "" : @sprintf("%.6f", r.nullity_ratio),
                @sprintf("%.4f", r.t_density)
            ], ","))
        end
    end
end

function print_summary(results)
    successful = filter(r -> r.success, results)

    part_a = filter(r -> r.part == "nullity_convergence", successful)
    if !isempty(part_a)
        println("\n" * "="^90)
        println("PART A: ABSOLUTE NULLITY CONVERGENCE (ν ≈ O(1), ν/t → 0)")
        println("="^90)
        @printf("%6s %8s %10s %10s %10s %10s\n",
                "N", "Inst", "Mean ν", "Std ν", "Mean ν/t", "Expect")
        println("-"^90)

        n_vals = sort(unique([r.n_qubits for r in part_a]))
        for n in n_vals
            nr = filter(r -> r.n_qubits == n, part_a)
            nullities = Float64.([r.nullity for r in nr])
            ratios = filter(!isnan, [r.nullity_ratio for r in nr])
            mean_null = mean(nullities)
            std_null = std(nullities)
            mean_ratio = isempty(ratios) ? NaN : mean(ratios)
            @printf("%6d %8d %10.3f %10.3f %10.4f %10s\n",
                    n, length(nr), mean_null, std_null, mean_ratio, "ν→O(1)")
        end
        println("="^90)
    end

    part_b = filter(r -> r.part == "rank_saturation", successful)
    if !isempty(part_b)
        println("\n" * "="^80)
        println("PART B: RANK SATURATION SUMMARY")
        println("="^80)
        @printf("%6s %8s %8s %10s %10s %10s\n",
                "N", "t/n", "Mean t", "Mean rank", "rank/N", "Saturated?")
        println("-"^80)

        n_vals = sort(unique([r.n_qubits for r in part_b]))
        for n in n_vals
            nr = filter(r -> r.n_qubits == n, part_b)
            densities = sort(unique([r.t_density for r in nr]))
            for td in densities
                dr = filter(r -> r.t_density == td, nr)
                mean_rank = mean([r.rank for r in dr])
                mean_t = mean([r.t for r in dr])
                ratio = mean_rank / n
                saturated = ratio > 0.9 ? "YES" : "no"
                @printf("%6d %8.1f %8.0f %10.1f %10.3f %10s\n",
                        n, td, mean_t, mean_rank, ratio, saturated)
            end
        end
        println("="^80)
    end
end

function main()
    mode = length(ARGS) >= 1 ? ARGS[1] : "medium"

    println("="^80)
    println("RANDOM CIRCUIT GF(2) RANK VERIFICATION")
    println("="^80)
    println("Mode: $mode")
    println("Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println()

    results_a = run_nullity_convergence(mode)

    results_b = run_rank_saturation(mode)

    all_results = vcat(results_a, results_b)

    output_dir = joinpath(camps_dir, "results")
    mkpath(output_dir)

    output_csv = joinpath(output_dir, "experiment_random_rank.csv")
    write_results_csv(all_results, output_csv)
    println("\nResults saved to: $output_csv")

    n_success = count(r -> r.success, all_results)
    n_failed = length(all_results) - n_success
    println("Total: $n_success successful, $n_failed failed")

    print_summary(all_results)

    return all_results
end

results = main()
