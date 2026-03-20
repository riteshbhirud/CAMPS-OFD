GF(2) Rank Analysis Across All 14 Circuit Families
====================================================

This script computes the GF(2) rank for every circuit in the benchmark suite,
bridging the empirical OFD success rates to Liu & Clark's theoretical framework.

Key output: For each circuit, rank(z) is computed where z is the GF(2) matrix
of twisted Pauli strings. The bond dimension is bounded by 2^(t - rank(z)).

Usage:
    julia benchmarks/gf2_rank_analysis.jl [mode]

Modes:
    "test"   - 14 circuits (one per family, fast)
    "medium" - Full sweep matching benchmark parameters (default)

Output:
    results/gf2_rank_analysis.csv          - Raw GF(2) data per circuit
    results/results_with_gf2.csv           - Joined with existing benchmark results
    (prints per-family summary to stdout)
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

function get_family_from_name(family_name::String)
    if family_name == "Random Clifford+T (Brick-wall)"
        return RandomBrickwallCliffordT()
    elseif family_name == "Random Clifford+T (All-to-all)"
        return RandomAllToAllCliffordT()
    elseif family_name == "Bernstein-Vazirani"
        return BernsteinVaziraniCircuit()
    elseif family_name == "Simon's Algorithm"
        return SimonCircuit()
    elseif family_name == "Deutsch-Jozsa"
        return DeutschJozsaCircuit()
    elseif family_name == "GHZ State"
        return GHZStateCircuit()
    elseif family_name == "Bell State / EPR Pairs"
        return BellStateCircuit()
    elseif family_name == "Graph State"
        return GraphStateCircuit()
    elseif family_name == "Cluster State (1D)"
        return ClusterStateCircuit()
    elseif family_name == "QAOA MaxCut (p=1, 3-regular)"
        return QAOAMaxCutCircuit()
    elseif family_name == "Surface Code"
        return SurfaceCodeFamily()
    elseif family_name == "Quantum Fourier Transform"
        return QFTFamily()
    elseif family_name == "Grover Search"
        return GroverFamily()
    elseif family_name == "VQE Hardware-Efficient Ansatz"
        return VQEFamily()
    else
        error("Unknown family: $family_name")
    end
end

function analyze_single_circuit_gf2(family_name::String, params::Dict)
    try
        family = get_family_from_name(family_name)

        Random.seed!(params[:seed])

        circuit_result = if family isa Union{QFTFamily, GroverFamily, VQEFamily, SurfaceCodeFamily}
            if family isa QFTFamily
                generate_circuit(family; n_qubits=params[:n_qubits],
                               density=params[:density], seed=params[:seed])
            elseif family isa GroverFamily
                generate_circuit(family; n_qubits=params[:n_qubits],
                               density=params[:density], seed=params[:seed])
            elseif family isa VQEFamily
                generate_circuit(family; n_qubits=params[:n_qubits],
                               layers=params[:layers], seed=params[:seed])
            elseif family isa SurfaceCodeFamily
                generate_circuit(family; n_qubits=params[:n_qubits],
                               n_t_gates=params[:n_t_gates], seed=params[:seed])
            end
        else
            generate_circuit(family, params)
        end

        if circuit_result isa CircuitInstance
            n_qubits = circuit_result.n_qubits
            gates = circuit_result.gates
            t_positions = circuit_result.t_gate_positions
        else
            n_qubits = circuit_result.n_qubits
            gates = circuit_result.gates
            t_positions = circuit_result.t_positions
        end

        Random.seed!(params[:seed])
        if family isa Union{QFTFamily, GroverFamily, VQEFamily, SurfaceCodeFamily}
            if family isa QFTFamily
                generate_circuit(family; n_qubits=params[:n_qubits],
                               density=params[:density], seed=params[:seed])
            elseif family isa GroverFamily
                generate_circuit(family; n_qubits=params[:n_qubits],
                               density=params[:density], seed=params[:seed])
            elseif family isa VQEFamily
                generate_circuit(family; n_qubits=params[:n_qubits],
                               layers=params[:layers], seed=params[:seed])
            elseif family isa SurfaceCodeFamily
                generate_circuit(family; n_qubits=params[:n_qubits],
                               n_t_gates=params[:n_t_gates], seed=params[:seed])
            end
        else
            generate_circuit(family, params)
        end

        gf2_result = compute_gf2_for_mixed_circuit(
            gates, t_positions, n_qubits;
            seed=params[:seed], simulate_ofd=true)

        t_density = n_qubits > 0 ? gf2_result.n_t_gates / n_qubits : NaN

        return (
            success = true,
            family = family_name,
            n_qubits = n_qubits,
            seed = params[:seed],
            n_t_gates = gf2_result.n_t_gates,
            n_total_gates = length(gates),
            gf2_rank = gf2_result.gf2_rank,
            nullity = gf2_result.nullity,
            predicted_chi_gf2 = gf2_result.predicted_chi,
            n_disentanglable = gf2_result.n_disentanglable,
            n_not_disentanglable = gf2_result.n_not_disentanglable,
            rank_to_t_ratio = gf2_result.rank_to_t_ratio,
            t_density = t_density,
            xbit_density = gf2_result.xbit_density
        )
    catch e
        return (
            success = false,
            family = family_name,
            n_qubits = get(params, :n_qubits, 0),
            seed = params[:seed],
            n_t_gates = 0,
            n_total_gates = 0,
            gf2_rank = 0,
            nullity = 0,
            predicted_chi_gf2 = 0,
            n_disentanglable = 0,
            n_not_disentanglable = 0,
            rank_to_t_ratio = NaN,
            t_density = NaN,
            xbit_density = NaN,
            error = "$(string(e))"
        )
    end
end

function generate_test_experiments()
    experiments = []
    seed_base = 1000

    phase1_configs = [
        ("Random Clifford+T (Brick-wall)", Dict{Symbol,Any}(:n_qubits => 8, :n_t_gates => 4, :clifford_depth => 2, :seed => seed_base)),
        ("Random Clifford+T (All-to-all)", Dict{Symbol,Any}(:n_qubits => 8, :n_t_gates => 4, :clifford_layers => 16, :seed => seed_base+1)),
        ("Bernstein-Vazirani", Dict{Symbol,Any}(:n_qubits => 8, :n_t_gates => 4, :seed => seed_base+2)),
        ("Simon's Algorithm", Dict{Symbol,Any}(:n_qubits => 8, :n_t_gates => 4, :seed => seed_base+3)),
        ("Deutsch-Jozsa", Dict{Symbol,Any}(:n_qubits => 8, :n_t_gates => 4, :function_type => :balanced, :seed => seed_base+4)),
        ("GHZ State", Dict{Symbol,Any}(:n_qubits => 8, :n_t_gates => 4, :seed => seed_base+5)),
        ("Bell State / EPR Pairs", Dict{Symbol,Any}(:n_qubits => 8, :n_t_gates => 4, :seed => seed_base+6)),
        ("Graph State", Dict{Symbol,Any}(:n_qubits => 8, :n_t_gates => 4, :edge_probability => 0.3, :seed => seed_base+7)),
        ("Cluster State (1D)", Dict{Symbol,Any}(:n_qubits => 8, :n_t_gates => 4, :seed => seed_base+8)),
    ]

    for (name, params) in phase1_configs
        push!(experiments, (family_name=name, params=params))
    end

    seed_base2 = 2000
    phase2_configs = [
        ("QAOA MaxCut (p=1, 3-regular)", Dict{Symbol,Any}(:n_qubits => 8, :n_t_gates => 4, :seed => seed_base2)),
        ("Surface Code", Dict{Symbol,Any}(:n_qubits => 8, :n_t_gates => 4, :seed => seed_base2+1)),
        ("Quantum Fourier Transform", Dict{Symbol,Any}(:n_qubits => 4, :density => :low, :seed => seed_base2+2)),
        ("Grover Search", Dict{Symbol,Any}(:n_qubits => 4, :density => :quarter, :seed => seed_base2+3)),
        ("VQE Hardware-Efficient Ansatz", Dict{Symbol,Any}(:n_qubits => 4, :layers => 1, :seed => seed_base2+4)),
    ]

    for (name, params) in phase2_configs
        push!(experiments, (family_name=name, params=params))
    end

    return experiments
end

function generate_medium_experiments(n_realizations=8)
    experiments = []

    families = get_phase1_families()
    n_range = [8, 12, 16]
    t_fraction_range = [0.5, 1.0, 1.5]

    for family in families
        family_name = get_name(family)
        for n in n_range
            for t_frac in t_fraction_range
                n_t = Int(round(n * t_frac))
                for real in 1:n_realizations
                    seed = hash((family_name, n, n_t, real)) % UInt32

                    params = Dict{Symbol, Any}(
                        :n_qubits => n,
                        :n_t_gates => n_t,
                        :seed => seed
                    )

                    if family isa RandomBrickwallCliffordT
                        params[:clifford_depth] = 2
                    elseif family isa RandomAllToAllCliffordT
                        params[:clifford_layers] = 2 * n
                    elseif family isa SimonCircuit && n % 2 != 0
                        continue
                    elseif family isa DeutschJozsaCircuit
                        rng_temp = Random.MersenneTwister(seed)
                        params[:function_type] = rand(rng_temp, [:constant, :balanced])
                    elseif family isa GraphStateCircuit
                        params[:edge_probability] = 0.3
                    end

                    push!(experiments, (family_name=family_name, params=params))
                end
            end
        end
    end

    for n in [8, 12, 16]
        for t_frac in [0.5, 1.0, 1.5]
            n_t = Int(round(n * t_frac))
            for real in 1:n_realizations
                seed = Int(hash(("QAOA", n, n_t, real)) % UInt32)
                push!(experiments, (family_name="QAOA MaxCut (p=1, 3-regular)",
                    params=Dict{Symbol,Any}(:n_qubits => n, :n_t_gates => n_t, :seed => seed)))
            end
        end
    end

    for n_target in [8, 12, 16]
        for n_t in [4, 8, 16]
            for real in 1:n_realizations
                seed = Int(hash(("Surface", n_target, n_t, real)) % UInt32)
                push!(experiments, (family_name="Surface Code",
                    params=Dict{Symbol,Any}(:n_qubits => n_target, :n_t_gates => n_t, :seed => seed)))
            end
        end
    end

    for n in [4, 6, 8]
        for density in [:low, :medium, :high]
            for real in 1:n_realizations
                seed = Int(hash(("QFT", n, density, real)) % UInt32)
                push!(experiments, (family_name="Quantum Fourier Transform",
                    params=Dict{Symbol,Any}(:n_qubits => n, :density => density, :seed => seed)))
            end
        end
    end

    for n in [4, 6, 8]
        for density in [:full, :half, :quarter]
            for real in 1:n_realizations
                seed = Int(hash(("Grover", n, density, real)) % UInt32)
                push!(experiments, (family_name="Grover Search",
                    params=Dict{Symbol,Any}(:n_qubits => n, :density => density, :seed => seed)))
            end
        end
    end

    for n in [4, 6, 8]
        for layers in [1, 2, 4]
            for real in 1:n_realizations
                seed = Int(hash(("VQE", n, layers, real)) % UInt32)
                push!(experiments, (family_name="VQE Hardware-Efficient Ansatz",
                    params=Dict{Symbol,Any}(:n_qubits => n, :layers => layers, :seed => seed)))
            end
        end
    end

    return experiments
end

function print_family_summary(results)
    println()
    println("="^100)
    println("GF(2) RANK ANALYSIS - PER-FAMILY SUMMARY")
    println("="^100)
    @printf("%-40s %6s %6s %8s %8s %8s %8s %8s\n",
            "Family", "Count", "T-gates", "Rank", "Nullity", "Rank/T", "Xbit%", "T-dens")
    println("-"^100)

    family_names = sort(unique([r.family for r in results]))

    for family in family_names
        family_results = filter(r -> r.family == family && r.success, results)
        if isempty(family_results)
            @printf("%-40s %6d  (all failed)\n", family, 0)
            continue
        end

        n = length(family_results)
        mean_t = mean([r.n_t_gates for r in family_results])
        mean_rank = mean([r.gf2_rank for r in family_results])
        mean_nullity = mean([r.nullity for r in family_results])
        mean_ratio = mean(filter(!isnan, [r.rank_to_t_ratio for r in family_results]))
        mean_xbit = mean(filter(!isnan, [r.xbit_density for r in family_results]))
        mean_tdensity = mean(filter(!isnan, [r.t_density for r in family_results]))

        @printf("%-40s %6d %6.1f %8.1f %8.1f %8.3f %8.3f %8.2f\n",
                family, n, mean_t, mean_rank, mean_nullity,
                isempty(filter(!isnan, [r.rank_to_t_ratio for r in family_results])) ? NaN : mean_ratio,
                isempty(filter(!isnan, [r.xbit_density for r in family_results])) ? NaN : mean_xbit,
                isempty(filter(!isnan, [r.t_density for r in family_results])) ? NaN : mean_tdensity)
    end

    println("="^100)
end

function print_correlation_analysis(results)
    println()
    println("="^80)
    println("CORRELATION ANALYSIS")
    println("="^80)

    family_names = sort(unique([r.family for r in results]))
    family_stats = []

    for family in family_names
        fr = filter(r -> r.family == family && r.success, results)
        if isempty(fr)
            continue
        end

        push!(family_stats, (
            family = family,
            mean_rank_to_t = mean(filter(!isnan, [r.rank_to_t_ratio for r in fr])),
            mean_nullity = mean([r.nullity for r in fr]),
            mean_t_density = mean(filter(!isnan, [r.t_density for r in fr])),
            mean_xbit_density = mean(filter(!isnan, [r.xbit_density for r in fr])),
            n_circuits = length(fr)
        ))
    end

    println()
    println("Per-family mean GF(2) metrics:")
    @printf("%-40s %10s %10s %10s %10s\n",
            "Family", "Rank/T", "Nullity", "T-density", "Xbit%")
    println("-"^80)

    for s in family_stats
        @printf("%-40s %10.4f %10.1f %10.3f %10.4f\n",
                s.family, s.mean_rank_to_t, s.mean_nullity,
                s.mean_t_density, s.mean_xbit_density)
    end

    println()
    println("Key insight: Families with low rank_to_t_ratio have high nullity,")
    println("meaning many T-gates are GF(2)-dependent and cannot be disentangled.")
    println("This directly explains why OFD fails for structured algorithms.")
    println("="^80)
end

function main()
    mode = length(ARGS) >= 1 ? ARGS[1] : "medium"

    println("="^80)
    println("GF(2) RANK ANALYSIS - ALL 14 CIRCUIT FAMILIES")
    println("="^80)
    println("Mode: $mode")
    println("Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println()

    experiments = if mode == "test"
        generate_test_experiments()
    elseif mode == "medium"
        generate_medium_experiments(8)
    else
        error("Unknown mode: $mode. Use 'test' or 'medium'.")
    end

    n_total = length(experiments)
    println("Total experiments: $n_total")
    println()
    println("Starting GF(2) analysis (Clifford-only, no MPS needed)...")
    println("-"^80)

    start_time = time()
    results = []

    for (i, (family_name, params)) in enumerate(experiments)
        result = analyze_single_circuit_gf2(family_name, params)
        push!(results, result)

        if i % 50 == 0 || i == n_total
            elapsed = time() - start_time
            @printf("[%5d/%5d] %.1f%% complete (%.1f sec)\n",
                    i, n_total, 100 * i / n_total, elapsed)
        end
    end

    total_time = time() - start_time
    println("-"^80)
    @printf("Completed %d experiments in %.1f seconds\n", n_total, total_time)

    n_success = count(r -> r.success, results)
    n_failed = n_total - n_success
    if n_failed > 0
        println("Warning: $n_failed experiments failed")
        for r in filter(r -> !r.success, results)
            if hasproperty(r, :error)
                println("  $(r.family) (n=$(r.n_qubits)): $(r.error)")
            end
        end
    end

    output_dir = joinpath(camps_dir, "results")
    mkpath(output_dir)

    gf2_csv = joinpath(output_dir, "gf2_rank_analysis.csv")
    open(gf2_csv, "w") do io
        println(io, "success,family,n_qubits,seed,n_t_gates,n_total_gates,gf2_rank,nullity,predicted_chi_gf2,n_disentanglable,n_not_disentanglable,rank_to_t_ratio,t_density,xbit_density")

        for r in results
            println(io, join([
                r.success,
                r.family,
                r.n_qubits,
                r.seed,
                r.n_t_gates,
                r.n_total_gates,
                r.gf2_rank,
                r.nullity,
                r.predicted_chi_gf2,
                r.n_disentanglable,
                r.n_not_disentanglable,
                isnan(r.rank_to_t_ratio) ? "" : @sprintf("%.6f", r.rank_to_t_ratio),
                isnan(r.t_density) ? "" : @sprintf("%.4f", r.t_density),
                isnan(r.xbit_density) ? "" : @sprintf("%.6f", r.xbit_density)
            ], ","))
        end
    end
    println("\nGF(2) results saved to: $gf2_csv")

    benchmark_csv = joinpath(output_dir, "results_for_ml.csv")
    if isfile(benchmark_csv)
        println("\nJoining with existing benchmark results: $benchmark_csv")
        try
            join_with_benchmark_results(results, benchmark_csv, output_dir)
        catch e
            println("Warning: Could not join with benchmark results: $e")
        end
    else
        println("\nNo existing benchmark results found at: $benchmark_csv")
        println("Run the benchmark first, then re-run this script to produce joined results.")
    end

    successful_results = filter(r -> r.success, results)
    print_family_summary(successful_results)
    print_correlation_analysis(successful_results)

    return results
end

function join_with_benchmark_results(gf2_results, benchmark_csv, output_dir)
    lines = readlines(benchmark_csv)
    header = split(lines[1], ",")

    family_col = findfirst(==("family"), header)
    nqubits_col = findfirst(==("n_qubits"), header)
    seed_col = findfirst(==("seed"), header)
    ofd_rate_col = findfirst(==("ofd_rate"), header)
    final_chi_col = findfirst(==("final_chi"), header)

    if any(isnothing, [family_col, nqubits_col, seed_col])
        println("Warning: Could not find join columns in benchmark CSV")
        return
    end

    benchmark_lookup = Dict{Tuple{String, Int, Int}, Dict{String, String}}()
    for line in lines[2:end]
        fields = split(line, ",")
        if length(fields) >= maximum([family_col, nqubits_col, seed_col])
            key = (fields[family_col],
                   parse(Int, fields[nqubits_col]),
                   parse(Int, fields[seed_col]))
            row_dict = Dict(string(header[i]) => fields[i] for i in 1:min(length(header), length(fields)))
            benchmark_lookup[key] = row_dict
        end
    end

    combined_csv = joinpath(output_dir, "results_with_gf2.csv")
    combined_header = vcat(string.(header),
        ["gf2_rank", "nullity", "predicted_chi_gf2",
         "n_disentanglable", "n_not_disentanglable",
         "rank_to_t_ratio", "t_density", "xbit_density"])

    n_joined = 0
    open(combined_csv, "w") do io
        println(io, join(combined_header, ","))

        for r in gf2_results
            if !r.success
                continue
            end

            key = (r.family, r.n_qubits, Int(r.seed))
            if haskey(benchmark_lookup, key)
                bm = benchmark_lookup[key]
                benchmark_row = [get(bm, string(h), "") for h in header]

                gf2_cols = [
                    string(r.gf2_rank),
                    string(r.nullity),
                    string(r.predicted_chi_gf2),
                    string(r.n_disentanglable),
                    string(r.n_not_disentanglable),
                    isnan(r.rank_to_t_ratio) ? "" : @sprintf("%.6f", r.rank_to_t_ratio),
                    isnan(r.t_density) ? "" : @sprintf("%.4f", r.t_density),
                    isnan(r.xbit_density) ? "" : @sprintf("%.6f", r.xbit_density)
                ]

                println(io, join(vcat(benchmark_row, gf2_cols), ","))
                n_joined += 1
            end
        end
    end

    println("Combined results saved to: $combined_csv")
    println("Joined $n_joined / $(count(r -> r.success, gf2_results)) GF(2) results with benchmark data")

    if ofd_rate_col !== nothing && final_chi_col !== nothing
        println("\nCorrelation spot-check (first 5 joined rows):")
        @printf("%-40s %8s %8s %8s %8s\n", "Family", "OFD Rate", "Rank/T", "Final_chi", "Pred_chi")
        println("-"^80)

        count = 0
        for r in gf2_results
            if !r.success || count >= 5
                continue
            end
            key = (r.family, r.n_qubits, Int(r.seed))
            if haskey(benchmark_lookup, key)
                bm = benchmark_lookup[key]
                ofd_rate = get(bm, "ofd_rate", "N/A")
                final_chi = get(bm, "final_chi", "N/A")
                @printf("%-40s %8s %8.4f %8s %8d\n",
                        r.family, ofd_rate,
                        isnan(r.rank_to_t_ratio) ? 0.0 : r.rank_to_t_ratio,
                        final_chi, r.predicted_chi_gf2)
                count += 1
            end
        end
    end
end

results = main()
