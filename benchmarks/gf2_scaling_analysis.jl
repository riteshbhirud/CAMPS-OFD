GF(2) Null Space Scaling Analysis — 4 to 256 Qubits
=====================================================

Computes GF(2) rank, nullity, and scaling behavior for all 14 circuit families
at qubit counts from 4 to 256. Uses Clifford-only walk (no MPS), enabling
analysis at scales far beyond full CAMPS simulation.

This is a key experiment for the paper: it establishes three scaling classes
(OFD-friendly, OFD-hostile, intermediate) based on how nullity ratio ν(n)/t(n)
behaves as n grows.

Usage:
    julia benchmarks/gf2_scaling_analysis.jl [mode]

Modes:
    "test"   - Small subset for validation (seconds)
    "medium" - Up to 64 qubits (minutes)
    "full"   - Up to 256 qubits (may take ~1 hour for Grover at large n)

Output:
    results/gf2_scaling_analysis.csv
    (prints summary table and scaling fits to stdout)
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

"""
    generate_qft_circuit_large(n_qubits; density=:medium, seed=42)

Generate QFT circuit with exact-angle Eq. 3 decomposition at arbitrary qubit count.
Same decomposition logic as QFTFamily but without the n≤8 restriction.

The QFT on n qubits consists of:
- n Hadamard gates
- n(n-1)/2 controlled-R_k gates (k=2..n)
- Each controlled-R_k decomposes via Eq. 3 into 3 non-Clifford Rz gates
  with exact angle θ = 2π/2^k

density controls the maximum k value:
- :low → k_max = n+1 (all rotations, full QFT)
- :medium → k_max = 6
- :high → k_max = 4 (fewest rotations, most approximate)
"""
function generate_qft_circuit_large(n_qubits::Int; density::Symbol=:medium, seed::Int=42)
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

            θ = 2π / 2^k
            push!(gates, RotationGate(target_qubit, :Z, θ/2))
            push!(t_positions, length(gates))
            push!(gates, CliffordGate([(:CNOT, control_qubit, target_qubit)], [control_qubit, target_qubit]))
            push!(gates, RotationGate(target_qubit, :Z, -θ/2))
            push!(t_positions, length(gates))
            push!(gates, CliffordGate([(:CNOT, control_qubit, target_qubit)], [control_qubit, target_qubit]))
            push!(gates, RotationGate(control_qubit, :Z, θ/2))
            push!(t_positions, length(gates))
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

"""
    add_toffoli_gate_large!(gates, t_positions, control1, control2, target)

Decompose Toffoli into Clifford+T (7 T-gates). Same as Grover's decomposition.
"""
function add_toffoli_gate_large!(gates::Vector{Gate}, t_positions::Vector{Int},
                                  control1::Int, control2::Int, target::Int)
    push!(gates, CliffordGate([(:H, target)], [target]))
    push!(gates, CliffordGate([(:CNOT, control2, target)], [control2, target]))
    push!(gates, RotationGate(target, :Z, -π/4))
    push!(t_positions, length(gates))
    push!(gates, CliffordGate([(:CNOT, control1, target)], [control1, target]))
    push!(gates, RotationGate(target, :Z, π/4))
    push!(t_positions, length(gates))
    push!(gates, CliffordGate([(:CNOT, control2, target)], [control2, target]))
    push!(gates, RotationGate(target, :Z, -π/4))
    push!(t_positions, length(gates))
    push!(gates, CliffordGate([(:CNOT, control1, target)], [control1, target]))
    push!(gates, RotationGate(control2, :Z, π/4))
    push!(t_positions, length(gates))
    push!(gates, RotationGate(target, :Z, π/4))
    push!(t_positions, length(gates))
    push!(gates, CliffordGate([(:CNOT, control1, control2)], [control1, control2]))
    push!(gates, RotationGate(control1, :Z, π/4))
    push!(t_positions, length(gates))
    push!(gates, RotationGate(control2, :Z, -π/4))
    push!(t_positions, length(gates))
    push!(gates, CliffordGate([(:CNOT, control1, control2)], [control1, control2]))
    push!(gates, CliffordGate([(:H, target)], [target]))
end

"""
    add_multi_controlled_not_large!(gates, t_positions, n_qubits)

Multi-controlled NOT via cascade of Toffoli gates.
"""
function add_multi_controlled_not_large!(gates::Vector{Gate}, t_positions::Vector{Int}, n_qubits::Int)
    if n_qubits == 1
        push!(gates, CliffordGate([(:X, 1)], [1]))
    elseif n_qubits == 2
        push!(gates, CliffordGate([(:CNOT, 1, 2)], [1, 2]))
    elseif n_qubits == 3
        add_toffoli_gate_large!(gates, t_positions, 1, 2, 3)
    else
        for i in 1:(n_qubits-2)
            add_toffoli_gate_large!(gates, t_positions, i, i+1, i+2)
        end
        for i in (n_qubits-3):-1:1
            add_toffoli_gate_large!(gates, t_positions, i, i+1, i+2)
        end
    end
end

"""
    generate_grover_circuit_large(n_qubits; density=:full, seed=42, max_iterations=nothing)

Generate Grover's search circuit at arbitrary qubit count.
Same structure as GroverFamily but without n≤8 restriction.

For large n, Grover requires O(√2^n) iterations, each with O(n) Toffoli gates.
T-count per iteration ≈ 7 * 2*(n-2) = 14(n-2) T-gates (two MCX passes per iteration).
Total T-count ≈ 14(n-2) * ceil(π/4 * √2^n).

At n=16, this is ~14*14 * 201 ≈ 39,396 T-gates — still tractable for GF(2) analysis.
At n=20, this is ~14*18 * 805 ≈ 202,860 — may be slow for GF(2).

max_iterations can cap the iteration count to keep analysis tractable.
"""
function generate_grover_circuit_large(n_qubits::Int; density::Symbol=:full, seed::Int=42,
                                        max_iterations::Union{Int,Nothing}=nothing)
    density in (:full, :half, :quarter) || throw(ArgumentError("density must be :full, :half, or :quarter"))

    rng = Random.MersenneTwister(seed)

    N = 2^n_qubits
    marked_state = rand(rng, 0:(N-1))

    R_optimal = ceil(Int, π/4 * sqrt(N))
    R = density == :full ? R_optimal : (density == :half ? max(1, div(R_optimal, 2)) : max(1, div(R_optimal, 4)))

    if max_iterations !== nothing
        R = min(R, max_iterations)
    end

    gates = Gate[]
    t_positions = Int[]

    for q in 1:n_qubits
        push!(gates, CliffordGate([(:H, q)], [q]))
    end

    for iteration in 1:R
        marked_bits = digits(marked_state, base=2, pad=n_qubits)

        for q in 1:n_qubits
            marked_bits[q] == 0 && push!(gates, CliffordGate([(:X, q)], [q]))
        end

        push!(gates, CliffordGate([(:H, n_qubits)], [n_qubits]))
        add_multi_controlled_not_large!(gates, t_positions, n_qubits)
        push!(gates, CliffordGate([(:H, n_qubits)], [n_qubits]))

        for q in 1:n_qubits
            marked_bits[q] == 0 && push!(gates, CliffordGate([(:X, q)], [q]))
        end

        for q in 1:n_qubits
            push!(gates, CliffordGate([(:H, q)], [q]))
        end
        for q in 1:n_qubits
            push!(gates, CliffordGate([(:X, q)], [q]))
        end

        push!(gates, CliffordGate([(:H, n_qubits)], [n_qubits]))
        add_multi_controlled_not_large!(gates, t_positions, n_qubits)
        push!(gates, CliffordGate([(:H, n_qubits)], [n_qubits]))

        for q in 1:n_qubits
            push!(gates, CliffordGate([(:X, q)], [q]))
        end
        for q in 1:n_qubits
            push!(gates, CliffordGate([(:H, q)], [q]))
        end
    end

    return (n_qubits=n_qubits, gates=gates, t_positions=t_positions,
            metadata=Dict{String,Any}("family"=>"Grover", "density"=>string(density),
                                       "n_iterations"=>R, "marked_state"=>marked_state,
                                       "n_t_gates"=>length(t_positions)))
end

"""
    generate_vqe_circuit_large(n_qubits; layers=2, seed=42)

Generate VQE hardware-efficient ansatz at arbitrary qubit count.
Same structure as VQEFamily but without n≤8 restriction.

T-count per layer ≈ 55 * n_qubits (one RY per qubit, ~55 T-gates per RY).
Total T-count ≈ 55 * n_qubits * layers.
"""
function generate_vqe_circuit_large(n_qubits::Int; layers::Int=2, seed::Int=42)
    layers >= 1 || throw(ArgumentError("layers must be ≥ 1"))

    rng = Random.MersenneTwister(seed)

    gates = Gate[]
    t_positions = Int[]

    for layer in 1:layers
        for q in 1:n_qubits
            θ = 2π * rand(rng)
            add_ry_gate_vqe!(gates, t_positions, q, θ)
        end
        for q in 1:(n_qubits-1)
            push!(gates, CliffordGate([(:CNOT, q, q+1)], [q, q+1]))
        end
    end

    return (n_qubits=n_qubits, gates=gates, t_positions=t_positions,
            metadata=Dict{String,Any}("family"=>"VQE", "layers"=>layers,
                                       "n_t_gates"=>length(t_positions)))
end

"""
    generate_surface_code_large(n_qubits; n_t_gates=8, seed=42)

Generate Surface Code circuit. The actual physical qubit count depends on
the code distance, set based on n_qubits target.

For large-scale analysis the distance mapping is extended:
- n ≤ 10 → (2,2) = 8 physical qubits
- n ≤ 18 → (3,2) = 13 physical qubits
- n ≤ 32 → (3,3) = 17 physical qubits
- n ≤ 50 → (4,3) = 25 physical qubits
- n ≤ 72 → (4,4) = 31 physical qubits
- n ≤ 100 → (5,4) = 41 physical qubits
- n ≤ 128 → (5,5) = 49 physical qubits
- n > 128 → (6,5) = 61 physical qubits
"""
function generate_surface_code_large(n_qubits::Int; n_t_gates::Int=8, seed::Int=42)
    rng = Random.MersenneTwister(seed)

    dx, dz = if n_qubits <= 10
        (2, 2)
    elseif n_qubits <= 18
        (3, 2)
    elseif n_qubits <= 32
        (3, 3)
    elseif n_qubits <= 50
        (4, 3)
    elseif n_qubits <= 72
        (4, 4)
    elseif n_qubits <= 100
        (5, 4)
    elseif n_qubits <= 128
        (5, 5)
    else
        (6, 5)
    end

    code = Surface(dx, dz)
    n_physical = code_n(code)

    encoding_circ = naive_encoding_circuit(code)
    syndrome_circ, _, _ = naive_syndrome_circuit(code)

    encoding_gates = convert_qc_circuit_to_camps(encoding_circ)
    syndrome_gates = convert_qc_circuit_to_camps(syndrome_circ)

    gates = Gate[]
    t_positions = Int[]

    append!(gates, encoding_gates)
    gate_index = length(gates)

    n_syndrome_rounds = 3 + rand(rng, 0:2)
    t_per_round = distribute_t_gates(n_t_gates, n_syndrome_rounds, rng)

    for round_idx in 1:n_syndrome_rounds
        append!(gates, syndrome_gates)
        gate_index += length(syndrome_gates)

        n_t_this_round = t_per_round[round_idx]
        if n_t_this_round > 0
            n_data = div(n_physical, 2)
            data_range = 1:max(1, n_data)

            qubit_positions = rand(rng, data_range, n_t_this_round)

            for qubit_pos in qubit_positions
                push!(gates, TGate(qubit_pos))
                push!(t_positions, gate_index + 1)
                gate_index += 1
            end
        end
    end

    append!(gates, syndrome_gates)

    return (n_qubits=n_physical, gates=gates, t_positions=t_positions,
            metadata=Dict{String,Any}("family"=>"Surface Code", "dx"=>dx, "dz"=>dz,
                                       "n_physical"=>n_physical, "n_t_gates"=>length(t_positions)))
end

"""
    analyze_circuit_gf2(gates, t_positions, n_qubits; seed=nothing)

Run Clifford-only GF(2) analysis on a circuit. Returns detailed metrics.
"""
function analyze_circuit_gf2(gates::Vector, t_positions::Vector{Int}, n_qubits::Int;
                              seed::Union{Integer, Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    gf2_result = compute_gf2_for_mixed_circuit(
        gates, t_positions, n_qubits;
        seed=seed, simulate_ofd=true)

    t = gf2_result.n_t_gates
    r = gf2_result.gf2_rank
    ν = gf2_result.nullity

    return (
        n_t_gates = t,
        n_total_gates = length(gates),
        gf2_rank = r,
        nullity = ν,
        nullity_ratio = t > 0 ? ν / t : NaN,
        rank_ratio = n_qubits > 0 ? r / n_qubits : NaN,
        rank_to_t_ratio = t > 0 ? r / t : NaN,
        t_density = n_qubits > 0 ? t / n_qubits : NaN,
        xbit_density = gf2_result.xbit_density,
        n_disentanglable = gf2_result.n_disentanglable,
        n_not_disentanglable = gf2_result.n_not_disentanglable
    )
end

"""
    generate_scaling_experiments(mode)

Generate the list of (family, n_qubits, params) experiments for scaling analysis.
"""
function generate_scaling_experiments(mode::String)
    experiments = []

    n_range = if mode == "test"
        [4, 8, 16]
    elseif mode == "medium"
        [4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64]
    else
        [4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256]
    end

    n_realizations = mode == "test" ? 2 : 4

    seed_base = 100000

    phase1_families = [
        ("Random Clifford+T (Brick-wall)", :brick),
        ("Random Clifford+T (All-to-all)", :a2a),
        ("Bernstein-Vazirani", :bv),
        ("Simon's Algorithm", :simon),
        ("Deutsch-Jozsa", :dj),
        ("GHZ State", :ghz),
        ("Bell State / EPR Pairs", :bell),
        ("Graph State", :graph),
        ("Cluster State (1D)", :cluster),
    ]

    for (family_name, tag) in phase1_families
        for n in n_range
            if tag == :simon && n % 2 != 0
                continue
            end

            for real in 1:n_realizations
                seed = Int(hash((tag, n, real)) % UInt32)

                params = Dict{Symbol, Any}(
                    :n_qubits => n,
                    :n_t_gates => n,
                    :seed => seed
                )

                if tag == :brick
                    params[:clifford_depth] = 2
                elseif tag == :a2a
                    params[:clifford_layers] = 2 * n
                elseif tag == :dj
                    rng_temp = Random.MersenneTwister(seed)
                    params[:function_type] = rand(rng_temp, [:constant, :balanced])
                elseif tag == :graph
                    params[:edge_probability] = 0.3
                end

                push!(experiments, (
                    family_name = family_name,
                    n_qubits = n,
                    generator = :phase1,
                    params = params
                ))
            end
        end
    end

    for n in n_range
        for real in 1:n_realizations
            seed = Int(hash(("QFT_scaling", n, real)) % UInt32)
            push!(experiments, (
                family_name = "Quantum Fourier Transform",
                n_qubits = n,
                generator = :qft_large,
                params = Dict{Symbol,Any}(:n_qubits => n, :density => :low, :seed => seed)
            ))
        end
    end

    grover_n_max = mode == "test" ? 8 : (mode == "medium" ? 16 : 20)
    grover_n_range = filter(n -> n <= grover_n_max, n_range)

    for n in grover_n_range
        for real in 1:n_realizations
            seed = Int(hash(("Grover_scaling", n, real)) % UInt32)
            max_iter = n <= 12 ? nothing : min(50, ceil(Int, π/4 * sqrt(2^n)))
            push!(experiments, (
                family_name = "Grover Search",
                n_qubits = n,
                generator = :grover_large,
                params = Dict{Symbol,Any}(:n_qubits => n, :density => :full, :seed => seed,
                                           :max_iterations => max_iter)
            ))
        end
    end

    for n in n_range
        for real in 1:n_realizations
            seed = Int(hash(("VQE_scaling", n, real)) % UInt32)
            push!(experiments, (
                family_name = "VQE Hardware-Efficient Ansatz",
                n_qubits = n,
                generator = :vqe_large,
                params = Dict{Symbol,Any}(:n_qubits => n, :layers => 2, :seed => seed)
            ))
        end
    end

    qaoa_n_range = filter(n -> n % 2 == 0 && n >= 4, n_range)
    for n in qaoa_n_range
        for real in 1:n_realizations
            seed = Int(hash(("QAOA_scaling", n, real)) % UInt32)
            push!(experiments, (
                family_name = "QAOA MaxCut (p=1, 3-regular)",
                n_qubits = n,
                generator = :phase1,
                params = Dict{Symbol,Any}(:n_qubits => n, :n_t_gates => div(5n, 2),
                                           :seed => seed, :gamma => π/4, :beta => π/8)
            ))
        end
    end

    surface_n_range = filter(n -> n >= 8, n_range)
    for n in surface_n_range
        for real in 1:n_realizations
            seed = Int(hash(("Surface_scaling", n, real)) % UInt32)
            n_t = max(4, div(n, 2))
            push!(experiments, (
                family_name = "Surface Code",
                n_qubits = n,
                generator = :surface_large,
                params = Dict{Symbol,Any}(:n_qubits => n, :n_t_gates => n_t, :seed => seed)
            ))
        end
    end

    return experiments
end

"""
    generate_and_analyze(experiment)

Generate a circuit and run GF(2) analysis. Returns a result NamedTuple.
"""
function generate_and_analyze(experiment)
    family_name = experiment.family_name
    params = experiment.params
    gen = experiment.generator

    try
        Random.seed!(params[:seed])

        circuit_result = if gen == :qft_large
            generate_qft_circuit_large(params[:n_qubits];
                density=params[:density], seed=params[:seed])

        elseif gen == :grover_large
            generate_grover_circuit_large(params[:n_qubits];
                density=params[:density], seed=params[:seed],
                max_iterations=get(params, :max_iterations, nothing))

        elseif gen == :vqe_large
            generate_vqe_circuit_large(params[:n_qubits];
                layers=params[:layers], seed=params[:seed])

        elseif gen == :surface_large
            generate_surface_code_large(params[:n_qubits];
                n_t_gates=params[:n_t_gates], seed=params[:seed])

        elseif gen == :phase1
            family = get_family_from_name(family_name)
            if family isa Union{QFTFamily, GroverFamily, VQEFamily, SurfaceCodeFamily}
                error("Phase2 family used with :phase1 generator")
            elseif family isa QAOAMaxCutCircuit
                generate_circuit(family, params)
            else
                generate_circuit(family, params)
            end
        else
            error("Unknown generator: $gen")
        end

        if circuit_result isa CircuitInstance
            n_qubits_actual = circuit_result.n_qubits
            gates = circuit_result.gates
            t_positions = circuit_result.t_gate_positions
        else
            n_qubits_actual = circuit_result.n_qubits
            gates = circuit_result.gates
            t_positions = circuit_result.t_positions
        end

        Random.seed!(params[:seed])
        if gen == :phase1
            family = get_family_from_name(family_name)
            if family isa QAOAMaxCutCircuit
                generate_circuit(family, params)
            elseif !(family isa Union{QFTFamily, GroverFamily, VQEFamily, SurfaceCodeFamily})
                generate_circuit(family, params)
            end
        end

        gf2 = analyze_circuit_gf2(gates, t_positions, n_qubits_actual; seed=params[:seed])

        return (
            success = true,
            family = family_name,
            n_qubits = n_qubits_actual,
            seed = params[:seed],
            n_t_gates = gf2.n_t_gates,
            n_total_gates = gf2.n_total_gates,
            gf2_rank = gf2.gf2_rank,
            nullity = gf2.nullity,
            nullity_ratio = gf2.nullity_ratio,
            rank_ratio = gf2.rank_ratio,
            rank_to_t_ratio = gf2.rank_to_t_ratio,
            t_density = gf2.t_density,
            xbit_density = gf2.xbit_density,
            n_disentanglable = gf2.n_disentanglable,
            n_not_disentanglable = gf2.n_not_disentanglable
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
            nullity_ratio = NaN,
            rank_ratio = NaN,
            rank_to_t_ratio = NaN,
            t_density = NaN,
            xbit_density = NaN,
            n_disentanglable = 0,
            n_not_disentanglable = 0,
            error_msg = sprint(showerror, e)
        )
    end
end

function write_results_csv(results, output_path)
    open(output_path, "w") do io
        println(io, "success,family,n_qubits,seed,n_t_gates,n_total_gates,gf2_rank,nullity,nullity_ratio,rank_ratio,rank_to_t_ratio,t_density,xbit_density,n_disentanglable,n_not_disentanglable")

        for r in results
            println(io, join([
                r.success,
                "\"$(r.family)\"",
                r.n_qubits,
                r.seed,
                r.n_t_gates,
                r.n_total_gates,
                r.gf2_rank,
                r.nullity,
                isnan(r.nullity_ratio) ? "" : @sprintf("%.6f", r.nullity_ratio),
                isnan(r.rank_ratio) ? "" : @sprintf("%.6f", r.rank_ratio),
                isnan(r.rank_to_t_ratio) ? "" : @sprintf("%.6f", r.rank_to_t_ratio),
                isnan(r.t_density) ? "" : @sprintf("%.4f", r.t_density),
                isnan(r.xbit_density) ? "" : @sprintf("%.6f", r.xbit_density),
                r.n_disentanglable,
                r.n_not_disentanglable
            ], ","))
        end
    end
end

function print_scaling_summary(results)
    successful = filter(r -> r.success, results)
    family_names = sort(unique([r.family for r in successful]))

    println()
    println("="^120)
    println("GF(2) NULL SPACE SCALING — PER-FAMILY SUMMARY")
    println("="^120)
    @printf("%-40s %6s %6s %8s %8s %8s %10s %10s\n",
            "Family", "n", "T", "Rank", "Null", "ν/t", "Rank/n", "T-dens")
    println("-"^120)

    for family in family_names
        fr = filter(r -> r.family == family, successful)
        qubit_sizes = sort(unique([r.n_qubits for r in fr]))

        for n in qubit_sizes
            nr = filter(r -> r.n_qubits == n, fr)
            if isempty(nr)
                continue
            end

            mean_t = mean([r.n_t_gates for r in nr])
            mean_rank = mean([r.gf2_rank for r in nr])
            mean_null = mean([r.nullity for r in nr])
            mean_nr = mean(filter(!isnan, [r.nullity_ratio for r in nr]))
            mean_rn = mean(filter(!isnan, [r.rank_ratio for r in nr]))
            mean_td = mean(filter(!isnan, [r.t_density for r in nr]))

            @printf("%-40s %6d %6.0f %8.1f %8.1f %8.4f %10.4f %10.2f\n",
                    family, n, mean_t, mean_rank, mean_null,
                    isnan(mean_nr) ? 0.0 : mean_nr,
                    isnan(mean_rn) ? 0.0 : mean_rn,
                    isnan(mean_td) ? 0.0 : mean_td)
        end
        println()
    end
    println("="^120)
end

function print_classification_summary(results)
    successful = filter(r -> r.success, results)
    family_names = sort(unique([r.family for r in successful]))

    println()
    println("="^100)
    println("SCALING CLASSIFICATION (based on nullity ratio ν/t at largest n)")
    println("="^100)

    classifications = []

    for family in family_names
        fr = filter(r -> r.family == family, successful)
        if isempty(fr)
            continue
        end

        max_n = maximum([r.n_qubits for r in fr])
        large_n_results = filter(r -> r.n_qubits == max_n, fr)

        mean_nullity_ratio = mean(filter(!isnan, [r.nullity_ratio for r in large_n_results]))
        mean_t = mean([r.n_t_gates for r in large_n_results])
        mean_rank = mean([r.gf2_rank for r in large_n_results])

        class = if mean_nullity_ratio < 0.3
            "OFD-friendly"
        elseif mean_nullity_ratio > 0.7
            "OFD-hostile"
        else
            "Intermediate"
        end

        push!(classifications, (
            family = family,
            class = class,
            max_n = max_n,
            nullity_ratio = mean_nullity_ratio,
            mean_t = mean_t,
            mean_rank = mean_rank
        ))
    end

    sort!(classifications, by=c -> c.nullity_ratio)

    @printf("%-40s %-15s %6s %8s %8s %8s\n",
            "Family", "Class", "max n", "ν/t", "T", "Rank")
    println("-"^100)

    for c in classifications
        @printf("%-40s %-15s %6d %8.4f %8.0f %8.1f\n",
                c.family, c.class, c.max_n, c.nullity_ratio, c.mean_t, c.mean_rank)
    end

    println()
    n_friendly = count(c -> c.class == "OFD-friendly", classifications)
    n_hostile = count(c -> c.class == "OFD-hostile", classifications)
    n_intermediate = count(c -> c.class == "Intermediate", classifications)
    println("Classification: $n_friendly OFD-friendly, $n_intermediate intermediate, $n_hostile OFD-hostile")
    println("="^100)
end

function main()
    mode = length(ARGS) >= 1 ? ARGS[1] : "medium"

    println("="^80)
    println("GF(2) NULL SPACE SCALING ANALYSIS")
    println("="^80)
    println("Mode: $mode")
    println("Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println()

    experiments = generate_scaling_experiments(mode)
    n_total = length(experiments)
    println("Total experiments: $n_total")

    families = unique([e.family_name for e in experiments])
    for fam in sort(families)
        fe = filter(e -> e.family_name == fam, experiments)
        n_sizes = sort(unique([e.n_qubits for e in fe]))
        println("  $fam: $(length(fe)) experiments, n ∈ {$(join(n_sizes, ", "))}")
    end
    println()
    println("Starting GF(2) scaling analysis (Clifford-only, no MPS)...")
    println("-"^80)

    start_time = time()
    results = []

    for (i, exp) in enumerate(experiments)
        t0 = time()
        result = generate_and_analyze(exp)
        dt = time() - t0
        push!(results, result)

        if !result.success
            err_msg = hasproperty(result, :error_msg) ? result.error_msg : "unknown"
            @printf("[%5d/%5d] FAIL %-35s n=%3d (%.1fs) — %s\n",
                    i, n_total, exp.family_name, exp.n_qubits, dt,
                    first(err_msg, 60))
        elseif i % 20 == 0 || i == n_total || dt > 5.0
            elapsed = time() - start_time
            @printf("[%5d/%5d] %-35s n=%3d t=%5d rank=%5d ν/t=%.3f (%.1fs, total %.0fs)\n",
                    i, n_total, exp.family_name, result.n_qubits,
                    result.n_t_gates, result.gf2_rank,
                    isnan(result.nullity_ratio) ? 0.0 : result.nullity_ratio,
                    dt, elapsed)
        end
    end

    total_time = time() - start_time
    println("-"^80)
    @printf("Completed %d experiments in %.1f seconds\n", n_total, total_time)

    n_success = count(r -> r.success, results)
    n_failed = n_total - n_success
    println("Success: $n_success, Failed: $n_failed")

    if n_failed > 0
        println("\nFailed experiments:")
        for r in filter(r -> !r.success, results)
            msg = hasproperty(r, :error_msg) ? r.error_msg : "unknown error"
            println("  $(r.family) n=$(r.n_qubits): $(first(msg, 80))")
        end
    end

    output_dir = joinpath(camps_dir, "results")
    mkpath(output_dir)

    output_csv = joinpath(output_dir, "gf2_scaling_analysis.csv")
    write_results_csv(results, output_csv)
    println("\nResults saved to: $output_csv")

    print_scaling_summary(results)
    print_classification_summary(results)

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end
