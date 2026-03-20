QFT GF(2) Pivot Column Analysis
=================================

For the rank = N-1 proof investigation.

Uses Generator C (qft_circuit from simulation.jl) with exact rotation angles.
For each N, computes the full GF(2) matrix of twisted Pauli x-bits, performs
RREF, and reports:
  1. Which columns have pivots
  2. Which single column is missing (the "free variable")
  3. For N=8: prints the full RREF matrix

This tells us whether the missing dimension has a clean pattern across N,
which determines feasibility of an analytic proof that rank = N-1.

Three scenarios:
  A. Missing column is always column 1 (or always column N) → proof likely easy
  B. Missing column follows some clean pattern → proof feasible
  C. Missing column varies unpredictably → proof hard

Usage:
    julia benchmarks/qft_pivot_analysis.jl
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
    build_qft_gf2_full(n_qubits::Int) -> NamedTuple

Build the complete GF(2) matrix for QFT(N) using Generator C.

Returns the full t×N matrix of x-bit rows, the RREF form, pivot columns,
and the missing (non-pivot) column.

Pipeline (identical to qft_gf2_rank_genC in experiment_qft_large_n_genC.jl):
  1. Generate circuit via qft_circuit(n)
  2. Walk through gates with CAMPSState (Clifford walk + OFD)
  3. At each non-Clifford RotationGate: compute twisted Pauli, extract x-bits
  4. Collect ALL x-bit rows into a t×N matrix
  5. Perform full RREF to identify pivot columns
"""
function build_qft_gf2_full(n_qubits::Int)
    circuit = qft_circuit(n_qubits)

    state = CAMPSState(n_qubits)
    initialize!(state)

    xbit_rows = BitVector[]
    gate_info = NamedTuple[]

    t_count = 0
    n_ofd = 0

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

                P_twisted = compute_twisted_pauli(state, gate.axis, gate.qubit)
                xb = BitVector(xbit(P_twisted))
                push!(xbit_rows, xb)
                push!(gate_info, (t_index=t_count, qubit=gate.qubit, angle=gate.angle))

                control = find_disentangling_qubit(P_twisted, state.free_qubits)
                if control !== nothing
                    D_gates = build_disentangler_gates(P_twisted, control)
                    D_flat = flatten_gate_sequence(D_gates)
                    apply_inverse_gates!(state.clifford, D_flat)
                    state.free_qubits[control] = false
                    n_ofd += 1
                end
            end
        end
    end

    t = length(xbit_rows)
    n = n_qubits
    M = zeros(Bool, t, n)
    for k in 1:t
        for j in 1:n
            M[k, j] = xbit_rows[k][j]
        end
    end

    M_rref = copy(M)
    pivot_cols = Int[]
    current_row = 1

    for col in 1:n
        pivot_row = 0
        for r in current_row:t
            if M_rref[r, col]
                pivot_row = r
                break
            end
        end

        if pivot_row == 0
            continue
        end

        if pivot_row != current_row
            M_rref[current_row, :], M_rref[pivot_row, :] =
                M_rref[pivot_row, :], M_rref[current_row, :]
        end

        for r in 1:t
            if r != current_row && M_rref[r, col]
                M_rref[r, :] .⊻= M_rref[current_row, :]
            end
        end

        push!(pivot_cols, col)
        current_row += 1
    end

    rank = length(pivot_cols)
    all_cols = Set(1:n)
    missing_cols = sort(collect(setdiff(all_cols, Set(pivot_cols))))

    return (
        n_qubits = n_qubits,
        t = t,
        rank = rank,
        nullity = t - rank,
        pivot_cols = pivot_cols,
        missing_cols = missing_cols,
        M_original = M,
        M_rref = M_rref,
        gate_info = gate_info,
        n_ofd = n_ofd
    )
end

"""
Cross-check: also compute rank using the library's gf2_rank to make sure
the RREF implementation agrees.
"""
function verify_rank(M::Matrix{Bool}, expected_rank::Int, n::Int)
    lib_rank = gf2_rank(M)
    if lib_rank != expected_rank
        @printf("  WARNING: RREF rank=%d but gf2_rank=%d — MISMATCH!\n",
                expected_rank, lib_rank)
        return false
    end
    if expected_rank != n - 1
        @printf("  WARNING: rank=%d but expected N-1=%d\n", expected_rank, n - 1)
        return false
    end
    return true
end

function main()
    println("="^90)
    println("QFT GF(2) PIVOT COLUMN ANALYSIS — Generator C (Exact Angles)")
    println("="^90)
    println()
    println("Purpose: Identify which column has no pivot in the RREF of the")
    println("twisted Pauli x-bit matrix. This determines feasibility of proving")
    println("rank = N-1 analytically.")
    println()

    test_ns = [4, 6, 8, 16, 32, 64]

    println("-"^90)
    @printf("%4s %6s %6s %6s   %-20s   %-20s   %6s\n",
            "N", "t", "rank", "N-1?", "Pivot columns", "Missing col(s)", "Verify")
    println("-"^90)

    results = []

    for n in test_ns
        t_start = time()
        result = build_qft_gf2_full(n)
        elapsed = time() - t_start

        ok = verify_rank(result.M_original, result.rank, n)

        if n <= 16
            pivot_str = "[" * join(result.pivot_cols, ",") * "]"
        else
            first_pivots = result.pivot_cols[1:min(5, length(result.pivot_cols))]
            last_pivots = result.pivot_cols[max(1,end-2):end]
            pivot_str = "[" * join(first_pivots, ",") * ",...," * join(last_pivots, ",") * "]"
        end

        missing_str = "[" * join(result.missing_cols, ",") * "]"
        rank_ok = result.rank == n - 1 ? "YES" : "NO"
        verify_str = ok ? "OK" : "FAIL"

        @printf("%4d %6d %6d %6s   %-20s   %-20s   %6s  (%.2fs)\n",
                n, result.t, result.rank, rank_ok, pivot_str, missing_str,
                verify_str, elapsed)

        push!(results, result)
    end
    println("-"^90)

    println()
    println("="^90)
    println("MISSING COLUMN PATTERN ANALYSIS")
    println("="^90)
    println()

    missing_pattern = [r.missing_cols for r in results]
    ns_tested = [r.n_qubits for r in results]

    for (n, mc) in zip(ns_tested, missing_pattern)
        if length(mc) == 1
            col = mc[1]
            if col == 1
                label = "column 1 (first qubit)"
            elseif col == n
                label = "column N (last qubit)"
            else
                label = "column $col (interior qubit)"
            end
            @printf("  N=%3d: missing column = %3d  → %s\n", n, col, label)
        else
            @printf("  N=%3d: missing columns = %s\n", n, mc)
        end
    end

    println()
    all_missing = [mc[1] for mc in missing_pattern if length(mc) == 1]
    if length(all_missing) == length(missing_pattern)
        if all(c -> c == all_missing[1], all_missing)
            println("PATTERN: Missing column is ALWAYS column $(all_missing[1]).")
            if all_missing[1] == 1
                println("→ Scenario A: Column 1 (first qubit) never has a pivot.")
                println("  This likely follows from the boundary structure of QFT.")
                println("  Proof is expected to be STRAIGHTFORWARD.")
            elseif all(c -> c == ns_tested[i] for (i, c) in enumerate(all_missing))
                println("→ Scenario A: Column N (last qubit) never has a pivot.")
                println("  Proof is expected to be STRAIGHTFORWARD.")
            else
                println("→ Scenario B: Fixed column, but not a boundary qubit.")
                println("  Proof is FEASIBLE but requires careful analysis.")
            end
        elseif all(i -> all_missing[i] == ns_tested[i], eachindex(all_missing))
            println("PATTERN: Missing column is always column N (= last qubit).")
            println("→ Scenario A: Proof is expected to be STRAIGHTFORWARD.")
        elseif all(i -> all_missing[i] == 1, eachindex(all_missing))
            println("PATTERN: Missing column is always column 1 (= first qubit).")
            println("→ Scenario A: Proof is expected to be STRAIGHTFORWARD.")
        else
            ratios = [all_missing[i] / ns_tested[i] for i in eachindex(all_missing)]
            if all(r -> abs(r - ratios[1]) < 0.01, ratios)
                @printf("PATTERN: Missing column ≈ %.2f × N\n", ratios[1])
                println("→ Scenario B: Consistent proportional pattern.")
            else
                println("PATTERN: Missing column varies across N:")
                for (n, c) in zip(ns_tested, all_missing)
                    @printf("  N=%d → col %d (col/N = %.2f)\n", n, c, c/n)
                end
                println("→ Scenario C: No obvious pattern. Proof may be HARD.")
            end
        end
    else
        println("WARNING: Some N values have multiple missing columns!")
        println("→ This should not happen if rank = N-1 exactly.")
    end

    println()
    println("="^90)
    println("FULL RREF MATRIX FOR N=8")
    println("="^90)
    println()

    result_8 = nothing
    for r in results
        if r.n_qubits == 8
            result_8 = r
            break
        end
    end

    if result_8 !== nothing
        n = 8
        rk = result_8.rank
        println("Matrix dimensions: $(result_8.t) rows × $n columns")
        println("Rank: $rk (= N-1 = $(n-1))")
        println("Pivot columns: $(result_8.pivot_cols)")
        println("Missing column: $(result_8.missing_cols)")
        println()

        println("RREF (non-zero rows only, $rk rows × $n columns):")
        println()

        @printf("         ")
        for j in 1:n
            @printf("col%d ", j)
        end
        println()
        @printf("         ")
        for j in 1:n
            if j in result_8.pivot_cols
                @printf("  P  ")
            else
                @printf("  *  ")
            end
        end
        println()
        println("         " * "-"^(5*n))

        row_count = 0
        for i in 1:result_8.t
            if any(result_8.M_rref[i, :])
                row_count += 1
                @printf("  row %2d:", row_count)
                for j in 1:n
                    @printf("  %d  ", Int(result_8.M_rref[i, j]))
                end
                println()
            end
        end
        println()

        println("ORIGINAL GF(2) MATRIX (first 20 of $(result_8.t) rows):")
        println()
        @printf("         ")
        for j in 1:n
            @printf("q%d ", j)
        end
        println("  | qubit  angle")
        println("         " * "-"^(3*n) * "  " * "-"^25)

        n_show = min(20, result_8.t)
        for k in 1:n_show
            @printf("  t=%3d: ", k)
            for j in 1:n
                @printf(" %d ", Int(result_8.M_original[k, j]))
            end
            info = result_8.gate_info[k]
            @printf("  | q=%2d  θ=%.4f\n", info.qubit, info.angle)
        end
        if result_8.t > n_show
            println("  ... ($(result_8.t - n_show) more rows)")
        end
    end

    println()
    println("="^90)
    println("FULL RREF MATRIX FOR N=4 (for manual verification)")
    println("="^90)
    println()

    result_4 = nothing
    for r in results
        if r.n_qubits == 4
            result_4 = r
            break
        end
    end

    if result_4 !== nothing
        n = 4
        rk = result_4.rank
        println("Matrix dimensions: $(result_4.t) rows × $n columns")
        println("Rank: $rk")
        println("Pivot columns: $(result_4.pivot_cols)")
        println("Missing column: $(result_4.missing_cols)")
        println()

        println("RREF (non-zero rows):")
        row_count = 0
        for i in 1:result_4.t
            if any(result_4.M_rref[i, :])
                row_count += 1
                @printf("  row %d: [", row_count)
                for j in 1:n
                    @printf("%d", Int(result_4.M_rref[i, j]))
                    j < n && print(" ")
                end
                println("]")
            end
        end
        println()

        println("ALL original rows ($(result_4.t) rows):")
        for k in 1:result_4.t
            info = result_4.gate_info[k]
            @printf("  t=%2d: [", k)
            for j in 1:n
                @printf("%d", Int(result_4.M_original[k, j]))
                j < n && print(" ")
            end
            @printf("]  q=%d θ=%.4f\n", info.qubit, info.angle)
        end
    end

    println()
    println("="^90)
    println("NULL SPACE ANALYSIS")
    println("="^90)
    println()
    println("The null space of M^T tells us which vector v ∈ F₂^N is orthogonal")
    println("to ALL rows. If rank = N-1, there is exactly one such v (up to scaling).")
    println()

    for r in results
        n = r.n_qubits

        Mt = zeros(Bool, n, r.t)
        for i in 1:r.t
            for j in 1:n
                Mt[j, i] = r.M_original[i, j]
            end
        end

        if length(r.missing_cols) == 1
            free_col = r.missing_cols[1]

            v = zeros(Bool, n)
            v[free_col] = true

            pivot_row_idx = 0
            for i in 1:r.t
                if any(r.M_rref[i, :])
                    pivot_row_idx += 1
                    if r.M_rref[i, free_col]
                        pc = r.pivot_cols[pivot_row_idx]
                        v[pc] = true
                    end
                end
            end

            all_zero = true
            for i in 1:r.t
                dot = false
                for j in 1:n
                    dot ⊻= (r.M_original[i, j] & v[j])
                end
                if dot
                    all_zero = false
                    break
                end
            end

            v_str = "[" * join([Int(v[j]) for j in 1:n], ",") * "]"
            verify_str = all_zero ? "VERIFIED" : "FAILED"

            @printf("  N=%3d: null vector v = %-30s  %s\n", n, v_str, verify_str)

            nonzero_positions = findall(v)
            if length(nonzero_positions) == n
                println("         v = [1,1,...,1] (all-ones vector)")
            elseif length(nonzero_positions) == 1
                println("         v = e_$(nonzero_positions[1]) (unit vector)")
            else
                println("         nonzero at positions: $nonzero_positions")
                hw = sum(v)
                @printf("         Hamming weight: %d (out of %d)\n", hw, n)
            end
        end
    end

    println()
    println("="^90)
    println("ANALYSIS COMPLETE")
    println("="^90)

    return results
end

results = main()
