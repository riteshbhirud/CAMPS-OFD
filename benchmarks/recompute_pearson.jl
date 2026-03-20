Recompute Pearson Correlation: Old QFT (Gen A/B) vs New QFT (Gen C)
====================================================================

The paper's scatter plot shows GF(2) rank/T-count (x-axis) vs
empirical OFD success rate (y-axis) for all 14 circuit families.

Data source: results/experiment_cross_validation.csv
  - 936 circuits, all 14 families
  - Contains both actual_ofd_success/fail (from benchmark) and gf2_rank (from GF(2))
  - Allows computing: rank_to_t_ratio = gf2_rank / n_t_gates
  -                    ofd_rate = actual_ofd_success / (actual_ofd_success + actual_ofd_fail)

This script:
  1. Loads the cross-validation data → computes rank/T and ofd_rate for all 936 circuits
  2. Computes old Pearson r (with Gen A/B QFT data, which is what's in the CSV)
  3. Replaces QFT rows with Generator C data from validate_qft_genC.csv
  4. Computes new Pearson r
  5. Reports both

For Generator C QFT (from validate_qft_genC.csv):
  - rank = N-1 (proven)
  - rank_to_t_ratio = (N-1) / t
  - ofd_rate = ofd_success / (ofd_success + ofd_fail)  [from actual simulation]

Usage:
    julia benchmarks/recompute_pearson.jl
=

camps_dir = dirname(dirname(@__FILE__))
println("Activating CAMPS.jl environment: $camps_dir")
using Pkg
Pkg.activate(camps_dir)

using Statistics
using Printf

function parse_csv_line(line::String)
    fields = String[]
    current = IOBuffer()
    in_quotes = false

    for ch in line
        if ch == '"'
            in_quotes = !in_quotes
        elseif ch == ',' && !in_quotes
            push!(fields, String(take!(current)))
        else
            write(current, ch)
        end
    end
    push!(fields, String(take!(current)))

    return fields
end

function load_csv(path::String)
    lines = readlines(path)
    header = parse_csv_line(lines[1])
    col_idx = Dict(h => i for (i, h) in enumerate(header))

    rows = Vector{Dict{String,String}}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = parse_csv_line(line)
        row = Dict{String,String}()
        for (i, h) in enumerate(header)
            if i <= length(fields)
                row[h] = fields[i]
            else
                row[h] = ""
            end
        end
        push!(rows, row)
    end

    return rows
end

function main()
    results_dir = joinpath(camps_dir, "results")

    cv_path = joinpath(results_dir, "experiment_cross_validation.csv")
    cv_rows = load_csv(cv_path)

    data_old = NamedTuple[]
    for r in cv_rows
        lowercase(r["success"]) == "true" || continue

        family = r["family"]
        n_qubits = parse(Int, r["n_qubits"])
        n_t = parse(Int, r["n_t_gates"])
        actual_ofd_s = parse(Int, r["actual_ofd_success"])
        actual_ofd_f = parse(Int, r["actual_ofd_fail"])
        gf2_rank = parse(Int, r["gf2_rank"])
        seed = parse(Int, r["seed"])

        total_non_cliff = actual_ofd_s + actual_ofd_f
        ofd_rate = total_non_cliff > 0 ? actual_ofd_s / total_non_cliff : NaN
        rank_to_t = n_t > 0 ? gf2_rank / n_t : NaN

        n_t == 0 && continue
        isnan(rank_to_t) && continue

        push!(data_old, (
            family = family,
            n_qubits = n_qubits,
            seed = seed,
            n_t_gates = n_t,
            gf2_rank = gf2_rank,
            actual_ofd_success = actual_ofd_s,
            actual_ofd_fail = actual_ofd_f,
            ofd_rate = ofd_rate,
            rank_to_t_ratio = rank_to_t
        ))
    end

    println("Cross-validation circuits (with T>0): $(length(data_old))")

    families = sort(unique([r.family for r in data_old]))
    println("Families: $(length(families))")
    for f in families
        n_f = count(r -> r.family == f, data_old)
        is_qft = contains(f, "Fourier")
        marker = is_qft ? " ← QFT (Gen A/B)" : ""
        @printf("  %-40s %4d circuits%s\n", f, n_f, marker)
    end
    println()

    x_old = Float64[r.rank_to_t_ratio for r in data_old]
    y_old = Float64[r.ofd_rate for r in data_old]

    n_old = length(x_old)
    r_old = cor(x_old, y_old)

    println("="^80)
    println("OLD PEARSON CORRELATION (Generator A/B QFT in cross-validation)")
    println("="^80)
    @printf("  n = %d circuits\n", n_old)
    @printf("  Pearson r = %.6f\n", r_old)
    @printf("  R² = %.6f\n", r_old^2)

    qft_old = filter(r -> contains(r.family, "Fourier"), data_old)
    println("\n  QFT rows (Gen A/B): $(length(qft_old)) circuits")
    for r in qft_old[1:min(5, length(qft_old))]
        @printf("    N=%d: rank=%d, T=%d, rank/T=%.4f, ofd_rate=%.4f (ofd=%d/%d)\n",
                r.n_qubits, r.gf2_rank, r.n_t_gates, r.rank_to_t_ratio,
                r.ofd_rate, r.actual_ofd_success, r.actual_ofd_success + r.actual_ofd_fail)
    end
    if length(qft_old) > 5
        println("    ... ($(length(qft_old) - 5) more)")
    end

    println("\n  QFT per-N summary (Gen A/B):")
    qft_ns = sort(unique([r.n_qubits for r in qft_old]))
    for n in qft_ns
        rows_n = filter(r -> r.n_qubits == n, qft_old)
        mean_rt = mean([r.rank_to_t_ratio for r in rows_n])
        mean_ofd = mean([r.ofd_rate for r in rows_n])
        @printf("    N=%d: %d circuits, mean rank/T=%.4f, mean ofd_rate=%.4f\n",
                n, length(rows_n), mean_rt, mean_ofd)
    end

    genC_path = joinpath(results_dir, "validate_qft_genC.csv")
    genC_rows = load_csv(genC_path)

    println()
    println("="^80)
    println("GENERATOR C QFT DATA (from validate_qft_genC.csv)")
    println("="^80)
    println()

    genC_data = NamedTuple[]
    for r in genC_rows
        n = tryparse(Int, r["n_qubits"])
        t = tryparse(Int, r["t_count"])
        ofd_s = tryparse(Int, r["ofd_success"])
        ofd_f = tryparse(Int, r["ofd_fail"])
        ofd_rate_str = tryparse(Float64, r["ofd_rate"])
        (n === nothing || t === nothing) && continue

        rank = n - 1
        rank_to_t = t > 0 ? rank / t : NaN
        total = ofd_s + ofd_f
        ofd_rate = total > 0 ? ofd_s / total : NaN

        push!(genC_data, (
            n_qubits = n,
            t_count = t,
            rank = rank,
            ofd_success = ofd_s,
            ofd_fail = ofd_f,
            ofd_rate = ofd_rate,
            rank_to_t_ratio = rank_to_t
        ))

        @printf("  N=%2d: t=%3d, rank=%2d, rank/T=%.6f, ofd_rate=%.6f (ofd=%d/%d)\n",
                n, t, rank, rank_to_t, ofd_rate, ofd_s, total)
    end

    data_no_qft = filter(r -> !contains(r.family, "Fourier"), data_old)
    n_removed = length(data_old) - length(data_no_qft)

    x_new = Float64[r.rank_to_t_ratio for r in data_no_qft]
    y_new = Float64[r.ofd_rate for r in data_no_qft]

    for r in genC_data
        push!(x_new, r.rank_to_t_ratio)
        push!(y_new, r.ofd_rate)
    end

    n_new = length(x_new)
    r_new = cor(x_new, y_new)

    println()
    println("="^80)
    println("NEW PEARSON CORRELATION (Generator C QFT)")
    println("="^80)
    @printf("  n = %d circuits (%d non-QFT + %d QFT Gen C)\n",
            n_new, length(data_no_qft), length(genC_data))
    @printf("  Removed: %d old QFT rows (Gen A/B)\n", n_removed)
    @printf("  Added:   %d new QFT rows (Gen C, N=4..16)\n", length(genC_data))
    @printf("  Pearson r = %.6f\n", r_new)
    @printf("  R² = %.6f\n", r_new^2)

    println()
    println("="^80)
    println("COMPARISON")
    println("="^80)
    println()
    @printf("  OLD (Gen A/B QFT): r = %.4f, R² = %.4f  (n=%d, %d QFT rows)\n",
            r_old, r_old^2, n_old, length(qft_old))
    @printf("  NEW (Gen C QFT):   r = %.4f, R² = %.4f  (n=%d, %d QFT rows)\n",
            r_new, r_new^2, n_new, length(genC_data))
    @printf("\n  Δr  = %+.4f\n", r_new - r_old)
    @printf("  ΔR² = %+.4f\n", r_new^2 - r_old^2)
    println()

    if r_new > r_old
        println("  → Correlation IMPROVED with Generator C QFT.")
    elseif r_new < r_old
        println("  → Correlation DECREASED slightly with Generator C QFT.")
    else
        println("  → Correlation UNCHANGED.")
    end

    println()
    println("  Why the change:")
    println("  Old Gen A/B QFT: ofd_rate=0.0 for ALL circuits (OFD never succeeded)")
    println("    but rank/T > 0 (e.g., 0.125 at N=4). These points sat at")
    println("    (x=0.125, y=0.0), far BELOW the y=x line, dragging r down.")
    println()
    println("  New Gen C QFT: ofd_rate = rank/T EXACTLY (all points on y=x line).")
    println("    rank/T = ofd_rate = (N-1)/t = 2/(3N), which decreases with N")
    println("    but always lies exactly on the diagonal.")

    println()
    println("="^80)
    println("PER-FAMILY SUMMARY (all data, for completeness)")
    println("="^80)
    println()
    @printf("%-40s %6s %10s %10s\n",
            "Family", "Count", "mean rank/T", "mean OFD%")
    println("-"^70)

    data_new_all = copy(data_no_qft)
    for r in genC_data
        push!(data_new_all, (
            family = "Quantum Fourier Transform (Gen C)",
            n_qubits = r.n_qubits,
            seed = 0,
            n_t_gates = r.t_count,
            gf2_rank = r.rank,
            actual_ofd_success = r.ofd_success,
            actual_ofd_fail = r.ofd_fail,
            ofd_rate = r.ofd_rate,
            rank_to_t_ratio = r.rank_to_t_ratio
        ))
    end

    new_families = sort(unique([r.family for r in data_new_all]))
    for f in new_families
        fr = filter(r -> r.family == f, data_new_all)
        n_f = length(fr)
        mean_rt = mean([r.rank_to_t_ratio for r in fr])
        mean_ofd = mean([r.ofd_rate for r in fr])
        @printf("%-40s %6d %10.4f %10.4f\n", f, n_f, mean_rt, mean_ofd)
    end
    println("-"^70)

    println()
    println("="^80)
    println("VERIFICATION: Gen C QFT rank/T = ofd_rate")
    println("="^80)
    println()
    for r in genC_data
        diff = abs(r.rank_to_t_ratio - r.ofd_rate)
        match_str = diff < 1e-4 ? "MATCH" : "MISMATCH"
        @printf("  N=%2d: rank/T=%.6f, ofd_rate=%.6f, |diff|=%.2e  %s\n",
                r.n_qubits, r.rank_to_t_ratio, r.ofd_rate, diff, match_str)
    end

    println()
    println("="^80)
    println("COMPLETE")
    println("="^80)
end

main()
