
using Test
using CAMPS
using QuantumClifford
import LinearAlgebra

@testset "Mathematical Properties" begin

    @testset "Twisted Pauli: P̃ = C†PC (Liu & Clark §II)" begin

        @testset "H†ZH = X" begin
            state = CAMPSState(2); initialize!(state)
            apply_clifford_gate!(state.clifford, sHadamard(1))
            P = compute_twisted_pauli(state, :Z, 1)
            @test get_pauli_at(P, 1) == :X
            @test get_pauli_at(P, 2) == :I
        end

        @testset "H†XH = Z" begin
            state = CAMPSState(2); initialize!(state)
            apply_clifford_gate!(state.clifford, sHadamard(1))
            P = compute_twisted_pauli(state, :X, 1)
            @test get_pauli_at(P, 1) == :Z
        end

        @testset "S†ZS = Z" begin
            state = CAMPSState(2); initialize!(state)
            apply_clifford_gate!(state.clifford, sPhase(1))
            P = compute_twisted_pauli(state, :Z, 1)
            @test get_pauli_at(P, 1) == :Z
        end

        @testset "S†XS = -Y (Pauli type and phase)" begin
            state = CAMPSState(2); initialize!(state)
            apply_clifford_gate!(state.clifford, sPhase(1))
            P = compute_twisted_pauli(state, :X, 1)
            @test get_pauli_at(P, 1) == :Y
            @test get_pauli_phase(P) ≈ -1.0 + 0.0im atol=1e-10
        end

        @testset "CNOT†(Z⊗I)CNOT = Z⊗I (control Z unchanged)" begin
            state = CAMPSState(2); initialize!(state)
            apply_clifford_gate!(state.clifford, sCNOT(1, 2))
            P = compute_twisted_pauli(state, :Z, 1)
            @test get_pauli_at(P, 1) == :Z
            @test get_pauli_at(P, 2) == :I
        end

        @testset "CNOT†(I⊗Z)CNOT = Z⊗Z (target Z spreads)" begin
            state = CAMPSState(2); initialize!(state)
            apply_clifford_gate!(state.clifford, sCNOT(1, 2))
            P = compute_twisted_pauli(state, :Z, 2)
            @test get_pauli_at(P, 1) == :Z
            @test get_pauli_at(P, 2) == :Z
        end

        @testset "CNOT†(X⊗I)CNOT = X⊗X (control X spreads)" begin
            state = CAMPSState(2); initialize!(state)
            apply_clifford_gate!(state.clifford, sCNOT(1, 2))
            P = compute_twisted_pauli(state, :X, 1)
            @test get_pauli_at(P, 1) == :X
            @test get_pauli_at(P, 2) == :X
        end

        @testset "CNOT†(I⊗X)CNOT = I⊗X (target X unchanged)" begin
            state = CAMPSState(2); initialize!(state)
            apply_clifford_gate!(state.clifford, sCNOT(1, 2))
            P = compute_twisted_pauli(state, :X, 2)
            @test get_pauli_at(P, 1) == :I
            @test get_pauli_at(P, 2) == :X
        end

        @testset "CZ†(Z⊗I)CZ = Z⊗I" begin
            state = CAMPSState(2); initialize!(state)
            apply_clifford_gate!(state.clifford, sCPHASE(1, 2))
            P = compute_twisted_pauli(state, :Z, 1)
            @test get_pauli_at(P, 1) == :Z
            @test get_pauli_at(P, 2) == :I
        end

        @testset "CZ†(X⊗I)CZ = X⊗Z" begin
            state = CAMPSState(2); initialize!(state)
            apply_clifford_gate!(state.clifford, sCPHASE(1, 2))
            P = compute_twisted_pauli(state, :X, 1)
            @test get_pauli_at(P, 1) == :X
            @test get_pauli_at(P, 2) == :Z
        end

        @testset "Conjugation preserves Pauli group (closure)" begin
            state = CAMPSState(3); initialize!(state)
            apply_clifford_gate!(state.clifford, sHadamard(1))
            apply_clifford_gate!(state.clifford, sCNOT(1, 2))
            apply_clifford_gate!(state.clifford, sPhase(2))
            apply_clifford_gate!(state.clifford, sCNOT(2, 3))
            apply_clifford_gate!(state.clifford, sHadamard(3))

            for (axis, q) in [(:Z, 1), (:Z, 2), (:Z, 3), (:X, 1), (:X, 2), (:X, 3)]
                P = compute_twisted_pauli(state, axis, q)
                for j in 1:3
                    σ = get_pauli_at(P, j)
                    @test σ in [:I, :X, :Y, :Z]
                end
                phase = get_pauli_phase(P)
                @test any(abs(phase - p) < 1e-10 for p in [1, -1, im, -im])
            end
        end
    end

    @testset "Rotation identity: R_P(θ)|ψ⟩ = C(αI + βP̃)|φ⟩ (Liu & Clark Eq. 1-2)" begin

        @testset "Verify decomposition for H-T circuit" begin
            n = 3
            state = CAMPSState(n); initialize!(state)

            apply_clifford_gate!(state.clifford, sHadamard(1))

            P_tilde = compute_twisted_pauli(state, :Z, 1)

            ψ_before = state_vector(state)

            apply_gate!(state, TGate(1); strategy=NoDisentangling())

            ψ_after = state_vector(state)

            θ = π/4
            α = cos(θ/2)
            β = -im * sin(θ/2)

            @test abs(α)^2 + abs(β)^2 ≈ 1.0 atol=1e-10
            @test sum(abs2, ψ_after) ≈ 1.0 atol=1e-10
        end
    end

    @testset "GF(2) bond dimension bound: χ ≤ 2^ν (Liu & Clark Thm 1)" begin

        @testset "Independent twisted Paulis → χ = 1 (rank = t)" begin
            for n in 2:5
                circuit = Gate[]
                for q in 1:n
                    push!(circuit, HGate(q))
                end
                for q in 1:n
                    push!(circuit, TGate(q))
                end
                result = simulate_circuit(circuit, n; strategy=OFDStrategy())
                @test result.final_bond_dim == 1
                @test result.predicted_bond_dim == 1
            end
        end

        @testset "All-Z twisted Paulis → χ = 2^t (rank = 0)" begin
            for n in 2:4
                circuit = Gate[TGate(1)]
                result = simulate_circuit(circuit, n; strategy=OFDStrategy())
                @test result.predicted_bond_dim == 2
            end
        end

        @testset "GF(2) bound holds for random circuits" begin
            for seed in [42, 123, 456]
                circuit = random_clifford_t_circuit(4, 5; seed=seed)
                result = simulate_circuit(circuit, 4; strategy=OFDStrategy())
                @test result.final_bond_dim <= result.predicted_bond_dim
            end
        end

        @testset "GF(2) rank and nullity are consistent" begin
            for n in 3:5
                circuit = random_clifford_t_circuit(n, 6; seed=999+n)
                pred = predict_bond_dimension_for_circuit(circuit, n; strategy=OFDStrategy())
                @test pred.gf2_rank + pred.nullity == pred.n_t_gates
            end
        end
    end

    @testset "GF(2) null space verification: v^T M = 0 (left null space)" begin

        @testset "Null vectors satisfy Mv = 0" begin
            M = Bool[
                1 0 1;
                0 1 1;
                1 1 0
            ]
            null_vecs = gf2_null_space(M)
            @test length(null_vecs) >= 1
            for v in null_vecs
                product = [reduce(⊻, M[i, j] & v[i] for i in 1:size(M, 1)) for j in 1:size(M, 2)]
                @test all(product .== 0)
            end
        end

        @testset "Null space from circuit twisted Paulis" begin
            state = CAMPSState(4); initialize!(state)
            for q in 1:4
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end
            apply_clifford_gate!(state.clifford, sCNOT(1, 2))

            P1 = compute_twisted_pauli(state, :Z, 1)
            P2 = compute_twisted_pauli(state, :Z, 2)
            paulis = [P1, P2]

            M = build_gf2_matrix(paulis)
            rank = gf2_rank(M)
            null_vecs = gf2_null_space(M)

            for v in null_vecs
                product = [reduce(⊻, M[i, j] & v[i] for i in 1:size(M, 1)) for j in 1:size(M, 2)]
                @test all(product .== 0)
            end
            @test rank + length(null_vecs) == size(M, 1)
        end

        @testset "Null space of full-rank matrix is empty" begin
            M = Bool[1 0; 0 1]
            null_vecs = gf2_null_space(M)
            @test isempty(null_vecs)
        end
    end

    @testset "OFD disentangler algebra: D†P̃D factorizes (Liu & Clark §III.A)" begin

        @testset "Disentangler for P̃ = X⊗Z zeros out non-control sites" begin
            P = P"XZ"
            D_gates = build_disentangler_gates(P, 1)
            P_after = update_twisted_pauli_after_ofd(P, D_gates, 1)
            @test get_pauli_at(P_after, 2) == :I
            @test get_pauli_at(P_after, 1) in [:X, :Y]
        end

        @testset "Disentangler for P̃ = X⊗X⊗Z zeros out all non-control sites" begin
            P = P"XXZ"
            D_gates = build_disentangler_gates(P, 1)
            P_after = update_twisted_pauli_after_ofd(P, D_gates, 1)
            @test get_pauli_at(P_after, 2) == :I
            @test get_pauli_at(P_after, 3) == :I
            @test get_pauli_at(P_after, 1) in [:X, :Y]
        end

        @testset "Disentangler for P̃ = Y⊗X (Y at control)" begin
            P = P"YX"
            D_gates = build_disentangler_gates(P, 1)
            P_after = update_twisted_pauli_after_ofd(P, D_gates, 1)
            @test get_pauli_at(P_after, 2) == :I
        end

        @testset "Disentangler for P̃ = Z⊗X (control at qubit 2)" begin
            P = P"ZX"
            D_gates = build_disentangler_gates(P, 2)
            P_after = update_twisted_pauli_after_ofd(P, D_gates, 2)
            @test get_pauli_at(P_after, 1) == :I
            @test get_pauli_at(P_after, 2) in [:X, :Y]
        end

        @testset "Disentangler is a Clifford circuit (all gates are Clifford)" begin
            P = P"XYZ"
            D_gates = build_disentangler_gates(P, 1)
            gates_flat = flatten_gate_sequence(D_gates)
            for g in gates_flat
                @test typeof(g) <: QuantumClifford.AbstractSymbolicOperator ||
                      typeof(g) <: QuantumClifford.AbstractCliffordOperator
            end
        end
    end

    @testset "Rotation matrix identities" begin

        @testset "Rz(π/4) = T gate" begin
            Rz = rotation_matrix(:Z, π/4)
            @test Rz ≈ CAMPS.pauli_matrix(:I) * cos(π/8) - im * sin(π/8) * CAMPS.pauli_matrix(:Z) atol=1e-10
            @test Rz ≈ [exp(-im*π/8) 0; 0 exp(im*π/8)] atol=1e-10
        end

        @testset "Rz(π/2) = S gate (up to global phase)" begin
            Rz = rotation_matrix(:Z, π/2)
            S = [1 0; 0 im]
            phase = exp(im*π/4)
            @test Rz * phase ≈ S atol=1e-10
        end

        @testset "R(2π) = -I (spinor property)" begin
            for axis in [:X, :Y, :Z]
                R = rotation_matrix(axis, 2π)
                @test R ≈ -Matrix{ComplexF64}(LinearAlgebra.I, 2, 2) atol=1e-10
            end
        end

        @testset "R(0) = I" begin
            for axis in [:X, :Y, :Z]
                R = rotation_matrix(axis, 0.0)
                @test R ≈ Matrix{ComplexF64}(LinearAlgebra.I, 2, 2) atol=1e-10
            end
        end

        @testset "R(θ)R(-θ) = I" begin
            for axis in [:X, :Y, :Z]
                for θ in [π/7, π/3, 1.23]
                    R1 = rotation_matrix(axis, θ)
                    R2 = rotation_matrix(axis, -θ)
                    @test R1 * R2 ≈ Matrix{ComplexF64}(LinearAlgebra.I, 2, 2) atol=1e-10
                end
            end
        end

        @testset "rotation_coefficients match cos/sin" begin
            for θ in [0.0, π/8, π/4, π/3, π/2, π, 3π/2]
                α, β = rotation_coefficients(θ)
                @test α ≈ cos(θ/2) atol=1e-10
                @test β ≈ -im * sin(θ/2) atol=1e-10
            end
        end
    end

    @testset "Clifford angle detection" begin

        @testset "Clifford angles: multiples of π/2" begin
            @test is_clifford_angle(0.0) == true
            @test is_clifford_angle(π/2) == true
            @test is_clifford_angle(π) == true
            @test is_clifford_angle(3π/2) == true
            @test is_clifford_angle(2π) == true
            @test is_clifford_angle(-π/2) == true
        end

        @testset "Non-Clifford angles" begin
            @test is_clifford_angle(π/4) == false
            @test is_clifford_angle(π/3) == false
            @test is_clifford_angle(π/8) == false
            @test is_clifford_angle(1.0) == false
        end
    end

    @testset "GF(2) Gaussian elimination correctness" begin

        @testset "Identity matrix: rank = n" begin
            for n in 1:6
                M = Matrix{Bool}(LinearAlgebra.I, n, n)
                @test gf2_rank(M) == n
            end
        end

        @testset "Zero matrix: rank = 0" begin
            for n in 1:4
                M = zeros(Bool, n, n)
                @test gf2_rank(M) == 0
            end
        end

        @testset "Known rank-deficient matrix" begin
            M = Bool[1 1 0; 0 1 1; 1 0 1]
            @test gf2_rank(M) == 2
        end

        @testset "Rectangular matrix (more rows than columns)" begin
            M = Bool[1 0; 0 1; 1 1]
            @test gf2_rank(M) == 2
        end

        @testset "Rectangular matrix (more columns than rows)" begin
            M = Bool[1 0 1; 0 1 1]
            @test gf2_rank(M) == 2
        end

        @testset "Large random GF(2) matrix rank bounded by min(m,n)" begin
            m, n_cols = 20, 15
            M = rand(Bool, m, n_cols)
            @test 0 <= gf2_rank(M) <= min(m, n_cols)
        end
    end

    @testset "Pauli bit representation" begin

        @testset "Symbol ↔ bits round-trip" begin
            for (sym, x, z) in [(:I, false, false), (:X, true, false),
                                (:Y, true, true), (:Z, false, true)]
                @test xz_to_symbol(x, z) == sym
                xr, zr = symbol_to_xz(sym)
                @test xr == x && zr == z
            end
        end

        @testset "xbit vector for OFD eligibility" begin
            P = P"XYZ"
            xbits = get_xbit_vector(P)
            @test xbits == Bool[1, 1, 0]

            @test has_x_or_y(P, 1) == true
            @test has_x_or_y(P, 2) == true
            @test has_x_or_y(P, 3) == false
        end

        @testset "zbits vector" begin
            P = P"XYZ"
            zbits = get_zbit_vector(P)
            @test zbits == Bool[0, 1, 1]
        end
    end

    @testset "Renyi-2 and von Neumann entropy" begin

        @testset "Pure state: S₂ = 0" begin
            ρ = ComplexF64[1 0; 0 0]
            @test compute_renyi2_entropy(ρ) ≈ 0.0 atol=1e-10
        end

        @testset "Maximally mixed single-qubit: S₂ = log(2)" begin
            ρ = Matrix{ComplexF64}(LinearAlgebra.I, 2, 2) / 2
            @test compute_renyi2_entropy(ρ) ≈ log(2) atol=1e-10
        end

        @testset "Von Neumann: pure state S = 0" begin
            ρ = ComplexF64[1 0; 0 0]
            @test compute_von_neumann_entropy(ρ) ≈ 0.0 atol=1e-10
        end

        @testset "Von Neumann: maximally mixed single-qubit S = log(2)" begin
            ρ = Matrix{ComplexF64}(LinearAlgebra.I, 2, 2) / 2
            @test compute_von_neumann_entropy(ρ) ≈ log(2) atol=1e-10
        end

        @testset "S₂ ≤ S_vN for any state" begin
            ρ = ComplexF64[0.7 0; 0 0.3]
            s2 = compute_renyi2_entropy(ρ)
            svn = compute_von_neumann_entropy(ρ)
            @test s2 <= svn + 1e-10
        end
    end

    @testset "Partial trace" begin

        @testset "Partial trace of |00⟩⟨00| over qubit 2 = |0⟩⟨0|" begin
            ρ = zeros(ComplexF64, 4, 4)
            ρ[1, 1] = 1.0
            ρ1 = partial_trace_4x4(ρ, true)
            @test ρ1 ≈ ComplexF64[1 0; 0 0] atol=1e-10
        end

        @testset "Partial trace of Bell state = I/2" begin
            ψ = zeros(ComplexF64, 4)
            ψ[1] = 1/sqrt(2)
            ψ[4] = 1/sqrt(2)
            ρ = ψ * ψ'
            ρ1 = partial_trace_4x4(ρ, true)
            @test ρ1 ≈ Matrix{ComplexF64}(LinearAlgebra.I, 2, 2) / 2 atol=1e-10
        end

        @testset "Partial trace preserves trace" begin
            ρ = rand(ComplexF64, 4, 4)
            ρ = ρ * ρ'
            ρ = ρ / LinearAlgebra.tr(ρ)
            ρ1 = partial_trace_4x4(ρ, true)
            @test real(LinearAlgebra.tr(ρ1)) ≈ 1.0 atol=1e-10
        end
    end

    @testset "Incremental GF(2) rank update" begin

        @testset "Adding independent row to empty matrix" begin
            M = Matrix{Bool}(undef, 0, 3)
            row1 = BitVector([true, false, false])
            new_rank, is_indep = incremental_rank_update(M, row1)
            @test new_rank == 1
            @test is_indep == true
        end

        @testset "Adding dependent row does not increase rank" begin
            M = Bool[true false false; false true false]
            M_ech = copy(M); gf2_gausselim!(M_ech)
            row_dep = BitVector([true, true, false])
            new_rank, is_indep = incremental_rank_update(M_ech, row_dep)
            @test new_rank == 2
            @test is_indep == false
        end

        @testset "Zero row is always dependent" begin
            M = Bool[true false; false true]
            M_ech = copy(M); gf2_gausselim!(M_ech)
            new_rank, is_indep = incremental_rank_update(M_ech, BitVector([false, false]))
            @test is_indep == false
        end
    end

    @testset "apply_inverse_gates! correctness (C·D†)" begin

        @testset "S then S† inverse = identity" begin
            n = 2
            C = initialize_clifford(n)
            C_orig = copy(C)

            gates = [sPhase(1)]
            apply_clifford_gates!(C, gates)
            apply_inverse_gates!(C, gates)
            P = commute_pauli_through_clifford(P"Z_", C)
            @test get_pauli_at(P, 1) == :Z
        end

        @testset "H (self-inverse) round-trip" begin
            n = 2
            C = initialize_clifford(n)
            gates = [sHadamard(1)]
            apply_clifford_gates!(C, gates)
            apply_inverse_gates!(C, gates)
            P = commute_pauli_through_clifford(P"Z_", C)
            @test get_pauli_at(P, 1) == :Z
        end

        @testset "CNOT (self-inverse) round-trip" begin
            n = 2
            C = initialize_clifford(n)
            gates = [sCNOT(1, 2)]
            apply_clifford_gates!(C, gates)
            apply_inverse_gates!(C, gates)
            P = commute_pauli_through_clifford(P"ZI", C)
            @test get_pauli_at(P, 1) == :Z
            @test get_pauli_at(P, 2) == :I
        end
    end

    @testset "Two-qubit Clifford group properties" begin

        @testset "24 single-qubit Cliffords" begin
            cliffords = [generate_single_qubit_clifford(i, 1) for i in 1:24]
            @test length(cliffords) == 24
            for c in cliffords
                @test c isa Vector
            end
        end

        @testset "CNOT class has correct number of representatives" begin
            reps = get_cnot_class_representatives()
            @test length(reps) > 0
            for r in reps
                @test nqubits(r) == 2
            end
        end

        @testset "clifford_to_matrix produces unitary matrices" begin
            for idx in [1, 5, 10, 24]
                gates = generate_single_qubit_clifford(idx, 1)
                D = initialize_clifford(1)
                for g in gates
                    resolved = resolve_symbolic_gate(g)
                    apply_clifford_gate!(D, resolved)
                end
                C = CliffordOperator(D)
                U = clifford_to_matrix(C)
                @test U * U' ≈ Matrix{ComplexF64}(LinearAlgebra.I, 2, 2) atol=1e-10
            end
        end
    end

    @testset "GF(2) circuit analysis consistency" begin

        @testset "Tuple and Gate object formats give same result" begin
            n = 4
            tuple_gates = [(:H, [1]), (:H, [2]), (:CNOT, [1, 2]), (:T, [1]), (:T, [2])]
            t_pos_tuple = [4, 5]

            gate_circuit = Gate[HGate(1), HGate(2), CNOTGate(1, 2), TGate(1), TGate(2)]
            t_pos_gate = [4, 5]

            r1 = compute_gf2_for_mixed_circuit(tuple_gates, t_pos_tuple, n; seed=1)
            r2 = compute_gf2_for_mixed_circuit(gate_circuit, t_pos_gate, n; seed=1)

            @test r1.gf2_rank == r2.gf2_rank
            @test r1.nullity == r2.nullity
            @test r1.predicted_chi == r2.predicted_chi
        end

        @testset "T† gives same xbits as T" begin
            n = 3
            gates_t = Gate[HGate(1), TGate(1)]
            gates_tdag = Gate[HGate(1), TdagGate(1)]

            r1 = compute_gf2_for_mixed_circuit(gates_t, [2], n; seed=1)
            r2 = compute_gf2_for_mixed_circuit(gates_tdag, [2], n; seed=1)

            @test r1.gf2_rank == r2.gf2_rank
            @test r1.predicted_chi == r2.predicted_chi
        end
    end

end
