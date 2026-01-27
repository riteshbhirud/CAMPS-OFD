using Test
using CAMPS
using QuantumClifford

@testset "Phase 2 Integration Tests" begin

    @testset "Complete CAMPS workflow" begin
        n = 4
        state = CAMPSState(n)
        initialize!(state)

        @test get_bond_dimension(state) == 1
        @test num_free_qubits(state) == n
        @test num_magic_qubits(state) == 0
        @test num_twisted_paulis(state) == 0

        for q in 1:n
            apply_clifford_gate!(state.clifford, sHadamard(q))
        end

        @test get_bond_dimension(state) == 1

        P_twisted = compute_twisted_pauli(state, :Z, 1)
        @test get_pauli_at(P_twisted, 1) == :X
        @test pauli_weight(P_twisted) == 1

        @test has_x_or_y(P_twisted, 1) == true
    end

    @testset "Twisted Pauli tracking" begin
        n = 3
        state = CAMPSState(n)
        initialize!(state)

        apply_clifford_gate!(state.clifford, sHadamard(1))

        P1 = compute_twisted_pauli(state, :Z, 1)
        add_twisted_pauli!(state, P1)
        @test num_twisted_paulis(state) == 1

        apply_clifford_gate!(state.clifford, sCNOT(1, 2))

        P2 = compute_twisted_pauli(state, :Z, 2)
        add_twisted_pauli!(state, P2)
        @test num_twisted_paulis(state) == 2

        chi_predicted = get_predicted_bond_dimension(state)
        @test chi_predicted >= 1
    end

    @testset "MPS manipulation with twisted rotations" begin
        n = 2
        mps, sites = initialize_mps(n)

        @test get_mps_bond_dimension(mps) == 1

        P_x = P"X_"
        apply_pauli_string!(mps, P_x, sites)
        @test sample_mps(mps) == [1, 0]

        mps2, sites2 = initialize_mps(n)

        P_z = P"Z_"
        apply_twisted_rotation!(mps2, sites2, P_z, π/4)
        @test get_mps_bond_dimension(mps2) == 1

        mps3, sites3 = initialize_mps(n)
        P_x2 = P"X_"
        apply_twisted_rotation!(mps3, sites3, P_x2, π/2)
        @test get_mps_bond_dimension(mps3) >= 1
    end

    @testset "GF(2) prediction validation" begin
        paulis_independent = [P"X___", P"_X__", P"__X_", P"___X"]
        chi1 = predict_bond_dimension(paulis_independent)
        @test chi1 == 1

        paulis_z = [P"Z___", P"_Z__", P"__Z_"]
        chi2 = predict_bond_dimension(paulis_z)
        @test chi2 == 8

        paulis_dependent = [P"XX_", P"_XX", P"X_X"]
        chi3 = predict_bond_dimension(paulis_dependent)
        @test chi3 == 2
    end

    @testset "Disentanglability analysis" begin
        n = 4

        P_x1 = single_x(n, 1)
        P_x2 = single_x(n, 2)
        P_z1 = single_z(n, 1)

        free_qubits = BitVector([true, true, true, true])

        @test can_disentangle(P_x1, free_qubits) == true
        @test find_disentangling_qubit(P_x1, free_qubits) == 1

        @test can_disentangle(P_z1, free_qubits) == false
        @test find_disentangling_qubit(P_z1, free_qubits) === nothing

        paulis = [P_x1, P_x2, P_z1]
        @test count_disentanglable(paulis, free_qubits) == 2
    end

    @testset "Entanglement entropy" begin
        n = 4
        mps, sites = initialize_mps(n)

        for bond in 1:(n-1)
            S = CAMPS.entanglement_entropy(mps, bond)
            @test S < 1e-10
        end

        entropies = CAMPS.entanglement_entropy_all_bonds(mps)
        @test length(entropies) == n - 1
        @test CAMPS.max_entanglement_entropy(mps) < 1e-10
    end

    @testset "Sampling and probabilities" begin
        n = 3
        mps, sites = initialize_mps(n)

        @test mps_probability(mps, [0, 0, 0], sites) ≈ 1.0 atol=1e-10
        @test mps_probability(mps, [1, 0, 0], sites) ≈ 0.0 atol=1e-10

        @test mps_amplitude(mps, [0, 0, 0], sites) ≈ 1.0 atol=1e-10

        apply_pauli_to_mps!(mps, sites, :X, 2)
        @test mps_probability(mps, [0, 1, 0], sites) ≈ 1.0 atol=1e-10
        @test mps_probability(mps, [0, 0, 0], sites) ≈ 0.0 atol=1e-10
    end

    @testset "Clifford gate resolution" begin
        h = HGate(1)
        gates = resolve_clifford_gate(h)
        @test length(gates) == 1
        @test gates[1] == sHadamard(1)

        cnot = CNOTGate(1, 2)
        gates2 = resolve_clifford_gate(cnot)
        @test gates2[1] == sCNOT(1, 2)

        state = CAMPSState(3)
        initialize!(state)

        apply_clifford_gate_to_state!(state, h)
        P = compute_twisted_pauli(state, :Z, 1)
        @test get_pauli_at(P, 1) == :X
    end

    @testset "Full Clifford+T simulation setup" begin
        n = 5
        state = CAMPSState(n; max_bond=64, cutoff=1e-12)
        initialize!(state)

        for q in 1:n
            apply_clifford_gate!(state.clifford, sHadamard(q))
        end

        for q in 1:3
            P = compute_twisted_pauli(state, :Z, q)
            add_twisted_pauli!(state, P)

            mark_as_magic!(state, q)
            state.free_qubits[q] = false
        end

        @test num_twisted_paulis(state) == 3
        @test num_free_qubits(state) == 2
        @test num_magic_qubits(state) == 3

        chi = get_predicted_bond_dimension(state)
        @test chi == 1

        @test is_initialized(state)
        @test state.clifford !== nothing
        @test state.mps !== nothing
    end
end

@testset "Edge Cases" begin
    @testset "Single qubit system" begin
        state = CAMPSState(1)
        initialize!(state)

        @test get_bond_dimension(state) == 1

        apply_clifford_gate!(state.clifford, sHadamard(1))
        P = compute_twisted_pauli(state, :Z, 1)
        @test get_pauli_at(P, 1) == :X
    end

    @testset "Large system" begin
        n = 20
        state = CAMPSState(n)
        initialize!(state)

        @test n_qubits(state) == n
        @test get_bond_dimension(state) == 1

        for q in 1:n
            apply_clifford_gate!(state.clifford, sHadamard(q))
        end

        P = compute_twisted_pauli(state, :Z, 10)
        @test pauli_weight(P) == 1
    end

    @testset "Empty twisted Pauli list" begin
        @test predict_bond_dimension(PauliOperator[]) == 1

        state = CAMPSState(3)
        initialize!(state)
        @test get_predicted_bond_dimension(state) == 1
    end
end
