using Test
using CAMPS
using QuantumClifford

@testset "OFD Tests" begin

    @testset "Disentangler Gate Construction" begin
        @testset "build_controlled_pauli_gate" begin
            gate_x = build_controlled_pauli_gate(:X, 1, 2)
            @test gate_x == sCNOT(1, 2)

            gate_z = build_controlled_pauli_gate(:Z, 1, 2)
            @test gate_z == sCPHASE(1, 2)

            gate_y = build_controlled_pauli_gate(:Y, 1, 2)
            @test gate_y isa Vector
            @test length(gate_y) == 3
        end

        @testset "build_disentangler_gates" begin
            P_x = P"X__"
            gates = build_disentangler_gates(P_x, 1)
            @test isempty(gates)

            P_xz = P"XZ_"
            gates = build_disentangler_gates(P_xz, 1)
            gates_flat = flatten_gate_sequence(gates)
            @test length(gates_flat) == 1
            @test gates_flat[1] == sCPHASE(1, 2)

            P_xx = P"XX_"
            gates = build_disentangler_gates(P_xx, 1)
            gates_flat = flatten_gate_sequence(gates)
            @test length(gates_flat) == 1
            @test gates_flat[1] == sCNOT(1, 2)

            P_xyz = P"XYZ"
            gates = build_disentangler_gates(P_xyz, 1)
            gates_flat = flatten_gate_sequence(gates)
            @test length(gates_flat) == 4
        end

        @testset "flatten_gate_sequence" begin
            mixed = [[1, 2], 3, [4, 5, 6]]
            flat = flatten_gate_sequence(mixed)
            @test flat == [1, 2, 3, 4, 5, 6]

            simple = [1, 2, 3]
            @test flatten_gate_sequence(simple) == [1, 2, 3]

            @test flatten_gate_sequence([]) == []
        end
    end

    @testset "OFD Application" begin
        @testset "apply_ofd! basic" begin
            state = CAMPSState(3)
            initialize!(state)

            apply_clifford_gate!(state.clifford, sHadamard(1))

            P_twisted = compute_twisted_pauli(state, :Z, 1)
            @test get_pauli_at(P_twisted, 1) == :X

            control = find_disentangling_qubit(P_twisted, state.free_qubits)
            @test control == 1

            apply_ofd!(state, P_twisted, π/4, control)

            @test state.free_qubits[1] == false
            @test state.magic_qubits[1] == true
            @test num_twisted_paulis(state) == 1
            @test get_bond_dimension(state) == 1
        end

        @testset "try_apply_ofd!" begin
            state = CAMPSState(4)
            initialize!(state)

            for q in 1:4
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end

            P_twisted = compute_twisted_pauli(state, :Z, 1)
            success, _ = try_apply_ofd!(state, P_twisted, π/4)
            @test success == true

            P_twisted2 = compute_twisted_pauli(state, :Z, 2)
            success2, _ = try_apply_ofd!(state, P_twisted2, π/4)
            @test success2 == true

            @test num_magic_qubits(state) == 2
            @test num_free_qubits(state) == 2
        end

        @testset "OFD fails when no X/Y available" begin
            state = CAMPSState(3)
            initialize!(state)

            P_twisted = compute_twisted_pauli(state, :Z, 1)
            @test get_pauli_at(P_twisted, 1) == :Z

            success, _ = try_apply_ofd!(state, P_twisted, π/4)
            @test success == false
            @test num_magic_qubits(state) == 0
        end
    end

    @testset "T-gate OFD convenience functions" begin
        @testset "apply_t_gate_ofd!" begin
            state = CAMPSState(5)
            initialize!(state)

            for q in 1:5
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end

            success, _ = apply_t_gate_ofd!(state, 1)
            @test success == true
            @test is_magic(state, 1)
            @test !is_free(state, 1)
        end

        @testset "apply_tdag_gate_ofd!" begin
            state = CAMPSState(3)
            initialize!(state)
            apply_clifford_gate!(state.clifford, sHadamard(1))

            success, _ = apply_tdag_gate_ofd!(state, 1)
            @test success == true
        end
    end

    @testset "OFDS (Sequential OFD)" begin
        @testset "apply_ofds! all succeed" begin
            state = CAMPSState(5)
            initialize!(state)

            for q in 1:5
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end

            result = apply_ofds!(state, [1, 2, 3])

            @test result.num_applied == 3
            @test result.num_failed == 0
            @test result.final_free_qubits == 2
            @test result.final_magic_qubits == 3
            @test result.actual_chi == 1
        end

        @testset "apply_ofds! with entangling" begin
            state = CAMPSState(4)
            initialize!(state)

            for q in 1:4
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end

            apply_clifford_gate!(state.clifford, sCNOT(1, 2))
            apply_clifford_gate!(state.clifford, sCNOT(3, 4))

            result = apply_ofds!(state, [1, 2])

            @test result.num_applied >= 1
        end
    end

    @testset "OFD Analysis" begin
        @testset "analyze_ofd_applicability" begin
            state = CAMPSState(4)
            initialize!(state)

            analysis1 = analyze_ofd_applicability(state, 1)
            @test analysis1.can_apply == false
            @test analysis1.control_qubit === nothing
            @test get_pauli_at(analysis1.twisted_pauli, 1) == :Z

            apply_clifford_gate!(state.clifford, sHadamard(1))
            analysis2 = analyze_ofd_applicability(state, 1)
            @test analysis2.can_apply == true
            @test analysis2.control_qubit == 1
            @test get_pauli_at(analysis2.twisted_pauli, 1) == :X
        end

        @testset "count_ofd_applicable" begin
            state = CAMPSState(6)
            initialize!(state)

            for q in 1:3
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end

            analysis = count_ofd_applicable(state, [1, 2, 3, 4])

            @test analysis.total == 4
            @test analysis.ofd_possible == 3
            @test analysis.ofd_impossible == 1
        end
    end

    @testset "Generate OFD Circuit" begin
        @testset "generate_ofd_circuit" begin
            P = P"XZ"
            circuit = generate_ofd_circuit(P, 1, π/4)

            @test length(circuit) >= 2
            @test circuit[1] isa CliffordGate
            @test circuit[end] isa RotationGate
            @test circuit[end].axis == :X
            @test circuit[end].angle ≈ π/4
        end
    end

    @testset "Magic State Properties" begin
        @testset "t_gate_magic_state" begin
            α, β = t_gate_magic_state()

            @test abs(α)^2 + abs(β)^2 ≈ 1.0 atol=1e-10

            @test abs(α) ≈ cos(π/8) atol=1e-10
            @test abs(β) ≈ sin(π/8) atol=1e-10

            @test abs(real(β)) < 1e-10
        end

        @testset "magic_state_vector" begin
            v_x = magic_state_vector(:X, π/4)
            @test length(v_x) == 2
            @test abs(v_x[1])^2 + abs(v_x[2])^2 ≈ 1.0 atol=1e-10

            v_z = magic_state_vector(:Z, π/4)
            @test abs(v_z[2]) < 1e-10

            v_y = magic_state_vector(:Y, π/2)
            @test abs(v_y[1]) ≈ abs(v_y[2]) atol=1e-10
        end
    end

    @testset "Clifford Update Correctness" begin
        @testset "OFD preserves Clifford structure" begin
            state = CAMPSState(3)
            initialize!(state)

            apply_clifford_gate!(state.clifford, sHadamard(1))

            P_z2_before = compute_twisted_pauli(state, :Z, 2)

            P_t1 = compute_twisted_pauli(state, :Z, 1)
            apply_ofd!(state, P_t1, π/4, 1)

            P_z2_after = compute_twisted_pauli(state, :Z, 2)

            @test get_pauli_at(P_z2_before, 2) == get_pauli_at(P_z2_after, 2)
        end

        @testset "OFD with non-trivial disentangler" begin
            state = CAMPSState(3)
            initialize!(state)

            apply_clifford_gate!(state.clifford, sCNOT(1, 2))
            apply_clifford_gate!(state.clifford, sHadamard(1))
            apply_clifford_gate!(state.clifford, sHadamard(2))

            P_t1 = compute_twisted_pauli(state, :Z, 1)
            @test get_pauli_at(P_t1, 1) == :X
            @test get_pauli_at(P_t1, 2) == :X

            control = find_disentangling_qubit(P_t1, state.free_qubits)
            @test control == 1

            apply_ofd!(state, P_t1, π/4, control)

            @test is_magic(state, 1)
            @test is_initialized(state)
        end
    end

    @testset "GF(2) Prediction Consistency" begin
        @testset "OFD maintains rank structure" begin
            state = CAMPSState(4)
            initialize!(state)

            for q in 1:4
                apply_clifford_gate!(state.clifford, sHadamard(q))
            end

            for q in 1:3
                success, _ = apply_t_gate_ofd!(state, q)
                @test success
            end

            predicted = get_predicted_bond_dimension(state)
            actual = get_bond_dimension(state)

            @test predicted == 1
            @test actual == 1
        end
    end

end
