using Test
using CAMPS
using QuantumClifford
using LinearAlgebra

@testset "Phase 4 Tests" begin

    @testset "Clifford Action on Basis States" begin
        @testset "Identity Clifford" begin
            C = initialize_clifford(3)

            for bits in [[0,0,0], [1,0,0], [0,1,0], [1,1,1]]
                output, phase = compute_clifford_action_on_basis_state(C, bits)
                @test output == bits
                @test abs(phase) ≈ 1.0 atol=1e-10
            end
        end

        @testset "Hadamard transformation" begin
            state = CAMPSState(2)
            initialize!(state)
            apply_gate!(state, HGate(1))

            amp_00 = amplitude(state, [0, 0])
            amp_10 = amplitude(state, [1, 0])

            @test abs(amp_00) ≈ 1/√2 atol=1e-10
            @test abs(amp_10) ≈ 1/√2 atol=1e-10
        end

        @testset "CNOT transformation" begin
            state = CAMPSState(2)
            initialize!(state)
            apply_gate!(state, XGate(1))
            apply_gate!(state, CNOTGate(1, 2))

            @test probability(state, [1, 1]) ≈ 1.0 atol=1e-10
            @test probability(state, [0, 0]) ≈ 0.0 atol=1e-10
        end

        @testset "Pauli eigenvalue computation" begin
            n = 3
            Z1 = single_z(n, 1)
            @test pauli_eigenvalue_on_computational_basis(Z1, [0, 0, 0]) == 1

            @test pauli_eigenvalue_on_computational_basis(Z1, [1, 0, 0]) == -1

            Z1Z2 = single_z(n, 1) * single_z(n, 2)
            @test pauli_eigenvalue_on_computational_basis(Z1Z2, [1, 1, 0]) == 1

            X1 = single_x(n, 1)
            @test pauli_eigenvalue_on_computational_basis(X1, [0, 0, 0]) == 0
        end
    end

    @testset "Amplitude Computation" begin
        @testset "Initial state |0⟩^n" begin
            for n in [2, 3, 4]
                state = CAMPSState(n)
                initialize!(state)

                all_zeros = zeros(Int, n)
                @test abs(amplitude(state, all_zeros)) ≈ 1.0 atol=1e-10

                all_ones = ones(Int, n)
                @test abs(amplitude(state, all_ones)) ≈ 0.0 atol=1e-10
            end
        end

        @testset "After Hadamard" begin
            state = CAMPSState(2)
            initialize!(state)
            apply_gate!(state, HGate(1))

            @test abs(amplitude(state, [0, 0])) ≈ 1/√2 atol=1e-10
            @test abs(amplitude(state, [1, 0])) ≈ 1/√2 atol=1e-10
            @test abs(amplitude(state, [0, 1])) ≈ 0.0 atol=1e-10
            @test abs(amplitude(state, [1, 1])) ≈ 0.0 atol=1e-10
        end

        @testset "GHZ state" begin
            state = CAMPSState(3)
            initialize!(state)

            for gate in ghz_circuit(3)
                apply_gate!(state, gate)
            end

            @test abs(amplitude(state, [0, 0, 0])) ≈ 1/√2 atol=1e-10
            @test abs(amplitude(state, [1, 1, 1])) ≈ 1/√2 atol=1e-10
            @test abs(amplitude(state, [1, 0, 0])) ≈ 0.0 atol=1e-10
        end

        @testset "After T gate" begin
            state = CAMPSState(2)
            initialize!(state)
            apply_gate!(state, HGate(1))
            apply_gate!(state, TGate(1))

            amp_00 = amplitude(state, [0, 0])
            amp_10 = amplitude(state, [1, 0])

            @test abs(amp_00) ≈ 1/√2 atol=1e-8
            @test abs(amp_10) ≈ 1/√2 atol=1e-8

            total_prob = abs2(amp_00) + abs2(amp_10)
            @test total_prob ≈ 1.0 atol=1e-8
        end
    end

    @testset "State Vector Extraction" begin
        @testset "Initial state" begin
            state = CAMPSState(3)
            initialize!(state)

            psi = state_vector(state)
            @test length(psi) == 8
            @test abs(psi[1]) ≈ 1.0 atol=1e-10
            @test all(abs.(psi[2:end]) .< 1e-10)
        end

        @testset "After H on all qubits" begin
            state = CAMPSState(2)
            initialize!(state)
            apply_gate!(state, HGate(1))
            apply_gate!(state, HGate(2))

            psi = state_vector(state)
            @test all(abs.(psi) .≈ 0.5)
        end

        @testset "Normalization" begin
            for n in [2, 3, 4]
                state = CAMPSState(n)
                initialize!(state)
                apply_gate!(state, HGate(1))
                if n > 1
                    apply_gate!(state, CNOTGate(1, 2))
                end

                psi = state_vector(state)
                @test sum(abs2.(psi)) ≈ 1.0 atol=1e-10
            end
        end

        @testset "Sparse state vector" begin
            state = CAMPSState(4)
            initialize!(state)

            sparse_psi = state_vector_sparse(state)
            @test length(sparse_psi) == 1
            @test haskey(sparse_psi, [0, 0, 0, 0])
            @test abs(sparse_psi[[0, 0, 0, 0]]) ≈ 1.0 atol=1e-10
        end
    end

    @testset "Fidelity Computation" begin
        @testset "Same state fidelity = 1" begin
            state1 = CAMPSState(3)
            initialize!(state1)
            apply_gate!(state1, HGate(1))

            state2 = CAMPSState(3)
            initialize!(state2)
            apply_gate!(state2, HGate(1))

            F = fidelity(state1, state2)
            @test F ≈ 1.0 atol=1e-10
        end

        @testset "Orthogonal states fidelity = 0" begin
            state1 = CAMPSState(2)
            initialize!(state1)

            state2 = CAMPSState(2)
            initialize!(state2)
            apply_gate!(state2, XGate(1))

            F = fidelity(state1, state2)
            @test F ≈ 0.0 atol=1e-10
        end

        @testset "Overlap computation" begin
            state1 = CAMPSState(2)
            initialize!(state1)
            apply_gate!(state1, HGate(1))

            state2 = CAMPSState(2)
            initialize!(state2)

            ov = overlap(state1, state2)
            @test abs(ov) ≈ 1/√2 atol=1e-10
        end

        @testset "Fidelity with target vector" begin
            state = CAMPSState(2)
            initialize!(state)
            apply_gate!(state, HGate(1))

            target = ComplexF64[1/√2, 1/√2, 0, 0]
            F = fidelity_with_target(state, target)
            @test F ≈ 1.0 atol=1e-10
        end
    end

    @testset "Standard Circuit Generators" begin
        @testset "GHZ circuit" begin
            for n in [2, 3, 4, 5]
                circuit = ghz_circuit(n)
                @test length(circuit) == n

                @test all(g -> g isa CliffordGate, circuit)

                result = simulate_circuit(circuit, n)
                @test result.n_non_clifford == 0
                @test result.final_bond_dim == 1
            end
        end

        @testset "QFT circuit" begin
            for n in [2, 3, 4]
                circuit = qft_circuit(n)
                @test length(circuit) > 0

                if n > 2
                    @test any(g -> g isa RotationGate, circuit)
                end
            end
        end

        @testset "Inverse QFT" begin
            n = 3
            qft = qft_circuit(n)
            inv_qft = inverse_qft_circuit(n)

            @test length(qft) == length(inv_qft)
        end

        @testset "W state circuit" begin
            for n in [2, 3, 4]
                circuit = w_state_circuit(n)
                @test length(circuit) > 0
            end
        end

        @testset "Random T-depth circuit" begin
            n = 4
            t_depth = 3
            circuit = random_t_depth_circuit(n, t_depth; seed=42)

            t_count = count(g -> g isa RotationGate && abs(g.angle) ≈ π/4, circuit)
            @test t_count > 0

            circuit2 = random_t_depth_circuit(n, t_depth; seed=42)
            @test length(circuit) == length(circuit2)
        end

        @testset "Hardware efficient ansatz" begin
            n = 4
            layers = 2
            circuit = hardware_efficient_ansatz(n, layers)

            @test any(g -> g isa RotationGate, circuit)

            @test any(g -> g isa CliffordGate, circuit)

            angles = fill(π/4, 2*n*layers)
            circuit_custom = hardware_efficient_ansatz(n, layers; rotation_angles=angles)
            @test length(circuit_custom) > 0
        end
    end

    @testset "Probability Conservation" begin
        @testset "After various circuits" begin
            for n in [2, 3]
                state = CAMPSState(n)
                initialize!(state)

                apply_gate!(state, HGate(1))
                if n > 1
                    apply_gate!(state, CNOTGate(1, 2))
                end
                apply_gate!(state, TGate(1))
                apply_gate!(state, SGate(n))

                total_prob = 0.0
                for i in 0:(2^n - 1)
                    bits = [(i >> j) & 1 for j in 0:(n-1)]
                    total_prob += probability(state, bits)
                end

                @test total_prob ≈ 1.0 atol=1e-8
            end
        end
    end

    @testset "Normalization" begin
        @testset "State norm" begin
            state = CAMPSState(3)
            initialize!(state)
            apply_gate!(state, HGate(1))
            apply_gate!(state, TGate(1))

            @test CAMPS.norm(state) ≈ 1.0 atol=1e-10
        end

        @testset "Normalize function" begin
            state = CAMPSState(2)
            initialize!(state)

            apply_gate!(state, HGate(1))
            CAMPS.normalize!(state)
            @test CAMPS.norm(state) ≈ 1.0 atol=1e-10
        end
    end

    @testset "Circuit Simulation Integration" begin
        @testset "Clifford-only circuit" begin
            circuit = Gate[
                HGate(1), HGate(2), HGate(3),
                CNOTGate(1, 2), CNOTGate(2, 3),
                SGate(1), SGate(2)
            ]

            result = simulate_circuit(circuit, 3)

            @test result.n_qubits == 3
            @test result.n_gates == 7
            @test result.n_clifford == 7
            @test result.n_non_clifford == 0
            @test result.final_bond_dim >= 1
        end

        @testset "Clifford+T circuit" begin
            circuit = Gate[
                HGate(1), HGate(2),
                TGate(1), TGate(2),
                CNOTGate(1, 2)
            ]

            result = simulate_circuit(circuit, 2; strategy=OFDStrategy())

            @test result.n_non_clifford == 2
            @test result.final_bond_dim >= 1
        end

        @testset "QFT simulation" begin
            n = 3
            circuit = qft_circuit(n)
            result = simulate_circuit(circuit, n; strategy=HybridStrategy())

            @test result.n_qubits == n
            @test is_initialized(result.state)
        end
    end

    @testset "Edge Cases" begin
        @testset "Single qubit" begin
            state = CAMPSState(1)
            initialize!(state)

            @test amplitude(state, [0]) ≈ 1.0 atol=1e-10

            apply_gate!(state, HGate(1))
            apply_gate!(state, TGate(1))

            total_prob = probability(state, [0]) + probability(state, [1])
            @test total_prob ≈ 1.0 atol=1e-8
        end

        @testset "Empty circuit" begin
            result = simulate_circuit(Gate[], 3)
            @test result.n_gates == 0
            @test result.final_bond_dim == 1
        end

        @testset "Large T count" begin
            n = 4
            circuit = Gate[]
            for q in 1:n
                push!(circuit, HGate(q))
            end
            for _ in 1:8
                for q in 1:n
                    push!(circuit, TGate(q))
                end
            end

            result = simulate_circuit(circuit, n; strategy=OFDStrategy())
            @test result.n_non_clifford == 8 * n
            @test is_initialized(result.state)
        end
    end

end
