using Test
using CAMPS
using QuantumClifford
using ITensors
using ITensorMPS

@testset "MPS Initialization" begin
    @testset "initialize_mps" begin
        mps, sites = initialize_mps(3)

        @test mps isa MPS
        @test length(mps) == 3
        @test length(sites) == 3

        @test abs(get_mps_norm(mps) - 1.0) < 1e-10

        @test get_mps_bond_dimension(mps) == 1
    end

    @testset "Different sizes" begin
        for n in [1, 2, 5, 10]
            mps, sites = initialize_mps(n)
            @test length(mps) == n
            @test length(sites) == n
            @test get_mps_bond_dimension(mps) == 1
        end
    end
end

@testset "MPS Properties" begin
    @testset "get_mps_bond_dimension" begin
        mps, sites = initialize_mps(5)
        @test get_mps_bond_dimension(mps) == 1
    end

    @testset "get_mps_norm" begin
        mps, sites = initialize_mps(3)
        @test abs(get_mps_norm(mps) - 1.0) < 1e-10

        mps[1] = 2.0 * mps[1]
        @test abs(get_mps_norm(mps) - 2.0) < 1e-10
    end

    @testset "normalize_mps!" begin
        mps, sites = initialize_mps(3)
        mps[1] = 3.0 * mps[1]

        normalize_mps!(mps)
        @test abs(get_mps_norm(mps) - 1.0) < 1e-10
    end
end

@testset "Gate Tensor Construction" begin
    @testset "pauli_to_itensor" begin
        mps, sites = initialize_mps(2)
        s = sites[1]

        I_gate = pauli_to_itensor(:I, s)
        @test I_gate isa ITensor

        X_gate = pauli_to_itensor(:X, s)
        Y_gate = pauli_to_itensor(:Y, s)
        Z_gate = pauli_to_itensor(:Z, s)

        @test X_gate isa ITensor
        @test Y_gate isa ITensor
        @test Z_gate isa ITensor
    end

    @testset "rotation_to_itensor" begin
        mps, sites = initialize_mps(2)
        s = sites[1]

        T_gate = rotation_to_itensor(:Z, π/4, s)
        @test T_gate isa ITensor

        for axis in [:X, :Y, :Z]
            for θ in [0.0, π/4, π/2, π]
                gate = rotation_to_itensor(axis, θ, s)
                @test gate isa ITensor
            end
        end
    end

    @testset "identity_itensor" begin
        mps, sites = initialize_mps(2)
        s = sites[1]

        I_gate = identity_itensor(s)
        @test I_gate isa ITensor
    end
end

@testset "Single-Site Gate Application" begin
    @testset "apply_local_rotation!" begin
        mps, sites = initialize_mps(2)

        apply_local_rotation!(mps, sites, 1, :Z, π)

        @test abs(get_mps_norm(mps) - 1.0) < 1e-10
    end

    @testset "apply_pauli_to_mps!" begin
        mps, sites = initialize_mps(1)
        apply_pauli_to_mps!(mps, sites, :X, 1)

        sample_result = sample_mps(mps)
        @test sample_result == [1]

        mps2, sites2 = initialize_mps(1)
        apply_pauli_to_mps!(mps2, sites2, :Z, 1)
        sample_result2 = sample_mps(mps2)
        @test sample_result2 == [0]
    end
end

@testset "Pauli String Application" begin
    @testset "apply_pauli_string!" begin
        mps, sites = initialize_mps(3)

        P = P"X__"
        apply_pauli_string!(mps, P, sites)

        sample_result = sample_mps(mps)
        @test sample_result == [1, 0, 0]
    end

    @testset "apply_pauli_string! with multi-qubit" begin
        mps, sites = initialize_mps(3)

        P = P"X_X"
        apply_pauli_string!(mps, P, sites)

        sample_result = sample_mps(mps)
        @test sample_result == [1, 0, 1]
    end

    @testset "apply_pauli_string_to_copy" begin
        mps, sites = initialize_mps(2)

        P = P"XX"
        mps_new = apply_pauli_string_to_copy(mps, P, sites)

        @test sample_mps(mps) == [0, 0]

        @test sample_mps(mps_new) == [1, 1]
    end
end

@testset "Twisted Rotation Application" begin
    @testset "apply_twisted_rotation! basic" begin
        mps, sites = initialize_mps(2)

        P = P"Z_"
        θ = π/4

        apply_twisted_rotation!(mps, sites, P, θ)

        @test abs(get_mps_norm(mps) - 1.0) < 1e-10

        @test get_mps_bond_dimension(mps) == 1
    end

    @testset "apply_twisted_rotation! with X" begin
        mps, sites = initialize_mps(2)

        P = P"X_"
        θ = π/2

        apply_twisted_rotation!(mps, sites, P, θ; max_bond=10)

        @test get_mps_bond_dimension(mps) >= 1

        norm_val = get_mps_norm(mps)
        @test abs(norm_val - 1.0) < 1e-10
    end

    @testset "apply_twisted_rotation_to_copy" begin
        mps, sites = initialize_mps(2)

        P = P"X_"
        θ = π/4

        mps_new = apply_twisted_rotation_to_copy(mps, sites, P, θ)

        @test sample_mps(mps) == [0, 0]

        @test mps_new isa MPS
    end
end

@testset "MPS Truncation" begin
    @testset "truncate_mps!" begin
        mps, sites = initialize_mps(4)

        for i in 1:3
            P = single_x(4, i)
            apply_twisted_rotation!(mps, sites, P, π/3; max_bond=100)
        end

        truncate_mps!(mps; max_bond=2, cutoff=1e-10)

        @test get_mps_bond_dimension(mps) <= 2
    end
end

@testset "Entanglement Entropy" begin
    @testset "Product state entropy" begin
        mps, sites = initialize_mps(4)

        for bond in 1:3
            S = CAMPS.entanglement_entropy(mps, bond)
            @test S < 1e-10
        end
    end

    @testset "entanglement_entropy_all_bonds" begin
        mps, sites = initialize_mps(4)

        entropies = CAMPS.entanglement_entropy_all_bonds(mps)
        @test length(entropies) == 3
        @test all(S -> S < 1e-10, entropies)
    end

    @testset "max_entanglement_entropy" begin
        mps, sites = initialize_mps(4)

        max_S = CAMPS.max_entanglement_entropy(mps)
        @test max_S < 1e-10
    end
end

@testset "MPS Sampling" begin
    @testset "sample_mps" begin
        mps, sites = initialize_mps(3)

        for _ in 1:10
            sample_result = sample_mps(mps)
            @test sample_result == [0, 0, 0]
        end
    end

    @testset "sample_mps after X gate" begin
        mps, sites = initialize_mps(3)

        apply_pauli_to_mps!(mps, sites, :X, 2)

        for _ in 1:10
            sample_result = sample_mps(mps)
            @test sample_result == [0, 1, 0]
        end
    end

    @testset "sample_mps_multiple" begin
        mps, sites = initialize_mps(2)

        samples = sample_mps_multiple(mps, 5)
        @test length(samples) == 5
        @test all(s -> s == [0, 0], samples)
    end
end

@testset "MPS Inner Products" begin
    @testset "mps_overlap" begin
        mps1, sites = initialize_mps(3)
        mps2, _ = initialize_mps(3)

        overlap = mps_overlap(mps1, mps2)
        @test abs(overlap - 1.0) < 1e-10
    end

    @testset "mps_probability" begin
        mps, sites = initialize_mps(3)

        prob = mps_probability(mps, [0, 0, 0], sites)
        @test abs(prob - 1.0) < 1e-10

        prob2 = mps_probability(mps, [1, 0, 0], sites)
        @test prob2 < 1e-10
    end

    @testset "mps_amplitude" begin
        mps, sites = initialize_mps(2)

        amp = mps_amplitude(mps, [0, 0], sites)
        @test abs(amp - 1.0) < 1e-10

        amp2 = mps_amplitude(mps, [1, 1], sites)
        @test abs(amp2) < 1e-10
    end
end

@testset "Two-Qubit Gates" begin
    @testset "matrix_to_two_qubit_itensor" begin
        mps, sites = initialize_mps(3)
        s1, s2 = sites[1], sites[2]

        CNOT_mat = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
        gate = matrix_to_two_qubit_itensor(CNOT_mat, s1, s2)

        @test gate isa ITensor
    end
end

@testset "CAMPSState MPS Operations" begin
    @testset "get_bond_dimension" begin
        state = CAMPSState(5)
        initialize!(state)

        @test get_bond_dimension(state) == 1
    end

    @testset "State initialization check" begin
        state = CAMPSState(3)

        @test !is_initialized(state)

        ensure_initialized!(state)
        @test is_initialized(state)

        @test get_bond_dimension(state) == 1
    end

    @testset "ensure_initialized! auto-initializes" begin
        state = CAMPSState(4)

        @test get_bond_dimension(state) == 1

        @test is_initialized(state)
    end
end
