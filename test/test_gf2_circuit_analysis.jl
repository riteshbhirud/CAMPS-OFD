using Test
using CAMPS
using QuantumClifford

@testset "GF(2) Circuit Analysis" begin

    @testset "Phase 1 Tuple Format - H then T" begin
        n = 4
        gates = [(:H, [1]), (:H, [2]), (:H, [3]), (:H, [4]),
                 (:T, [1]), (:T, [2]), (:T, [3]), (:T, [4])]
        t_pos = [5, 6, 7, 8]

        result = compute_gf2_for_mixed_circuit(gates, t_pos, n)
        @test result.n_t_gates == 4
        @test result.gf2_rank == 4
        @test result.nullity == 0
        @test result.predicted_chi == 1
        @test result.n_disentanglable == 4
        @test result.n_not_disentanglable == 0
        @test result.rank_to_t_ratio == 1.0
    end

    @testset "Phase 2 Gate Objects - H then T" begin
        n = 4
        gates = Gate[HGate(1), HGate(2), HGate(3), HGate(4),
                     TGate(1), TGate(2), TGate(3), TGate(4)]
        t_pos = [5, 6, 7, 8]

        result = compute_gf2_for_mixed_circuit(gates, t_pos, n)
        @test result.n_t_gates == 4
        @test result.gf2_rank == 4
        @test result.nullity == 0
        @test result.predicted_chi == 1
    end

    @testset "Pure Z Paulis - T without H" begin
        n = 3
        gates = [(:T, [1]), (:T, [2]), (:T, [3])]
        t_pos = [1, 2, 3]

        result = compute_gf2_for_mixed_circuit(gates, t_pos, n)
        @test result.n_t_gates == 3
        @test result.gf2_rank == 0
        @test result.nullity == 3
        @test result.predicted_chi == 8
        @test result.xbit_density == 0.0
    end

    @testset "Mixed - Some H, Some Not" begin
        n = 3
        gates = [(:H, [1]), (:T, [1]), (:T, [2])]
        t_pos = [2, 3]

        result = compute_gf2_for_mixed_circuit(gates, t_pos, n)
        @test result.n_t_gates == 2
        @test result.gf2_rank == 1
        @test result.nullity == 1
        @test result.predicted_chi == 2
    end

    @testset "Empty Circuit" begin
        result = compute_gf2_for_mixed_circuit(Any[], Int[], 4)
        @test result.n_t_gates == 0
        @test result.gf2_rank == 0
        @test result.nullity == 0
        @test result.predicted_chi == 1
        @test isnan(result.rank_to_t_ratio)
    end

    @testset "Clifford-Only Circuit (No T-gates)" begin
        n = 3
        gates = Gate[HGate(1), CNOTGate(1, 2), HGate(2)]
        t_pos = Int[]

        result = compute_gf2_for_mixed_circuit(gates, t_pos, n)
        @test result.n_t_gates == 0
        @test result.gf2_rank == 0
        @test result.predicted_chi == 1
    end

    @testset "Consistency with predict_bond_dimension_for_circuit" begin
        n = 5
        gates = Gate[HGate(1), HGate(2), HGate(3), CNOTGate(1, 2),
                     TGate(1), CNOTGate(2, 3), TGate(2), TGate(3)]
        t_pos = [5, 7, 8]

        new_result = compute_gf2_for_mixed_circuit(gates, t_pos, n)
        old_result = predict_bond_dimension_for_circuit(gates, n)

        @test new_result.gf2_rank == old_result.gf2_rank
        @test new_result.nullity == old_result.nullity
        @test new_result.n_t_gates == old_result.n_t_gates
        @test new_result.predicted_chi == old_result.predicted_chi
    end

    @testset "Consistency - Larger Circuit" begin
        n = 6
        gates = Gate[
            HGate(1), HGate(2), HGate(3), HGate(4), HGate(5), HGate(6),
            CNOTGate(1, 2), CNOTGate(3, 4), CNOTGate(5, 6),
            TGate(1), TGate(2), TGate(3), TGate(4),
            CNOTGate(2, 3), CNOTGate(4, 5),
            TGate(5), TGate(6)
        ]
        t_pos = [10, 11, 12, 13, 16, 17]

        new_result = compute_gf2_for_mixed_circuit(gates, t_pos, n)
        old_result = predict_bond_dimension_for_circuit(gates, n)

        @test new_result.gf2_rank == old_result.gf2_rank
        @test new_result.nullity == old_result.nullity
        @test new_result.predicted_chi == old_result.predicted_chi
    end

    @testset "simulate_ofd=false mode" begin
        n = 4
        gates = Gate[HGate(1), HGate(2), HGate(3), HGate(4),
                     TGate(1), TGate(2), TGate(3), TGate(4)]
        t_pos = [5, 6, 7, 8]

        result_ofd = compute_gf2_for_mixed_circuit(gates, t_pos, n; simulate_ofd=true)
        result_noofd = compute_gf2_for_mixed_circuit(gates, t_pos, n; simulate_ofd=false)

        @test result_ofd.n_t_gates == result_noofd.n_t_gates

        @test result_ofd.gf2_rank >= 0
        @test result_noofd.gf2_rank >= 0
        @test result_ofd.nullity >= 0
        @test result_noofd.nullity >= 0
    end

    @testset "CNOT Entangling Effect" begin
        n = 3
        gates = Gate[HGate(1), HGate(2), CNOTGate(1, 2), TGate(1)]
        t_pos = [4]

        result = compute_gf2_for_mixed_circuit(gates, t_pos, n)
        @test result.n_t_gates == 1
        @test result.gf2_rank == 1
        @test result.nullity == 0
        @test result.predicted_chi == 1
        @test length(result.twisted_paulis) == 1
    end

    @testset "Twisted Paulis are Collected" begin
        n = 3
        gates = Gate[HGate(1), TGate(1), HGate(2), TGate(2)]
        t_pos = [2, 4]

        result = compute_gf2_for_mixed_circuit(gates, t_pos, n)
        @test length(result.twisted_paulis) == 2
        @test all(P -> nqubits(P) == n, result.twisted_paulis)
    end

    @testset "Sdag Gate Handling" begin
        n = 2
        gates = [(:H, [1]), (:Sdag, [1]), (:T, [1])]
        t_pos = [3]

        result = compute_gf2_for_mixed_circuit(gates, t_pos, n)
        @test result.n_t_gates == 1
        @test result.gf2_rank >= 0
    end
end
