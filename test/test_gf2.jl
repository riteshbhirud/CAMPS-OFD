using Test
using CAMPS
using QuantumClifford

@testset "GF(2) Matrix Construction" begin
    @testset "build_gf2_matrix basic" begin
        M = build_gf2_matrix([P"X"])
        @test size(M) == (1, 1)
        @test M[1, 1] == true

        M = build_gf2_matrix([P"Z"])
        @test M[1, 1] == false
    end

    @testset "build_gf2_matrix multi-qubit" begin
        P = P"IXYZ"
        M = build_gf2_matrix([P])
        @test size(M) == (1, 4)
        @test M[1, :] == [false, true, true, false]
    end

    @testset "build_gf2_matrix multiple paulis" begin
        paulis = [P"XZI", P"ZXI", P"YYI"]
        M = build_gf2_matrix(paulis)

        @test size(M) == (3, 3)

        @test M[1, :] == [true, false, false]

        @test M[2, :] == [false, true, false]

        @test M[3, :] == [true, true, false]
    end

    @testset "build_gf2_matrix empty" begin
        M = build_gf2_matrix(PauliOperator[])
        @test size(M) == (0, 0)
    end

    @testset "build_gf2_matrix_from_xbits" begin
        xbits = [
            BitVector([true, false, false]),
            BitVector([false, true, false]),
            BitVector([true, true, false])
        ]
        M = build_gf2_matrix_from_xbits(xbits)

        @test size(M) == (3, 3)
        @test M[1, :] == [true, false, false]
        @test M[2, :] == [false, true, false]
        @test M[3, :] == [true, true, false]
    end
end

@testset "GF(2) Rank" begin
    @testset "gf2_rank identity" begin
        M = Bool[true false false; false true false; false false true]
        @test gf2_rank(M) == 3
    end

    @testset "gf2_rank zero matrix" begin
        M = Bool[false false; false false; false false]
        @test gf2_rank(M) == 0
    end

    @testset "gf2_rank linearly dependent" begin
        M = Bool[true true false;
                 false true true;
                 true false true]
        @test gf2_rank(M) == 2
    end

    @testset "gf2_rank single row" begin
        M = Bool[true true false]
        @test gf2_rank(reshape(M, 1, 3)) == 1

        M_zero = Bool[false false false]
        @test gf2_rank(reshape(M_zero, 1, 3)) == 0
    end

    @testset "gf2_rank empty" begin
        @test gf2_rank(Matrix{Bool}(undef, 0, 0)) == 0
    end

    @testset "gf2_rank! modifies in place" begin
        M = Bool[true true false;
                 false true true;
                 true false true]
        M_copy = copy(M)

        r = gf2_rank!(M_copy)
        @test r == 2
        @test M_copy != M
    end
end

@testset "Bond Dimension Prediction" begin
    @testset "predict_bond_dimension empty" begin
        @test predict_bond_dimension(PauliOperator[]) == 1
    end

    @testset "predict_bond_dimension all Z" begin
        paulis = [P"Z", P"Z", P"Z"]
        @test predict_bond_dimension(paulis) == 8
    end

    @testset "predict_bond_dimension independent X" begin
        paulis = [P"X__", P"_X_", P"__X"]
        M = build_gf2_matrix(paulis)
        @test predict_bond_dimension(paulis) == 1
    end

    @testset "predict_bond_dimension dependent" begin
        paulis = [P"XX_", P"_XX", P"X_X"]

        M = build_gf2_matrix(paulis)
        @test size(M) == (3, 3)

        @test M[1, :] == [true, true, false]
        @test M[2, :] == [false, true, true]
        @test M[3, :] == [true, false, true]

        @test predict_bond_dimension(paulis) == 2
    end

    @testset "predict_bond_dimension from matrix" begin
        M = Bool[true false; true false]
        @test predict_bond_dimension(M) == 2
    end
end

@testset "Disentanglability Analysis" begin
    @testset "can_disentangle" begin
        P = P"XZI"

        @test can_disentangle(P, BitVector([true, false, false])) == true

        @test can_disentangle(P, BitVector([false, true, true])) == false

        @test can_disentangle(P, BitVector([false, false, true])) == false
    end

    @testset "find_disentangling_qubit" begin
        P = P"XYI"

        @test find_disentangling_qubit(P, BitVector([true, false, false])) == 1

        @test find_disentangling_qubit(P, BitVector([false, true, false])) == 2

        @test find_disentangling_qubit(P, BitVector([true, true, false])) == 1

        @test find_disentangling_qubit(P, BitVector([false, false, true])) === nothing

        P_z = P"ZZZ"
        @test find_disentangling_qubit(P_z, BitVector([true, true, true])) === nothing
    end

    @testset "count_disentanglable" begin
        paulis = [P"X__", P"_X_", P"ZZZ"]
        free = BitVector([true, true, true])

        @test count_disentanglable(paulis, free) == 2

        free2 = BitVector([false, false, true])
        @test count_disentanglable(paulis, free2) == 0
    end
end

@testset "GF(2) Structure Analysis" begin
    @testset "analyze_gf2_structure basic" begin
        paulis = [P"X_", P"_X"]

        result = analyze_gf2_structure(paulis)

        @test result.t == 2
        @test result.n == 2
        @test result.rank == 2
        @test result.nullity == 0
        @test result.predicted_chi == 1
    end

    @testset "analyze_gf2_structure with dependencies" begin
        paulis = [P"XX_", P"_XX", P"X_X"]

        result = analyze_gf2_structure(paulis)

        @test result.t == 3
        @test result.n == 3
        @test result.rank == 2
        @test result.nullity == 1
        @test result.predicted_chi == 2
    end

    @testset "analyze_gf2_structure empty" begin
        result = analyze_gf2_structure(PauliOperator[])

        @test result.t == 0
        @test result.n == 0
        @test result.rank == 0
        @test result.nullity == 0
        @test result.predicted_chi == 1
    end
end

@testset "GF(2) Null Space" begin
    @testset "gf2_null_space full rank" begin
        M = Bool[true false; false true]
        ns = gf2_null_space(M)
        @test isempty(ns)
    end

    @testset "gf2_null_space rank deficient" begin
        M = Bool[true true; true true]
        ns = gf2_null_space(M)

        @test length(ns) == 1
        @test ns[1] == BitVector([true, true])
    end

    @testset "gf2_null_space 3 rows" begin
        M = Bool[true true false;
                 false true true;
                 true false true]
        ns = gf2_null_space(M)

        @test length(ns) == 1
        @test ns[1] == BitVector([true, true, true])
    end
end

@testset "Incremental Rank Update" begin
    @testset "incremental_rank_update new independent row" begin
        M = Bool[true false; false true]
        new_row = BitVector([true, true])

        M_echelon = copy(M)
        gf2_gausselim!(M_echelon)

        new_rank, is_indep = incremental_rank_update(M_echelon, new_row)

        @test new_rank == 2
        @test is_indep == false
    end

    @testset "incremental_rank_update new dependent row" begin
        M = Bool[true false; false true]
        new_row = BitVector([true, false])

        M_echelon = copy(M)
        gf2_gausselim!(M_echelon)

        new_rank, is_indep = incremental_rank_update(M_echelon, new_row)

        @test is_indep == false
        @test new_rank == 2
    end

    @testset "incremental_rank_update empty matrix" begin
        M = Matrix{Bool}(undef, 0, 3)
        new_row = BitVector([true, false, true])

        new_rank, is_indep = incremental_rank_update(M, new_row)

        @test new_rank == 1
        @test is_indep == true
    end

    @testset "incremental_rank_update zero row" begin
        M = Bool[true false]
        new_row = BitVector([false, false])

        M_echelon = copy(M)
        gf2_gausselim!(M_echelon)

        new_rank, is_indep = incremental_rank_update(reshape(M_echelon, 1, 2), new_row)

        @test is_indep == false
        @test new_rank == 1
    end
end

@testset "find_independent_rows" begin
    @testset "Identity matrix" begin
        M = Bool[true false false; false true false; false false true]
        rows = find_independent_rows(M)
        @test sort(rows) == [1, 2, 3]
    end

    @testset "All dependent rows" begin
        M = Bool[true true; true true; true true]
        rows = find_independent_rows(M)
        @test length(rows) == 1
        @test rows[1] ∈ [1, 2, 3]
    end

    @testset "Mixed independence with swaps" begin
        M = Bool[false true false;
                 true false false;
                 true true false]
        rows = find_independent_rows(M)
        @test length(rows) == 2
        @test sort(rows) == [1, 2]
    end

    @testset "Specific case for permutation tracking" begin
        M = Bool[false false true;
                 false true false;
                 true false false]
        rows = find_independent_rows(M)
        @test length(rows) == 3
        @test sort(rows) == [1, 2, 3]
    end

    @testset "Empty matrix" begin
        M = Matrix{Bool}(undef, 0, 0)
        @test find_independent_rows(M) == Int[]
    end

    @testset "find_independent_rows_with_basis" begin
        M = Bool[true true false;
                 false true true;
                 true false true]

        rows, echelon = find_independent_rows_with_basis(M)

        @test length(rows) == 2
        @test size(echelon) == size(M)
        nonzero_rows = sum(any(echelon[r, :]) for r in 1:3)
        @test nonzero_rows == 2
    end
end

@testset "Integration: GF(2) with Twisted Paulis" begin
    @testset "After Hadamard layer" begin
        n = 5
        paulis = [single_x(n, k) for k in 1:n]

        @test predict_bond_dimension(paulis) == 1
    end

    @testset "All Z Paulis" begin
        n = 4
        t = 3
        paulis = [single_z(n, k) for k in 1:t]

        @test predict_bond_dimension(paulis) == 2^t
    end

    @testset "Mixed scenario" begin
        paulis = [P"XY_", P"YX_", P"XX_"]

        @test predict_bond_dimension(paulis) == 4
    end
end
