using Test
using CAMPS

@testset "DisentanglingStrategy Types" begin
    @testset "OFDStrategy" begin
        s = OFDStrategy()
        @test s isa DisentanglingStrategy
        @test s isa OFDStrategy

        @test contains(repr(s), "OFDStrategy")
    end

    @testset "OBDStrategy" begin
        s1 = OBDStrategy()
        @test s1.max_sweeps == 10
        @test s1.improvement_threshold == 1e-10

        s2 = OBDStrategy(max_sweeps=5, improvement_threshold=1e-8)
        @test s2.max_sweeps == 5
        @test s2.improvement_threshold == 1e-8

        @test_throws ArgumentError OBDStrategy(max_sweeps=0)
        @test_throws ArgumentError OBDStrategy(max_sweeps=-1)
        @test_throws ArgumentError OBDStrategy(improvement_threshold=-0.1)

        @test contains(repr(s1), "OBDStrategy")
        @test contains(repr(s1), "max_sweeps=10")
    end

    @testset "HybridStrategy" begin
        s1 = HybridStrategy()
        @test s1.obd_sweeps_on_failure == 2
        @test s1.obd_improvement_threshold == 1e-10

        s2 = HybridStrategy(obd_sweeps_on_failure=5, obd_improvement_threshold=1e-12)
        @test s2.obd_sweeps_on_failure == 5
        @test s2.obd_improvement_threshold == 1e-12

        @test_throws ArgumentError HybridStrategy(obd_sweeps_on_failure=0)
        @test_throws ArgumentError HybridStrategy(obd_improvement_threshold=-1.0)

        @test contains(repr(s1), "HybridStrategy")
    end

    @testset "NoDisentangling" begin
        s = NoDisentangling()
        @test s isa DisentanglingStrategy
        @test s isa NoDisentangling
        @test contains(repr(s), "NoDisentangling")
    end
end

@testset "Gate Types" begin
    @testset "RotationGate" begin
        t = TGate(1)
        @test t isa Gate
        @test t isa RotationGate
        @test t.qubit == 1
        @test t.axis == :Z
        @test t.angle ≈ π/4

        tdag = TdagGate(2)
        @test tdag.qubit == 2
        @test tdag.axis == :Z
        @test tdag.angle ≈ -π/4

        rz = RzGate(3, π/3)
        @test rz.qubit == 3
        @test rz.axis == :Z
        @test rz.angle ≈ π/3

        rx = RxGate(1, π/2)
        @test rx.axis == :X
        @test rx.angle ≈ π/2

        ry = RyGate(2, 0.5)
        @test ry.axis == :Y
        @test ry.angle ≈ 0.5

        @test_throws ArgumentError RotationGate(0, :Z, π/4)
        @test_throws ArgumentError RotationGate(-1, :Z, π/4)

        @test_throws ArgumentError RotationGate(1, :W, π/4)

        @test contains(repr(TGate(1)), "π/4")
        @test contains(repr(TdagGate(1)), "-π/4")
    end

    @testset "CliffordGate" begin
        h = HGate(1)
        @test h isa Gate
        @test h isa CliffordGate
        @test h.qubits == [1]
        @test length(h.gates) == 1

        s = SGate(2)
        @test s.qubits == [2]

        sdag = SdagGate(3)
        @test sdag.qubits == [3]

        x = XGate(1)
        y = YGate(2)
        z = ZGate(3)
        @test x.qubits == [1]
        @test y.qubits == [2]
        @test z.qubits == [3]

        cnot = CNOTGate(1, 2)
        @test cnot.qubits == [1, 2]

        cz = CZGate(2, 3)
        @test cz.qubits == [2, 3]

        swap = SWAPGate(1, 3)
        @test swap.qubits == [1, 3]

        @test_throws ArgumentError CNOTGate(1, 1)
        @test_throws ArgumentError CZGate(2, 2)
        @test_throws ArgumentError SWAPGate(3, 3)

        @test_throws ArgumentError HGate(0)
        @test_throws ArgumentError CNOTGate(0, 1)
        @test_throws ArgumentError CNOTGate(1, -1)

        @test contains(repr(h), "CliffordGate")
        @test contains(repr(h), "1")
    end
end

@testset "CAMPSState" begin
    @testset "Construction" begin
        state = CAMPSState(5)
        @test n_qubits(state) == 5
        @test state.max_bond == 1024
        @test state.cutoff == 1e-15

        state2 = CAMPSState(10; max_bond=512, cutoff=1e-12)
        @test n_qubits(state2) == 10
        @test state2.max_bond == 512
        @test state2.cutoff == 1e-12

        @test_throws ArgumentError CAMPSState(0)
        @test_throws ArgumentError CAMPSState(-1)
        @test_throws ArgumentError CAMPSState(5; max_bond=0)
        @test_throws ArgumentError CAMPSState(5; max_bond=-1)
        @test_throws ArgumentError CAMPSState(5; cutoff=0.0)
        @test_throws ArgumentError CAMPSState(5; cutoff=-1e-10)
    end

    @testset "Qubit Tracking" begin
        state = CAMPSState(5)

        @test num_free_qubits(state) == 5
        @test num_magic_qubits(state) == 0
        @test get_free_qubit_indices(state) == [1, 2, 3, 4, 5]
        @test get_magic_qubit_indices(state) == Int[]

        for q in 1:5
            @test is_free(state, q) == true
            @test is_magic(state, q) == false
        end

        mark_as_magic!(state, 2)
        @test is_free(state, 2) == false
        @test is_magic(state, 2) == true
        @test num_free_qubits(state) == 4
        @test num_magic_qubits(state) == 1
        @test get_free_qubit_indices(state) == [1, 3, 4, 5]
        @test get_magic_qubit_indices(state) == [2]

        mark_as_magic!(state, 4)
        @test num_free_qubits(state) == 3
        @test num_magic_qubits(state) == 2
        @test get_free_qubit_indices(state) == [1, 3, 5]
        @test get_magic_qubit_indices(state) == [2, 4]

        @test_throws BoundsError is_free(state, 0)
        @test_throws BoundsError is_free(state, 6)
        @test_throws BoundsError is_magic(state, 0)
        @test_throws BoundsError is_magic(state, 6)
        @test_throws BoundsError mark_as_magic!(state, 0)
        @test_throws BoundsError mark_as_magic!(state, 6)
    end

    @testset "Twisted Pauli Tracking" begin
        state = CAMPSState(3)

        @test num_twisted_paulis(state) == 0
        @test isempty(state.twisted_paulis)

        push!(state.twisted_paulis, :placeholder1)
        @test num_twisted_paulis(state) == 1

        push!(state.twisted_paulis, :placeholder2)
        push!(state.twisted_paulis, :placeholder3)
        @test num_twisted_paulis(state) == 3
    end

    @testset "Display" begin
        state = CAMPSState(5)

        short_repr = repr(state)
        @test contains(short_repr, "CAMPSState")
        @test contains(short_repr, "n=5")
        @test contains(short_repr, "free=5")
        @test contains(short_repr, "magic=0")

        io = IOBuffer()
        show(io, MIME("text/plain"), state)
        long_repr = String(take!(io))
        @test contains(long_repr, "Qubits: 5")
        @test contains(long_repr, "Free qubits: 5")
        @test contains(long_repr, "Magic qubits: 0")
        @test contains(long_repr, "Max bond dimension: 1024")
    end
end

@testset "Type Hierarchy" begin
    @test OFDStrategy() isa DisentanglingStrategy
    @test OBDStrategy() isa DisentanglingStrategy
    @test HybridStrategy() isa DisentanglingStrategy
    @test NoDisentangling() isa DisentanglingStrategy

    @test TGate(1) isa Gate
    @test HGate(1) isa Gate
    @test CNOTGate(1, 2) isa Gate

    @test TGate(1) isa RotationGate
    @test HGate(1) isa CliffordGate
end
