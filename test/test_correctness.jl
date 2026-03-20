
using Test
using CAMPS
using QuantumClifford
import LinearAlgebra

"""Build the 2^n × 2^n matrix for a single-qubit gate on qubit `q` in an `n`-qubit system."""
function exact_single_qubit_gate(U::Matrix{ComplexF64}, q::Int, n::Int)
    result = Matrix{ComplexF64}(LinearAlgebra.I, 1, 1)
    for i in n:-1:1
        if i == q
            result = kron(result, U)
        else
            result = kron(result, Matrix{ComplexF64}(LinearAlgebra.I, 2, 2))
        end
    end
    return result
end

"""Build the 2^n × 2^n matrix for a CNOT gate (control c, target t) in an `n`-qubit system."""
function exact_cnot_gate(c::Int, t::Int, n::Int)
    dim = 2^n
    U = zeros(ComplexF64, dim, dim)
    for j in 0:(dim-1)
        bits = digits(j, base=2, pad=n)
        c_bit = bits[c]
        new_bits = copy(bits)
        if c_bit == 1
            new_bits[t] = 1 - new_bits[t]
        end
        i = sum(new_bits[k] * 2^(k-1) for k in 1:n)
        U[i+1, j+1] = 1.0
    end
    return U
end

"""Build the 2^n × 2^n matrix for a CZ gate in an `n`-qubit system."""
function exact_cz_gate(q1::Int, q2::Int, n::Int)
    dim = 2^n
    U = Matrix{ComplexF64}(LinearAlgebra.I, dim, dim)
    for j in 0:(dim-1)
        bits = digits(j, base=2, pad=n)
        if bits[q1] == 1 && bits[q2] == 1
            U[j+1, j+1] = -1.0
        end
    end
    return U
end

"""Build the 2^n × 2^n matrix for a SWAP gate."""
function exact_swap_gate(q1::Int, q2::Int, n::Int)
    dim = 2^n
    U = zeros(ComplexF64, dim, dim)
    for j in 0:(dim-1)
        bits = digits(j, base=2, pad=n)
        new_bits = copy(bits)
        new_bits[q1], new_bits[q2] = bits[q2], bits[q1]
        i = sum(new_bits[k] * 2^(k-1) for k in 1:n)
        U[i+1, j+1] = 1.0
    end
    return U
end

const GATE_H = ComplexF64[1 1; 1 -1] / sqrt(2)
const GATE_S = ComplexF64[1 0; 0 im]
const GATE_Sdag = ComplexF64[1 0; 0 -im]
const GATE_T = ComplexF64[1 0; 0 exp(im*π/4)]
const GATE_Tdag = ComplexF64[1 0; 0 exp(-im*π/4)]
const GATE_X = ComplexF64[0 1; 1 0]
const GATE_Y = ComplexF64[0 -im; im 0]
const GATE_Z = ComplexF64[1 0; 0 -1]
const GATE_I2 = ComplexF64[1 0; 0 1]

"""Rz(θ) = exp(-iθZ/2) = diag(e^{-iθ/2}, e^{iθ/2})"""
function gate_rz(θ::Real)
    return ComplexF64[exp(-im*θ/2) 0; 0 exp(im*θ/2)]
end

"""Rx(θ) = exp(-iθX/2)"""
function gate_rx(θ::Real)
    c = cos(θ/2)
    s = sin(θ/2)
    return ComplexF64[c -im*s; -im*s c]
end

"""Ry(θ) = exp(-iθY/2)"""
function gate_ry(θ::Real)
    c = cos(θ/2)
    s = sin(θ/2)
    return ComplexF64[c -s; s c]
end

"""
    exact_simulate(circuit::Vector{Gate}, n::Int) -> Vector{ComplexF64}

Simulate a circuit by direct matrix multiplication, returning the final statevector.
Completely independent of the CAMPS pipeline.
"""
function exact_simulate(circuit::Vector{Gate}, n::Int)
    dim = 2^n
    ψ = zeros(ComplexF64, dim)
    ψ[1] = 1.0

    for gate in circuit
        if gate isa CliffordGate
            U = exact_clifford_gate_matrix(gate, n)
        elseif gate isa RotationGate
            U = exact_rotation_gate_matrix(gate, n)
        else
            error("Unknown gate type: $(typeof(gate))")
        end
        ψ = U * ψ
    end
    return ψ
end

"""Convert a CliffordGate to its exact 2^n × 2^n matrix."""
function exact_clifford_gate_matrix(gate::CliffordGate, n::Int)
    U = Matrix{ComplexF64}(LinearAlgebra.I, 2^n, 2^n)
    for spec in gate.gates
        if spec isa Tuple
            name = spec[1]
            if name == :H
                U = exact_single_qubit_gate(GATE_H, spec[2], n) * U
            elseif name == :S
                U = exact_single_qubit_gate(GATE_S, spec[2], n) * U
            elseif name == :Sdag
                U = exact_single_qubit_gate(GATE_Sdag, spec[2], n) * U
            elseif name == :X
                U = exact_single_qubit_gate(GATE_X, spec[2], n) * U
            elseif name == :Y
                U = exact_single_qubit_gate(GATE_Y, spec[2], n) * U
            elseif name == :Z
                U = exact_single_qubit_gate(GATE_Z, spec[2], n) * U
            elseif name == :CNOT
                U = exact_cnot_gate(spec[2], spec[3], n) * U
            elseif name == :CZ
                U = exact_cz_gate(spec[2], spec[3], n) * U
            elseif name == :SWAP
                U = exact_swap_gate(spec[2], spec[3], n) * U
            else
                error("Unknown Clifford gate: $name")
            end
        else
            error("Non-tuple gate spec in CliffordGate: $spec")
        end
    end
    return U
end

"""Convert a RotationGate to its exact 2^n × 2^n matrix."""
function exact_rotation_gate_matrix(gate::RotationGate, n::Int)
    if gate.axis == :Z
        U2 = gate_rz(gate.angle)
    elseif gate.axis == :X
        U2 = gate_rx(gate.angle)
    elseif gate.axis == :Y
        U2 = gate_ry(gate.angle)
    else
        error("Unknown rotation axis: $(gate.axis)")
    end
    return exact_single_qubit_gate(U2, gate.qubit, n)
end

"""Compare two statevectors up to global phase."""
function states_equal_up_to_phase(ψ1::Vector{ComplexF64}, ψ2::Vector{ComplexF64}; atol=1e-10)
    @assert length(ψ1) == length(ψ2)
    idx = findfirst(i -> abs(ψ1[i]) > atol, 1:length(ψ1))
    if idx === nothing
        return LinearAlgebra.norm(ψ2) < atol
    end
    if abs(ψ2[idx]) < atol
        return false
    end
    phase = ψ2[idx] / ψ1[idx]
    if abs(abs(phase) - 1.0) > atol
        return false
    end
    return LinearAlgebra.norm(ψ1 * phase - ψ2) < atol * sqrt(length(ψ1))
end

@testset "End-to-End Correctness" begin

    @testset "CAMPS vs exact statevector — Clifford-only circuits" begin

        @testset "Bell state: H-CNOT on 2 qubits" begin
            circuit = Gate[HGate(1), CNOTGate(1, 2)]
            n = 2
            result = simulate_circuit(circuit, n; strategy=HybridStrategy())
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
            @test abs(ψ_exact[1]) ≈ 1/sqrt(2) atol=1e-10
            @test abs(ψ_exact[4]) ≈ 1/sqrt(2) atol=1e-10
        end

        @testset "GHZ state on 3 qubits" begin
            circuit = Gate[HGate(1), CNOTGate(1, 2), CNOTGate(2, 3)]
            n = 3
            result = simulate_circuit(circuit, n)
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
        end

        @testset "All single-qubit Cliffords on 3 qubits" begin
            circuit = Gate[
                HGate(1), SGate(2), XGate(3),
                SGate(1), HGate(3),
                YGate(2), ZGate(1),
                SdagGate(3)
            ]
            n = 3
            result = simulate_circuit(circuit, n)
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
        end

        @testset "Entangling Clifford circuit on 4 qubits" begin
            circuit = Gate[
                HGate(1), HGate(2), HGate(3), HGate(4),
                CNOTGate(1, 2), CNOTGate(3, 4),
                CZGate(2, 3),
                SGate(1), SdagGate(4),
                CNOTGate(1, 3), CNOTGate(2, 4),
                HGate(1), HGate(4)
            ]
            n = 4
            result = simulate_circuit(circuit, n)
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
        end
    end

    @testset "CAMPS vs exact — Clifford+T circuits with OFD" begin

        @testset "Single T gate: H-T on 2 qubits" begin
            circuit = Gate[HGate(1), TGate(1)]
            n = 2
            result = simulate_circuit(circuit, n; strategy=OFDStrategy())
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
            @test result.final_bond_dim == 1
        end

        @testset "T† gate: H-T† on 2 qubits" begin
            circuit = Gate[HGate(1), TdagGate(1)]
            n = 2
            result = simulate_circuit(circuit, n; strategy=OFDStrategy())
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
        end

        @testset "Multiple T gates with OFD: H^⊗3 then T^⊗3 on 4 qubits" begin
            circuit = Gate[
                HGate(1), HGate(2), HGate(3),
                TGate(1), TGate(2), TGate(3)
            ]
            n = 4
            result = simulate_circuit(circuit, n; strategy=OFDStrategy())
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
            @test result.final_bond_dim == 1
        end

        @testset "T after entangling Cliffords on 3 qubits" begin
            circuit = Gate[
                HGate(1), HGate(2), HGate(3),
                CNOTGate(1, 2), CNOTGate(2, 3),
                TGate(1), TGate(2)
            ]
            n = 3
            result = simulate_circuit(circuit, n; strategy=OFDStrategy())
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
        end

        @testset "Interleaved Clifford and T on 3 qubits" begin
            circuit = Gate[
                HGate(1), HGate(2), HGate(3),
                TGate(1),
                CNOTGate(1, 2),
                TGate(2),
                CNOTGate(2, 3),
                TGate(3),
                HGate(1),
                TGate(1)
            ]
            n = 3
            result = simulate_circuit(circuit, n; strategy=HybridStrategy())
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
        end
    end

    @testset "CAMPS vs exact — Non-T rotations (Rz arbitrary angles)" begin

        @testset "Rz(π/3) — non-Clifford angle" begin
            circuit = Gate[HGate(1), RzGate(1, π/3)]
            n = 2
            result = simulate_circuit(circuit, n; strategy=HybridStrategy())
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
        end

        @testset "Rz(π/8) — eighth-turn" begin
            circuit = Gate[HGate(1), RzGate(1, π/8)]
            n = 2
            result = simulate_circuit(circuit, n; strategy=HybridStrategy())
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
        end

        @testset "Rx and Ry rotations" begin
            for (axis, gate_fn) in [(:X, RxGate), (:Y, RyGate)]
                circuit = Gate[HGate(1), gate_fn(1, π/5)]
                n = 2
                result = simulate_circuit(circuit, n; strategy=HybridStrategy())
                ψ_camps = state_vector(result.state)
                ψ_exact = exact_simulate(circuit, n)
                @test states_equal_up_to_phase(ψ_exact, ψ_camps)
            end
        end
    end

    @testset "CAMPS vs exact — OBD strategy" begin

        @testset "OBD does not corrupt the quantum state" begin
            circuit = Gate[
                HGate(1), HGate(2),
                CNOTGate(1, 2),
                TGate(1), TGate(2)
            ]
            n = 2
            result = simulate_circuit(circuit, n; strategy=OBDStrategy(max_sweeps=2))
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
        end

        @testset "HybridStrategy matches exact for 3-qubit circuit" begin
            circuit = Gate[
                HGate(1), HGate(2), HGate(3),
                CNOTGate(1, 2),
                TGate(1),
                CNOTGate(2, 3),
                TGate(2), TGate(3)
            ]
            n = 3
            result = simulate_circuit(circuit, n; strategy=HybridStrategy())
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
        end
    end

    @testset "CAMPS vs exact — NoDisentangling baseline" begin

        @testset "NoDisentangling produces correct state" begin
            circuit = Gate[
                HGate(1), HGate(2),
                CNOTGate(1, 2),
                TGate(1), TGate(2)
            ]
            n = 2
            result = simulate_circuit(circuit, n; strategy=NoDisentangling())
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
        end
    end

    @testset "CAMPS vs exact — Circuit generators" begin

        @testset "GHZ circuit correctness" begin
            for n in 2:4
                circuit = ghz_circuit(n)
                result = simulate_circuit(circuit, n)
                ψ_camps = state_vector(result.state)
                ψ_exact = exact_simulate(circuit, n)
                @test states_equal_up_to_phase(ψ_exact, ψ_camps)
                @test abs(ψ_exact[1]) ≈ 1/sqrt(2) atol=1e-10
                @test abs(ψ_exact[end]) ≈ 1/sqrt(2) atol=1e-10
            end
        end

        @testset "QFT circuit correctness — DFT on |0⟩" begin
            for n in 2:4
                circuit = qft_circuit(n)
                result = simulate_circuit(circuit, n; strategy=HybridStrategy())
                ψ_camps = state_vector(result.state)
                ψ_exact = exact_simulate(circuit, n)
                @test states_equal_up_to_phase(ψ_exact, ψ_camps)
                expected_amp = 1.0 / sqrt(2^n)
                for amp in ψ_exact
                    @test abs(amp) ≈ expected_amp atol=1e-10
                end
            end
        end

        @testset "QFT circuit — DFT on |1⟩ (roots of unity)" begin
            for n in [2, 3]
                N = 2^n
                circuit_prep = Gate[XGate(1)]
                circuit_qft = qft_circuit(n)
                full_circuit = vcat(circuit_prep, circuit_qft)
                ψ_exact = exact_simulate(full_circuit, n)
                for amp in ψ_exact
                    @test abs(amp) ≈ 1/sqrt(N) atol=1e-10
                end
            end
        end

        @testset "Inverse QFT undoes QFT" begin
            for n in 2:4
                circuit = vcat(qft_circuit(n), inverse_qft_circuit(n))
                result = simulate_circuit(circuit, n; strategy=HybridStrategy())
                ψ_camps = state_vector(result.state)
                @test abs(ψ_camps[1]) ≈ 1.0 atol=1e-8
                total_prob = sum(abs2, ψ_camps)
                @test total_prob ≈ 1.0 atol=1e-8
                for i in 2:2^n
                    @test abs(ψ_camps[i]) < 1e-8
                end
            end
        end

        @testset "W state circuit — CAMPS matches exact" begin
            for n in 3:4
                circuit = w_state_circuit(n)
                result = simulate_circuit(circuit, n; strategy=NoDisentangling(), max_bond=1024)
                ψ_camps = state_vector(result.state)
                ψ_exact = exact_simulate(circuit, n)
                @test states_equal_up_to_phase(ψ_exact, ψ_camps)
                @test sum(abs2, ψ_exact) ≈ 1.0 atol=1e-10
            end
        end
    end

    @testset "OFD state preservation — OFD vs no-OFD produce same state" begin

        @testset "Single OFD: same state with and without" begin
            circuit = Gate[HGate(1), HGate(2), TGate(1)]
            n = 3
            result_ofd = simulate_circuit(circuit, n; strategy=OFDStrategy())
            result_none = simulate_circuit(circuit, n; strategy=NoDisentangling())
            ψ_ofd = state_vector(result_ofd.state)
            ψ_none = state_vector(result_none.state)
            @test states_equal_up_to_phase(ψ_ofd, ψ_none)
            @test result_ofd.final_bond_dim <= result_none.final_bond_dim
        end

        @testset "Multiple OFDs: same state with and without" begin
            circuit = Gate[
                HGate(1), HGate(2), HGate(3), HGate(4),
                CNOTGate(1, 2), CNOTGate(3, 4),
                TGate(1), TGate(2), TGate(3)
            ]
            n = 4
            result_ofd = simulate_circuit(circuit, n; strategy=OFDStrategy())
            result_none = simulate_circuit(circuit, n; strategy=NoDisentangling())
            ψ_ofd = state_vector(result_ofd.state)
            ψ_none = state_vector(result_none.state)
            @test states_equal_up_to_phase(ψ_ofd, ψ_none)
        end

        @testset "All strategies produce the same state" begin
            circuit = Gate[
                HGate(1), HGate(2), HGate(3),
                CNOTGate(1, 2),
                TGate(1), TGate(2),
                CNOTGate(2, 3),
                TGate(3)
            ]
            n = 3
            ψ_exact = exact_simulate(circuit, n)

            for strategy in [OFDStrategy(), HybridStrategy(), NoDisentangling()]
                result = simulate_circuit(circuit, n; strategy=strategy)
                ψ = state_vector(result.state)
                @test states_equal_up_to_phase(ψ_exact, ψ)
            end
        end
    end

    @testset "Probability conservation" begin

        @testset "Probabilities sum to 1 after simulation" begin
            circuits = [
                (Gate[HGate(1), TGate(1)], 2),
                (Gate[HGate(1), HGate(2), CNOTGate(1,2), TGate(1), TGate(2)], 2),
                (Gate[HGate(1), HGate(2), HGate(3), TGate(1), TGate(2), TGate(3)], 3),
            ]
            for (circuit, n) in circuits
                result = simulate_circuit(circuit, n; strategy=HybridStrategy())
                total = 0.0
                for bits in 0:(2^n - 1)
                    bv = digits(bits, base=2, pad=n)
                    total += probability(result.state, bv)
                end
                @test total ≈ 1.0 atol=1e-8
            end
        end
    end

    @testset "Expectation value correctness" begin

        @testset "⟨Z⟩ for |0⟩ = +1, ⟨Z⟩ for |1⟩ = −1" begin
            state0 = CAMPSState(1); initialize!(state0)
            @test real(expectation_value_z(state0, 1)) ≈ 1.0 atol=1e-10

            state1 = CAMPSState(1); initialize!(state1)
            apply_gate!(state1, XGate(1))
            @test real(expectation_value_z(state1, 1)) ≈ -1.0 atol=1e-10
        end

        @testset "⟨X⟩ for |+⟩ = +1" begin
            state = CAMPSState(1); initialize!(state)
            apply_gate!(state, HGate(1))
            @test real(expectation_value_x(state, 1)) ≈ 1.0 atol=1e-10
        end

        @testset "⟨Y⟩ for |0⟩ = 0" begin
            state = CAMPSState(1); initialize!(state)
            @test abs(expectation_value_y(state, 1)) < 1e-10
        end

        @testset "⟨Y⟩ for S|+⟩ = |i+⟩ should be +1" begin
            state = CAMPSState(1); initialize!(state)
            apply_gate!(state, HGate(1))
            apply_gate!(state, SGate(1))
            @test real(expectation_value_y(state, 1)) ≈ 1.0 atol=1e-10
        end

        @testset "⟨ZZ⟩ for Bell state = +1" begin
            state = CAMPSState(2); initialize!(state)
            apply_gate!(state, HGate(1))
            apply_gate!(state, CNOTGate(1, 2))
            P = P"ZZ"
            @test real(expectation_value(state, P)) ≈ 1.0 atol=1e-10
        end

        @testset "⟨XX⟩ for Bell state = +1" begin
            state = CAMPSState(2); initialize!(state)
            apply_gate!(state, HGate(1))
            apply_gate!(state, CNOTGate(1, 2))
            P = P"XX"
            @test real(expectation_value(state, P)) ≈ 1.0 atol=1e-10
        end

        @testset "⟨ZI⟩ for Bell state = 0" begin
            state = CAMPSState(2); initialize!(state)
            apply_gate!(state, HGate(1))
            apply_gate!(state, CNOTGate(1, 2))
            @test abs(expectation_value_z(state, 1)) < 1e-10
        end
    end

    @testset "Random circuit correctness (seeded)" begin

        @testset "Random Clifford+T circuit, n=3, t=4" begin
            circuit = random_clifford_t_circuit(3, 4; seed=12345)
            n = 3
            result = simulate_circuit(circuit, n; strategy=NoDisentangling())
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
        end

        @testset "Random Clifford+T circuit, n=4, t=6" begin
            circuit = random_clifford_t_circuit(4, 6; seed=67890)
            n = 4
            result = simulate_circuit(circuit, n; strategy=NoDisentangling())
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
        end

        @testset "Random brickwork circuit, n=4" begin
            circuit = random_brickwork_circuit(4, 3, 0.3; seed=42)
            n = 4
            result = simulate_circuit(circuit, n; strategy=NoDisentangling())
            ψ_camps = state_vector(result.state)
            ψ_exact = exact_simulate(circuit, n)
            @test states_equal_up_to_phase(ψ_exact, ψ_camps)
        end
    end

end
