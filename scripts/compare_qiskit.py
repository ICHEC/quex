import numpy as np
from qiskit import qasm3, transpile
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector

import quex as qx


def verify_against_qiskit(num_qubits=6, depth=10, backend="numpy"):
    print(f"--- Qiskit vs Quex Verification | {num_qubits} Qubits, Depth {depth} ---")

    simulator = {"numpy": qx.NumpySimulator, "cupy": qx.CupySimulator, "jax": qx.JaxSimulator}
    if backend not in simulator.keys():
        raise ValueError(f"unknown backend {backend}. should be one of numpy, cupy, jax")
    # 1. Generate Random Circuit in Qiskit
    # We restrict the basis gates to standard ones to ensure clean QASM translation
    qc_qiskit_raw = random_circuit(num_qubits, depth, measure=False)

    # transpile to standard gates #
    qc_qiskit = transpile(qc_qiskit_raw, basis_gates=["rx", "ry", "rz", "cx", "h", "x", "y", "z", "s", "t"])

    # 2. Extract Qiskit's Exact Statevector
    # Using quantum_info.Statevector is faster and cleaner than spinning up the Aer simulator
    state_qiskit_raw = np.array(Statevector(qc_qiskit))

    # --- FIXING QISKIT'S ENDIANNESS ---
    # Qiskit is Little-Endian. We must reshape, reverse the axes, and flatten
    # so it matches standard textbook Big-Endian physics engines.
    state_qiskit_physics = state_qiskit_raw.reshape([2] * num_qubits)
    state_qiskit_physics = state_qiskit_physics.transpose(range(num_qubits)[::-1]).flatten()

    # 3. Export to OpenQASM 3
    qasm_str = qasm3.dumps(qc_qiskit)
    print(qasm_str)
    # 4. Import into Quex
    # (Note: Replace 'from_qasm' with whatever API method you built to parse QASM in Quex)
    try:
        qc_quex = qx.Circuit.from_qasm(qasm_str)
    except AttributeError:
        print("Error: Make sure to use Quex's actual OpenQASM parsing function here!")
        return

    # 5. Run Quex in your highly optimized Fused mode
    qc_quex.simulator = simulator[backend](exec_mode="fused", max_fused_qubits=4)
    state_quex = qc_quex.run()

    # Flatten just in case your state is still stored as an N-dimensional tensor
    # state_quex = np.array(qc_quex.statevector).flatten()

    # --- THE MATHEMATICAL COMPARISON ---

    # Calculate State Fidelity |<psi|phi>|^2 (Ignores global phase)
    fidelity = np.abs(np.vdot(state_qiskit_physics, state_quex)) ** 2
    print(f"Quantum Fidelity: {fidelity:.8f}")

    # Calculate strict element-wise equality (Accounts for float rounding errors)
    # We divide out the phase of the first non-zero element to align global phases manually
    idx = np.argmax(np.abs(state_quex) > 1e-8)
    phase_shift = state_qiskit_physics[idx] / state_quex[idx] if state_quex[idx] != 0 else 1.0

    is_exact = np.allclose(state_qiskit_physics, state_quex * phase_shift, atol=1e-7)

    if is_exact and fidelity > 0.999999:
        print("✅ EXACT MATCH: Quex physics engine is mathematically perfect compared to Qiskit.")
    elif fidelity > 0.999999:
        print("⚠️ FIDELITY MATCH: Probabilities match, but raw amplitudes differ. Check your gate matrix definitions for global phase differences.")
    else:
        print("❌ FAIL: Statevectors do not match. A gate translation or routing error occurred.")


# Run the validation!
if __name__ == "__main__":
    verify_against_qiskit(num_qubits=6, depth=10)
