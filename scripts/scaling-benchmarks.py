import argparse
import sys

import quex as qx
from quex.utils import run_backend_benchmark


def main():
    parser = argparse.ArgumentParser(description="Quex Hardware Scaling Benchmark Suite")

    # Define the simulator backend
    parser.add_argument("--backend", choices=["numpy", "cupy", "jax"], default="numpy", help="Which hardware backend to benchmark.")

    # Define the scanning mode
    parser.add_argument("--scan", choices=["depth", "qubits"], required=True, help="Which variable to scale up (depth or qubits).")

    # Define the parameters
    parser.add_argument("--fixed", type=int, required=True, help="The value of the variable NOT being scanned.")
    parser.add_argument("--start", type=int, required=True, help="Starting value for the loop.")
    parser.add_argument("--stop", type=int, required=True, help="Stopping value for the loop (inclusive).")
    parser.add_argument("--step", type=int, default=2, help="Step size for the loop.")

    # Optional custom filename
    parser.add_argument("--out", type=str, help="Custom output filename.")

    args = parser.parse_args()

    # 1. Map the CLI string to the actual Python Class
    # We use getattr as a safety net in case a user hasn't installed the GPU extras
    backend_map = {"numpy": getattr(qx, "NumpySimulator", None), "cupy": getattr(qx, "CupySimulator", None), "jax": getattr(qx, "JaxSimulator", None)}

    SimulatorClass = backend_map[args.backend]
    if SimulatorClass is None:
        print(f"Error: '{args.backend}' simulator could not be loaded. Are the hardware extras installed?")
        sys.exit(1)

    # 2. Auto-generate a clean filename (Now includes the backend name!)
    if not args.out:
        args.out = f"time-vs-{args.scan}-{args.backend}-fixed-{args.fixed}.dat"

    print(f"--- Quex Benchmarking ---")
    print(f"Backend: {args.backend.upper()}")
    print(f"Mode: Scanning {args.scan} [{args.start} to {args.stop}, step {args.step}]")
    print(f"Fixed Parameter: {args.fixed}")
    print(f"Saving to: {args.out}\n")

    # 3. Open the file context safely
    with open(args.out, "w") as ff:
        # Write a helpful header for plotting later
        if args.scan == "depth":
            ff.write("# Depth | Execution Time (s)\n")
        else:
            ff.write("# Qubits | Execution Time (s)\n")

        # 4. The Benchmark Loop
        for val in range(args.start, args.stop + 1, args.step):
            if args.scan == "depth":
                num_qubits = args.fixed
                depth = val
            else:
                num_qubits = val
                depth = args.fixed

            # Pass the dynamically selected SimulatorClass
            time = run_backend_benchmark(SimulatorClass, num_qubits=num_qubits, depth=depth, exec_mode="fused", max_fused=4)

            # Write and immediately flush to the physical disk
            ff.write(f"{val} {time:.12f}\n")
            ff.flush()

            print(f"Completed: {args.scan}={val} | Time: {time:.4f}s")


if __name__ == "__main__":
    main()
