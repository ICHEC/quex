# src/quex/parser.py
import math
from typing import Any, Dict

import openqasm3
from openqasm3 import ast

_QASM_CONSTANTS = {
    "pi": math.pi,
    "π": math.pi,
    "tau": math.tau,
    "τ": math.tau,
    "euler": math.e,
    "ℇ": math.e,
}

_QASM_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
}


def _eval_qasm_expr(node):
    """Recursively evaluate OpenQASM 3 AST expression nodes into floats or symbol names."""
    if isinstance(node, (ast.FloatLiteral, ast.IntegerLiteral)):
        return float(node.value)

    if isinstance(node, ast.Identifier):
        if node.name in _QASM_CONSTANTS:
            return _QASM_CONSTANTS[node.name]
        # Preserve symbolic input parameters (e.g., `input float[64] theta;`)
        return node.name

    if isinstance(node, ast.UnaryExpression):
        val = _eval_qasm_expr(node.expression)
        op = node.op.name if hasattr(node.op, "name") else str(node.op)
        if op in ("MINUS", "-"):
            return -val
        if op in ("PLUS", "+"):
            return val

    if isinstance(node, ast.BinaryExpression):
        lhs = _eval_qasm_expr(node.lhs)
        rhs = _eval_qasm_expr(node.rhs)
        op = node.op.name if hasattr(node.op, "name") else str(node.op)
        if op in ("PLUS", "+"):
            return lhs + rhs
        if op in ("MINUS", "-"):
            return lhs - rhs
        if op in ("TIMES", "*"):
            return lhs * rhs
        if op in ("DIVIDE", "/"):
            return lhs / rhs
        if op in ("POWER", "**"):
            return lhs**rhs

    if isinstance(node, ast.FunctionCall):
        func_name = node.name.name
        if func_name in _QASM_FUNCS and len(node.arguments) == 1:
            return _QASM_FUNCS[func_name](_eval_qasm_expr(node.arguments[0]))

    raise ValueError(f"Unsupported OpenQASM 3 AST expression: {type(node).__name__} ({node})")


def parse_qasm_string(qasm_string: str) -> Dict[str, Any]:
    """
    Parses an OpenQASM 3 string into a list of Quex operations, and global phase.

    Returns a list of dictionaries, where each dict represents a gate
    and the target qubits it acts upon.

    Later fix: Recursively evaluate OpenQASM 3 AST expression nodes into floats or symbol names.
    """
    try:
        program = openqasm3.parse(qasm_string)
    except Exception as e:
        raise ValueError(f"Failed to parse OpenQASM: {e}")

    registers = {}
    circuit_ops = []
    global_phase = 0.0

    for statement in program.statements:
        if isinstance(statement, ast.QubitDeclaration):
            reg_name = statement.qubit.name
            reg_size = int(_eval_qasm_expr(statement.size)) if statement.size is not None else 1
            registers[reg_name] = reg_size

        elif isinstance(statement, ast.QuantumGate):
            gate_name = statement.name.name

            # 1. Extract Target Qubits (handles both indexed `q[0]` and standalone `q`)
            target_qubits = []
            for q in statement.qubits:
                if isinstance(q, ast.IndexedIdentifier):
                    qubit_name = q.name.name
                    qubit_index = int(_eval_qasm_expr(q.indices[0][0]))
                    target_qubits.append((qubit_name, qubit_index))
                elif isinstance(q, ast.Identifier):
                    target_qubits.append((q.name, 0))

            # 2. Evaluate all parameter AST nodes (literals, constants, unary/binary math, symbols)
            params = [_eval_qasm_expr(arg) for arg in statement.arguments] if statement.arguments else []

            circuit_ops.append(
                {
                    "gate": gate_name,
                    "params": params,
                    "targets": target_qubits,
                }
            )

        elif isinstance(statement, ast.QuantumPhase):
            global_phase += float(_eval_qasm_expr(statement.argument))

    return {
        "registers": registers,
        "operations": circuit_ops,
        "global_phase": global_phase,
    }
