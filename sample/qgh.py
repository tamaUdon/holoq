import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from constants import Constants
from pointcloud import create_rectangle_points


def _bits_for_size(size: int) -> int:
    if size <= 1:
        return 1
    return (size - 1).bit_length()


def _encode_fixed_point(value: float, frac_bits: int) -> int:
    if value < 0:
        raise ValueError("fixed-point encoding expects non-negative values")
    return int(round(value * (1 << frac_bits)))


def _apply_mcx_on_value(
    qc: QuantumCircuit, control_bits: Sequence[Tuple[int, int]], target
) -> None:
    flips = [qc.qubits[q] for q, bit in control_bits if bit == 0]
    for q in flips:
        qc.x(q)
    qc.mcx([qc.qubits[q] for q, _ in control_bits], target)
    for q in flips:
        qc.x(q)


def _load_value_on_match(
    qc: QuantumCircuit,
    control_bits: Sequence[Tuple[int, int]],
    target_reg: QuantumRegister,
    value: int,
) -> None:
    for bit_idx in range(len(target_reg)):
        if (value >> bit_idx) & 1:
            _apply_mcx_on_value(qc, control_bits, target_reg[bit_idx])


def build_pointcloud_state(
    points: np.ndarray,
    constants: Constants,
    frac_bits: int = 16,
    aj_values: Sequence[int] | None = None,
    measure: bool = False,
) -> QuantumCircuit:
    """
    Prepare |aj>|Pj>\otimes|xj>|yj> using basis encoding.

    aj is optional; when omitted it defaults to 1 for all points.
    Pj is fixed-point rho_j = p^2 / (2 * lambda * zj).
    """
    if aj_values is None:
        aj_values = [1] * len(points)
    if len(aj_values) != len(points):
        raise ValueError("aj_values must match points length")

    x_bits = _bits_for_size(constants.X)
    y_bits = _bits_for_size(constants.Y)

    max_aj = max(aj_values) if aj_values else 0
    aj_bits = max(1, int(max_aj).bit_length())

    rho_values = []
    for _, _, zj in points:
        rho = (constants.pp * constants.pp) / (2.0 * constants.λ * float(zj))
        rho_values.append(_encode_fixed_point(rho, frac_bits))
    max_rho = max(rho_values) if rho_values else 0
    rho_bits = max(1, int(max_rho).bit_length())

    aj_reg = QuantumRegister(aj_bits, "aj")
    rho_reg = QuantumRegister(rho_bits, "rho")
    x_reg = QuantumRegister(x_bits, "x")
    y_reg = QuantumRegister(y_bits, "y")

    regs: List[QuantumRegister] = [aj_reg, rho_reg, x_reg, y_reg]
    if measure:
        cr = ClassicalRegister(aj_bits + rho_bits + x_bits + y_bits, "cr")
        qc = QuantumCircuit(*regs, cr)
    else:
        qc = QuantumCircuit(*regs)

    # Superposition over x and y indices.
    qc.h(x_reg)
    qc.h(y_reg)

    for (xj, yj, zj), aj in zip(points, aj_values):
        x_idx = int(round(float(xj)))
        y_idx = int(round(float(yj)))

        if not (0 <= x_idx < constants.X and 0 <= y_idx < constants.Y):
            raise ValueError("point index out of range")

        control_bits: List[Tuple[int, int]] = []
        for i in range(x_bits):
            control_bits.append((qc.find_bit(x_reg[i]).index, (x_idx >> i) & 1))
        for i in range(y_bits):
            control_bits.append((qc.find_bit(y_reg[i]).index, (y_idx >> i) & 1))

        rho = (constants.pp * constants.pp) / (2.0 * constants.λ * float(zj))
        rho_fp = _encode_fixed_point(rho, frac_bits)

        _load_value_on_match(qc, control_bits, aj_reg, int(aj))
        _load_value_on_match(qc, control_bits, rho_reg, rho_fp)

    if measure:
        qc.measure(range(qc.num_qubits), range(qc.num_clbits))

    return qc


def main() -> None:
    constants = Constants()
    points = create_rectangle_points(constants)
    qc = build_pointcloud_state(points, constants, frac_bits=16, measure=False)
    print(qc)


if __name__ == "__main__":
    main()
