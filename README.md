# QuantumDots

A Python library for simulating quantum dot systems, including double quantum dots (DQD) and triple quantum dots (TQD) with various physical effects like spin-orbit coupling, Coulomb interactions, and Floquet dynamics.

This project was developed as part of a research collaboration with the **Instituto de Ciencia de Materiales de Madrid (ICMM-CSIC)**, focusing on the study of quantum dots and their applications in quantum information.

## Overview

Quantum dots are nanoscale semiconductor structures that confine electrons in all three spatial dimensions, creating discrete energy levels similar to atoms ("artificial atoms"). This project provides tools for:

- **Hamiltonian construction** using second quantization formalism
- **Open system dynamics** via Lindblad master equations
- **Floquet theory** for periodically driven systems
- **Numerical simulations** using QuTiP

## Physics Background

### Quantum Dot Systems

Quantum dots can be coupled together to form multi-dot systems:

- **Double Quantum Dots (DQD)**: Two coupled quantum dots, useful for studying charge and spin qubits
- **Triple Quantum Dots (TQD)**: Three coupled quantum dots, enabling richer physics including dark states and interference effects

### Key Physical Effects

1. **Tunneling**: Electron hopping between dots, characterized by tunneling amplitude τ
2. **Spin-Orbit Coupling**: Spin-flip tunneling processes, enabling spin manipulation without magnetic fields
3. **Coulomb Interaction**: Electron-electron repulsion (intradot U, interdot V)
4. **Zeeman Effect**: Energy splitting due to magnetic fields
5. **Floquet Dynamics**: Response to periodic driving fields (AC voltages)

### Lindblad Master Equation

For open systems coupled to electron reservoirs, we use the Lindblad master equation:

```
dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - 1/2 {L_k† L_k, ρ})
```

where ρ is the density matrix, H is the Hamiltonian, and L_k are jump operators describing coupling to reservoirs.

## Installation

### Requirements

- Python 3.8+
- NumPy
- SciPy
- SymPy
- QuTiP
- Matplotlib

### Install Dependencies

```bash
pip install numpy scipy sympy qutip matplotlib
```

For publication-quality plots (optional):
```bash
pip install scienceplots
```

## Project Structure

```
ICMM-Quantum-Dots/
├── code/
│   ├── lib/                    # Core library modules
│   │   ├── __init__.py
│   │   ├── hamiltonian.py      # Second quantization framework
│   │   ├── density.py          # Lindblad master equation tools
│   │   └── qutipDots.py        # QuTiP-based numerical tools
│   ├── examples/               # Example simulations
│   │   ├── double_quantum_dot/ # DQD examples
│   │   ├── triple_quantum_dot/ # TQD examples
│   │   ├── floquet/            # Floquet dynamics examples
│   │   └── lindblad/           # Open system dynamics examples
│   └── README.md               # Module documentation
├── README.md
└── LICENSE
```

## Usage

### Basic Example: Two-Electron DQD Hamiltonian

```python
from code.lib.hamiltonian import Space, calc_Hamiltonian
from sympy import Symbol

# Define the Hilbert space
spin = Space('spin', ['up', 'down'])
location = Space('location', ['left', 'right'])
spin_location = location * spin

# Get creation and annihilation operators
cLu_, cLd_, cRu_, cRd_ = spin_location.creations()
cLu, cLd, cRu, cRd = spin_location.annihilations()

# Define symbolic parameters
eL = Symbol('epsilon_L')
eR = Symbol('epsilon_R')
tau = Symbol('tau')
U = Symbol('U')

# Define Hamiltonian terms: (coefficient, [operators])
H = [
    # On-site energies
    (eL, [cLu_, cLu]),
    (eL, [cLd_, cLd]),
    (eR, [cRu_, cRu]),
    (eR, [cRd_, cRd]),
    # Tunneling
    (-tau, [cLu_, cRu]),
    (-tau, [cRu_, cLu]),
    (-tau, [cLd_, cRd]),
    (-tau, [cRd_, cLd]),
    # Coulomb interaction
    (U, [cLu_, cLu, cLd_, cLd]),
    (U, [cRu_, cRu, cRd_, cRd]),
]

# Define basis states (two electrons)
basis = [
    [(1, [cLu_, cLd_])],  # |↑↓, 00⟩
    [(1, [cRu_, cRd_])],  # |00, ↑↓⟩
    [(1, [cLu_, cRu_])],  # |↑0, ↑0⟩
    [(1, [cLd_, cRd_])],  # |0↓, 0↓⟩
]

# Calculate Hamiltonian matrix
H_matrix = calc_Hamiltonian(H, basis)
```

### Example: QuTiP-based TQD Simulation

```python
from code.lib.qutipDots import f_destroy, eqdot_state, red_H
import numpy as np
import qutip as qt

# Create fermionic operators for 3-site spinless system
c1, c2, c3 = [f_destroy(3, i) for i in range(3)]
c1_, c2_, c3_ = [c.dag() for c in [c1, c2, c3]]

# Parameters
e1, e2, e3 = 0, 0, 0  # On-site energies
tau = 0.1             # Tunneling amplitude

# Build Hamiltonian
H = (e1 * c1_ * c1 + e2 * c2_ * c2 + e3 * c3_ * c3
     - tau * (c1_ * c2 + c2_ * c1 + c2_ * c3 + c3_ * c2))

# Define allowed states (1 electron max)
states = [
    eqdot_state([0, 0, 0]),  # Empty
    eqdot_state([1, 0, 0]),  # Electron on dot 1
    eqdot_state([0, 1, 0]),  # Electron on dot 2
    eqdot_state([0, 0, 1]),  # Electron on dot 3
]

# Reduce Hamiltonian to this subspace
H_red = qt.Qobj(red_H(H, states))

# Time evolution
tlist = np.linspace(0, 100, 1000)
psi0 = qt.basis(4, 1)  # Start with electron on dot 1
result = qt.mesolve(H_red, psi0, tlist, [], [])
```

## Key Modules

### hamiltonian.py
- `Space`: Define Hilbert spaces and their tensor products
- `calc_Hamiltonian`: Compute Hamiltonian matrix elements
- Fermionic operator algebra using Wick's theorem

### density.py
- `get_Lamb_matrix`: Calculate decoherence rates
- `get_density_equation`: Symbolic Lindblad equation
- `get_density_equation_vect`: Vectorized Liouvillian

### qutipDots.py
- `f_destroy`, `f_create`: Fermionic operators with Jordan-Wigner
- `eqdot_state`: Construct Fock states
- `get_Liouville`: Numerical Liouvillian construction
- `red_H`: Hamiltonian reduction to subspaces

## Examples

The `code/examples/` directory contains working examples:

- **double_quantum_dot/**: DQD with 2 electrons, spin-orbit coupling
- **triple_quantum_dot/**: TQD with spin, spin-flip tunneling
- **floquet/**: Periodically driven systems, quasienergy spectra
- **lindblad/**: Open system dynamics, steady-state currents

## Physical Units

The code uses **natural units** where ℏ = 1. Typical energy scales:
- Tunneling amplitude τ: ~10-100 μeV
- Coulomb interaction U: ~1-10 meV
- Interdot Coulomb V: ~10-100 μeV
- Tunnel rates Γ: ~τ/10

## References

Key physics concepts implemented in this code:

1. Second quantization and Wick's theorem for fermionic systems
2. Jordan-Wigner transformation for fermionic operators
3. Lindblad master equation for open quantum systems
4. Floquet theory for periodically driven systems

## Acknowledgements

This research was conducted in collaboration with the **Instituto de Ciencia de Materiales de Madrid (ICMM)**, part of the Spanish National Research Council (CSIC). The work focuses on theoretical modeling of quantum dot systems for applications in quantum computing and spintronics.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Joel Martínez, 2023
