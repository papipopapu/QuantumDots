"""
Hamiltonian module for quantum dot systems.

This module provides tools for constructing and calculating Hamiltonians
for quantum dot systems using second quantization formalism.

The module implements:
- Fermionic creation and annihilation operators
- Fock space representation of quantum states
- Hamiltonian matrix element calculations
- Multi-site quantum dot systems with arbitrary spin configurations
"""

from typing import List, Tuple, Dict, Any
from numbers import Number
from numpy import conjugate as scalar_conjugate, zeros, ndarray


class c_internal:
    """
    Internal representation of a fermionic creation/annihilation operator.
    
    This class represents a single fermionic operator acting on a specific
    quantum state (site) in the system. The operator can be either a creation
    operator (create=True) or an annihilation operator (create=False).
    
    Attributes:
        create: Boolean indicating if this is a creation operator
        site: Dictionary mapping quantum numbers to their values (e.g., {'spin': 'up', 'location': 'left'})
    """
    
    def __init__(self, create: bool = False, site: Dict = {'spin': 'up', 'site': 'down'}):
        """
        Initialize a fermionic operator.
        
        Args:
            create: True for creation operator, False for annihilation operator
            site: Dictionary of quantum numbers defining the site
        """
        self.create = create
        self.site = site
        
    def d(self):
        """
        Return the Hermitian conjugate (dagger) of this operator.
        
        For fermionic operators:
        - Creation operator dagger -> Annihilation operator
        - Annihilation operator dagger -> Creation operator
        
        Returns:
            c_internal: The conjugate operator
        """
        return c_internal(not self.create, self.site)
    
    def __str__(self):
        return str(self.create) + str(self.site)


def delta_site(a: c_internal, b: c_internal) -> bool:
    """
    Check if two operators act on the same site (Kronecker delta).
    
    This function compares the quantum numbers of two operators
    to determine if they refer to the same fermionic mode.
    
    Args:
        a: First operator
        b: Second operator
        
    Returns:
        bool: True if operators act on the same site
    """
    if len(a.site) > len(b.site):
        for key in b.site.keys():
            if key not in a.site.keys() or a.site[key] != b.site[key]:
                return False
    else:
        for key in a.site.keys():
            if key not in b.site.keys() or a.site[key] != b.site[key]:
                return False
            
    return True


def split_c_list(c_list: List[c_internal]) -> Tuple[List[Tuple[c_internal, int]], List[Tuple[c_internal, int]]]:
    """
    Split a list of operators into creation and annihilation operators.
    
    Args:
        c_list: List of fermionic operators
        
    Returns:
        Tuple containing:
        - List of (creation operator, original index) tuples
        - List of (annihilation operator, original index) tuples
    """
    creation_list = []
    annihilation_list = []
    for i, c_i in enumerate(c_list):
        if c_i.create:
            creation_list.append((c_i, i))
        else:
            annihilation_list.append((c_i, i))
    return creation_list, annihilation_list


def calc_vac(c_list: List[c_internal]):
    """
    Calculate the vacuum expectation value of a product of operators.
    
    Uses Wick's theorem to evaluate <0|c1 c2 ... cn|0>.
    The result is non-zero only if there is an equal number of
    creation and annihilation operators that can be paired.
    
    Args:
        c_list: List of fermionic operators in normal order
        
    Returns:
        The vacuum expectation value (0, 1, or -1 accounting for fermionic signs)
    """
    result = 0
    creation_list, annihilation_list = split_c_list(c_list)
    
    # VEV is zero unless equal numbers of creation and annihilation operators
    if len(creation_list) != len(annihilation_list) or creation_list == []:
        return 0
    elif len(creation_list) == 1:
        # Base case: single pair
        if delta_site(creation_list[0][0], annihilation_list[0][0]) and creation_list[0][1] > annihilation_list[0][1]:
            return 1
        else:
            return 0
    
    # Recursive case: apply Wick's theorem
    c_i, i = annihilation_list[0]
    for c_j, j in creation_list:
        if delta_site(c_i, c_j) and j > i:
            # Contract operators and account for fermionic sign
            result += calc_vac(c_list[:i] + c_list[i+1:j] + c_list[j+1:]) * (-1)**(abs(i-j)-1)
            
    return result


def conjugate(c_list: List[c_internal]) -> List[c_internal]:
    """
    Compute the Hermitian conjugate of a product of operators.
    
    For fermionic operators: (c1 c2 ... cn)† = cn† ... c2† c1†
    The order is reversed and each operator is conjugated.
    
    Args:
        c_list: List of operators
        
    Returns:
        List of conjugated operators in reversed order
    """
    result = []
    for c_i in c_list:
        result.append(c_i.d())
    return result[::-1]


def calc_Hamiltonian(H: List[Tuple[Number, List[c_internal]]], basis: List[List[Tuple[Number, List[c_internal]]]]) -> ndarray:
    """
    Calculate the Hamiltonian matrix in a given basis.
    
    The Hamiltonian is specified as a sum of terms, each consisting of
    a coefficient and a product of creation/annihilation operators.
    The basis states are specified as linear combinations of
    products of creation operators acting on the vacuum.
    
    Args:
        H: List of (coefficient, [operators]) tuples defining the Hamiltonian
        basis: List of basis states, each state is a list of (coefficient, [creation operators]) tuples
        
    Returns:
        ndarray: The Hamiltonian matrix H_ij = <i|H|j>
    """
    result = zeros((len(basis), len(basis)), dtype=object)
    
    for i, bra_i in enumerate(basis):
        for factor_i, bra_i_el in bra_i:  # Each element of the bra
            bra_i_el = conjugate(bra_i_el)
            factor_i = scalar_conjugate(factor_i)
            
            for j, ket_j in enumerate(basis):
                for factor_j, ket_j_el in ket_j:  # Each element of the ket
                    for factor_k, H_k in H:
                        # Calculate <bra|H|ket> using vacuum expectation value
                        result[i][j] += factor_i * factor_j * calc_vac(bra_i_el + H_k + ket_j_el) * factor_k
                        
    return result


class Space:
    """
    Represents a Hilbert space for quantum dot systems.
    
    A Space is defined by a name (or list of names for tensor products)
    and a span of basis states. For example, a spin space could be
    Space('spin', ['up', 'down']).
    
    Spaces can be multiplied together to form tensor product spaces,
    useful for multi-site systems with spin degrees of freedom.
    
    Attributes:
        name: Name(s) of the space (e.g., 'spin', 'location')
        span: List of basis state labels
    """
    
    def __init__(self, name: str | List[str], span: List | List[List]):
        """
        Initialize a Hilbert space.
        
        Args:
            name: Name of the space or list of names for composite spaces
            span: List of basis state labels or list of lists for composite spaces
        """
        if not isinstance(name, list):
            name = [name]
        self.name = name
        
        # Ensure span elements are lists for consistency
        for i, el in enumerate(span):
            if not isinstance(el, list):
                span[i] = [span[i]]
        self.span = span

    def __mul__(self, other):
        """
        Tensor product of two spaces.
        
        Creates a new Space representing the tensor product of self and other.
        The resulting space has combined quantum numbers and all possible
        combinations of basis states.
        
        Args:
            other: Another Space to form tensor product with
            
        Returns:
            Space: The tensor product space
        """
        new_span = []
        for ei in self.span:
            for ej in other.span:
                new_span.append(ei + ej)
        new_space = Space(self.name + other.name, new_span)
        return new_space
    
    def dim(self):
        """Return the number of quantum number labels."""
        return len(self.name)
    
    def creations(self) -> List[c_internal]:
        """
        Generate creation operators for all basis states in this space.
        
        Returns:
            List[c_internal]: List of creation operators c†_i for each basis state
        """
        cs = []
        for el in self.span:
            site = dict(zip(self.name, el))
            cs.append(c_internal(True, site))
        return cs
    
    def annihilations(self) -> List[c_internal]:
        """
        Generate annihilation operators for all basis states in this space.
        
        Returns:
            List[c_internal]: List of annihilation operators c_i for each basis state
        """
        cs = []
        for el in self.span:
            site = dict(zip(self.name, el))
            cs.append(c_internal(False, site))
        return cs
