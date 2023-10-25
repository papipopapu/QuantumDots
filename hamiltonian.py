from typing import List, Tuple, Dict, Any
from numbers import Number

class c_internal:
    def __init__(self, create: bool = False, site: Dict = {'spin':'up', 'site':'down'}):
        self.create = create
        self.site = site
    def d(self):
        return c_internal(not self.create, self.site)
    def __str__(self):
        return str(self.create) + str(self.site)
    
    
def delta_site(a: c_internal, b: c_internal) -> bool:

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

    creation_list = []
    annihilation_list = []
    for i, c_i in enumerate(c_list):
        if c_i.create:
            creation_list.append((c_i, i))
        else:
            annihilation_list.append((c_i, i))
    return creation_list, annihilation_list

def calc_vac(c_list):
  
    result = 0
    creation_list, anhilation_list = split_c_list(c_list)
    
    if len(creation_list) != len(anhilation_list) or creation_list == []:
        return 0
    elif len(creation_list) == 1:
        if delta_site(creation_list[0][0], anhilation_list[0][0]) and creation_list[0][1] > anhilation_list[0][1]:
            return 1
        else:
            return 0   
        
    c_i, i = anhilation_list[0]
    for c_j, j in creation_list:
        if delta_site(c_i, c_j) and j > i:
            result += calc_vac(c_list[:i] + c_list[i+1:j] + c_list[j+1:]) * (-1)**(abs(i-j)-1)
            
    return result
            


    
    
def calc_Hamiltonian_old(H: List[List[c_internal]], factor: List[Number], bra: List[List[c_internal]], ket: List[List[c_internal]]) -> List[List[Number]]:
        
    result = []
    for i, bra_i in enumerate(bra):
        result.append([])
        for j, ket_j in enumerate(ket):
            result[i].append(0)
            for k, H_k in enumerate(H):
                result[i][j] += calc_vac(bra_i + H_k + ket_j) * factor[k]
    return result

def conjugate(c_list: List[c_internal]) -> List[c_internal]:
    # conjugate and reverse
    result = []
    for c_i in c_list:
        result.append(c_i.d())
    return result[::-1  ]

def calc_Hamiltonian(H: List[Tuple[Number, List[c_internal]]], basis: List[List[c_internal]]) -> List[List[Number]]:
        
    result = []
    for i, bra_i in enumerate(basis):
        bra_i = conjugate(bra_i)
        result.append([])
        for j, ket_j in enumerate(basis):
            result[i].append(0)
            for factor_k, H_k in H:
                result[i][j] += calc_vac(bra_i + H_k + ket_j) * factor_k
    return result
    

        
class Space:
    def __init__(self, name: str | List[str], span: List | List[List]):
        if not isinstance(name, list):
            name = [name]
        self.name = name
        for i, el in enumerate(span):
            if not isinstance(el, list):
                span[i] = [span[i]]

        self.span = span
        

    def __mul__(self, other):
        new_span = [] 
        for ei in self.span:
            for ej in other.span:
                new_span.append(ei + ej)      
        new_space = Space(self.name + other.name, new_span)
        return new_space
    
    def dim(self):
        return len(self.name)
    
    def creations(self) -> List[c_internal]:
        cs = []
        for el in self.span:
            site = dict(zip(self.name, el))
            cs.append(c_internal(True, site))
        return cs
    def annihilations(self) -> List[c_internal]:
        cs = []
        for el in self.span:
            site = dict(zip(self.name, el))
            cs.append(c_internal(False, site))
        return cs
    
