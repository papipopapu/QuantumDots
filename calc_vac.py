from typing import List, Tuple, Dict, Union
import sympy as sp
sp.init_printing(use_unicode=True) # allow LaTeX printing
class c:
    def __init__(self, create: bool = False, site: Dict = {'spin':'up', 'site':'down'}):
        self.create = create
        self.site = site
    
    def __str__(self):
        return str(self.create) + str(self.site)
    
    
def delta_site(a: c, b: c) -> bool:

    if len(a.site) > len(b.site):
        for key in b.site.keys():
            if key not in a.site.keys() or a.site[key] != b.site[key]:
                return False
    else:
        for key in a.site.keys():
            if key not in b.site.keys() or a.site[key] != b.site[key]:
                return False
            
    return True

def split_c_list(c_list: List[c]) -> Tuple[List[Tuple[c, int]], List[Tuple[c, int]]]:

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
            


    
    
def calc_Hamiltonian(H: List[List[c]], factor: List[sp.Symbol], bra: List[List[c]], ket: List[List[c]]) -> sp.Matrix:
        
    result = []
    for i, bra_i in enumerate(bra):
        result.append([])
        for j, ket_j in enumerate(ket):
            result[i].append(0)
            for k, H_k in enumerate(H):
                result[i][j] += calc_vac(bra_i + H_k + ket_j) * factor[k]
    return sp.Matrix(result)


    

        
    
    
    