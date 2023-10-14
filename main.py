from calc_vac import *
def test_DQD_2e():
    cL_ = c(create=True, site={'site':'L'})      
    cL = c(create=False, site={'site':'L'})
    cR_ = c(create=True, site={'site':'R'})
    cR = c(create=False, site={'site':'R'})
                

    cLu_ = c(create=True, site={'site':'L', 'spin':'up'})
    cLu = c(create=False, site={'site':'L', 'spin':'up'})
    cLd_ = c(create=True, site={'site':'L', 'spin':'down'})
    cLd = c(create=False, site={'site':'L', 'spin':'down'})
    cRu_ = c(create=True, site={'site':'R', 'spin':'up'})
    cRu = c(create=False, site={'site':'R', 'spin':'up'})
    cRd_ = c(create=True, site={'site':'R', 'spin':'down'})
    cRd = c(create=False, site={'site':'R', 'spin':'down'})


    eRu = sp.Symbol('\epsilon_{R\\uparrow}')
    eRd = sp.Symbol('\epsilon_{R\\downarrow}')
    eLu = sp.Symbol('\epsilon_{L\\uparrow}')
    eLd = sp.Symbol('\epsilon_{L\\downarrow}')

    tau = sp.Symbol('tau')
    UL = sp.Symbol('U_L')
    UR = sp.Symbol('U_R')
    V = sp.Symbol('V')


    H = [
        [cRu_, cRu],
        [cRd_, cRd],
        [cLu_, cLu],
        [cLd_, cLd],
        [cLu_, cRu],
        [cRu_, cLu],
        [cLd_, cRd],
        [cRd_, cLd],
        [cLu_, cLu, cLd_, cLd],
        [cRu_, cRu, cRd_, cRd],
        [cL_, cL, cR_, cR]
    ]

    factor = [eRu, eRd, eLu, eLd, -tau, -tau, -tau, -tau, UL, UR, V]

    ket = [
        [cLu_, cLd_], # |11, 00>
        [cRu_, cRd_], # |00, 11>
        [cLu_, cRu_], # |10, 10>
        [cLd_, cRd_], # |01, 01>
        [cLu_, cRd_], # |10, 01>
        [cLd_, cRu_] # |01, 10>
    ]

    bra = [
        [cLd, cLu], # <11, 00|
        [cRd, cRu], # <00, 11|
        [cRu, cLu], # <10, 10|
        [cRd, cLd], # <01, 01|
        [cRd, cLu], # <10, 01|
        [cRu, cLd] # <01, 10|
    ]

    H = calc_Hamiltonian(H, factor, bra, ket)

    print(sp.latex(H))
    
    
def test_TQD_1e_no_spin():
    
    cL_ = c(create=True, site={'site':'L'})      
    cL = c(create=False, site={'site':'L'})
    cC_ = c(create=True, site={'site':'C'})
    cC = c(create=False, site={'site':'C'})
    cR_ = c(create=True, site={'site':'R'})
    cR = c(create=False, site={'site':'R'})
    

    eL = sp.Symbol('\epsilon_{L}')
    eC = sp.Symbol('\epsilon_{C}')
    eR = sp.Symbol('\epsilon_{R}')
   
    


    tau = sp.Symbol('tau')
    UL = sp.Symbol('U_L')
    UR = sp.Symbol('U_R')
    UC = sp.Symbol('U_C')



    H = [
        [cL_, cL],
        [cC_, cC],
        [cR_, cR],
        [cL_, cC],
        [cC_, cL],
        [cC_, cR],
        [cR_, cC]
    ]
    
    factor = [eL, eC, eR, -tau, -tau, -tau, -tau]
    
    bra = [
        [cL], # <1, 0, 0|
        [cC], # <0, 1, 0|
        [cR] # <0, 0, 1|
    ]
    
    ket = [
        [cL_], # |1, 0, 0>
        [cC_], # |0, 1, 0>
        [cR_] # |0, 0, 1>
    ]
    
    H = calc_Hamiltonian(H, factor, bra, ket)
    
    print(sp.latex(H))
    
if __name__ == '__main__':
    test_TQD_1e_no_spin()
    
       