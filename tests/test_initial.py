'''
First tests suite
'''
import numpy as np
from quantumpropagator import astridParser, abs2, chunksOf, chunksOfList

def test_astridParser():
    '''
    Test for the array shapes returned by the parser of Astrid data ...
    '''
    (dist,newEne,newDipo,newNAC,newGac) = astridParser(4,400,'tests/test_files/ast2')
    test1 = dist.shape == (400,)
    test2 = newEne.shape == (400, 4)
    test3 = newDipo.shape == (400, 1, 4, 4)
    test4 = newNAC.shape == (400, 4, 4)
    test5 = newGac.shape == (400, 4, 4)
    assert all([test1,test2,test3,test4,test5])

def test_abs2():
    '''
    Test for the function abs2
    '''
    assert abs2(3.2+9.3j) == 96.73000000000002

def test_chunksOf():
    '''
    Tests for the functions chunksOf and chunksOfList
    '''
    test1 = (list(chunksOf(np.arange(10),4))[0] == [np.array([0, 1, 2,
        3])]).all()
    #test2 = chunksOfList(range(10),4) == [range(0, 4),
    #        range(4, 8), range(8, 10)]
    test2 = list(chunksOfList(range(10),4))[0] == range(0, 4)
    test3 = len(list(chunksOf(np.arange(10),4))) == 3
    print(test1,test2,test3)
    assert all([test1,test2,test3])


