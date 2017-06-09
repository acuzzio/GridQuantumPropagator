'''
First tests suite
'''
from quantumpropagator import astridParser, abs2

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


