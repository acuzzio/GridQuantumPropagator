'''
This is not a main anymore. I keep it in case I need singlepoint Integrator
'''

#import quantumpropagator.h5Reader as hf
import quantumpropagator.Propagator as Pr
import quantumpropagator.pulse as pp
import quantumpropagator.SinglePointIntegrator as spInt

##################################################
#                                                #
#                    MAIN                        #
#                                                #
##################################################

def main():
    '''
    singlePointIntegration(h5file, timestepDT, timestepN, Graph, OutputFile)
    h5file            String      Name of the single point h5 file
    timestepDT        Double      dt
    timestepN         Double      number of steps
    pulse             [[Double]]  [t1[x,y,z],t2[x,y,z]] extern electromagnetic field at each t(x,y,z)
    Graph             True/False  To make the graph
    OutputFile        True/False  To write an external file
    '''

    spInt.singlePointIntegration('LiH.rassi.h5', 0.04, 1000, pp.specificPulse, True, True)

if __name__ == "__main__":
    main()
