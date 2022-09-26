'''
Created on Oct 11, 2016

@author: cristi
'''

from stl import SignalSpace, STL
import sys

def get_specification_ast(name):
    if name == 'double_integrator_spec':
        return double_integrator_spec()
    elif name == 'rear_wheel_car_spec':
        return rear_wheel_car_spec()
    raise ValueError('Unknown specification "%s"!', str(name))

def double_integrator_spec():
    
    # Autonmious driving specs: 

    # space = SignalSpace(var_names=['x1', 'x2'], bounds=[[0, 4.5], [-1, 1.5]])
    # # Region 1 (Eventually the trajectroy must enter the region within [0-1]seconds)
    # pred1 = STL(STL.PRED, space=space, mu=0.1, rel=STL.GREATER, var_name='x1')
    # pred2 = STL(STL.PRED, space=space, mu=0.4, rel=STL.LESS, var_name='x1')
    # pred3 = STL(STL.PRED, space=space, mu=0.1, rel=STL.GREATER, var_name='x2')
    # pred4 = STL(STL.PRED, space=space, mu=0.2, rel=STL.LESS, var_name='x2')
    # conj1 = STL(STL.AND, space=space, subformulae=[pred1, pred2, pred3, pred4])
    # event1 = STL(STL.EVENTUALLY, space=space, low=0, high=2, subformula=conj1)
    # # Region 2 (Always the trajectroy must be in the region within [1-3]seconds) Start accelrating to show the intent of passing 
    # pred5 = STL(STL.PRED, space=space, mu=0.4, rel=STL.GREATER, var_name='x1')
    # pred6 = STL(STL.PRED, space=space, mu=1, rel=STL.LESS, var_name='x1')
    # pred7 = STL(STL.PRED, space=space, mu=.15, rel=STL.GREATER, var_name='x2')
    # pred8 = STL(STL.PRED, space=space, mu=.5, rel=STL.LESS, var_name='x2')
    # conj2 = STL(STL.AND, space=space, subformulae=[pred5, pred6, pred7, pred8])
    # always1 = STL(STL.ALWAYS, space=space, low=2, high=4, subformula=conj2)
    # # Region 3 (Always the trajectroy must be in the region within [3.5-4.5]seconds) passing, "be fast" 
    # pred9 = STL(STL.PRED, space=space, mu=1.5, rel=STL.GREATER, var_name='x1')
    # pred10 = STL(STL.PRED, space=space, mu=3, rel=STL.LESS, var_name='x1')
    # pred11 = STL(STL.PRED, space=space, mu=.3, rel=STL.GREATER, var_name='x2')
    # pred12 = STL(STL.PRED, space=space, mu=.7, rel=STL.LESS, var_name='x2')
    # conj3 = STL(STL.AND, space=space, subformulae=[pred9, pred10, pred11, pred12])
    # always2 = STL(STL.EVENTUALLY, space=space, low=4, high=4.5, subformula=conj3)
    # # Region 4 (Always the trajectroy must be in the region within [5-6.5]seconds) after passing  
    # pred13 = STL(STL.PRED, space=space, mu=3.5, rel=STL.GREATER, var_name='x1')
    # pred14 = STL(STL.PRED, space=space, mu=4, rel=STL.LESS, var_name='x1')
    # pred15 = STL(STL.PRED, space=space, mu=.1, rel=STL.GREATER, var_name='x2')
    # pred16 = STL(STL.PRED, space=space, mu=.3, rel=STL.LESS, var_name='x2')
    # conj4 = STL(STL.AND, space=space, subformulae=[pred13, pred14, pred15, pred16])
    # always3 = STL(STL.EVENTUALLY, space=space, low=5, high=6.5, subformula=conj4)

    # # Region 5 (eventially the trajectroy must be in the region within [8-15]seconds) ensuring that we passed:   
    # pred17 = STL(STL.PRED, space=space, mu=3.5, rel=STL.GREATER, var_name='x1')
    # pred18 = STL(STL.PRED, space=space, mu=4, rel=STL.LESS, var_name='x1')
    # pred19 = STL(STL.PRED, space=space, mu=.1, rel=STL.GREATER, var_name='x2')
    # pred20 = STL(STL.PRED, space=space, mu=.3, rel=STL.LESS, var_name='x2')
    # conj5 = STL(STL.AND, space=space, subformulae=[pred17, pred18, pred19, pred20])
    # event2 = STL(STL.EVENTUALLY, space=space, low=8, high=15, subformula=conj5)

    # # Obstacle Region (Avoid for all time): 
    # pred21 = STL(STL.PRED, space=space, mu=1.8, rel=STL.LESS, var_name='x1')
    # pred22 = STL(STL.PRED, space=space, mu=2.5, rel=STL.GREATER, var_name='x1')
    # pred23 = STL(STL.PRED, space=space, mu=-0.1, rel=STL.LESS, var_name='x2')
    # pred24 = STL(STL.PRED, space=space, mu=0.1, rel=STL.GREATER, var_name='x2')
    # disj1 = STL(STL.OR, space=space, subformulae=[pred21, pred22, pred23, pred24])
    # always4 = STL(STL.ALWAYS, space=space, low=0, high=10, subformula=disj1)

    # spec = STL(STL.AND, space=space, subformulae=[event1, event2,always1, always2, always3, always4])



    # Old-new Specs: 
    space = SignalSpace(var_names=['x1', 'x2'], bounds=[[0, 4], [-1, 1]])
    
    pred1 = STL(STL.PRED, space=space, mu=3.5, rel=STL.GREATER, var_name='x1')
    pred2 = STL(STL.PRED, space=space, mu=4.0, rel=STL.LESS, var_name='x1')
    pred3 = STL(STL.PRED, space=space, mu=-0.2, rel=STL.GREATER, var_name='x2')
    pred4 = STL(STL.PRED, space=space, mu=0.2, rel=STL.LESS, var_name='x2')
    conj1 = STL(STL.AND, space=space, subformulae=[pred1, pred2, pred3, pred4])
    event1 = STL(STL.EVENTUALLY, space=space, low=1, high=15, subformula=conj1)
    
    pred55 = STL(STL.PRED, space=space, mu=-0.1, rel=STL.GREATER, var_name='x1')
    pred66 = STL(STL.PRED, space=space, mu=2.0, rel=STL.LESS, var_name='x1')
    pred5 = STL(STL.PRED, space=space, mu=-0.5, rel=STL.GREATER, var_name='x2')
    pred6 = STL(STL.PRED, space=space, mu=0.5, rel=STL.LESS, var_name='x2')
    conj2 = STL(STL.AND, space=space, subformulae=[pred5, pred6,pred55, pred66])
    always1 = STL(STL.ALWAYS, space=space, low=0, high=1, subformula=conj2)
    
    pred55_ = STL(STL.PRED, space=space, mu=2.0, rel=STL.GREATER, var_name='x1')
    pred66_ = STL(STL.PRED, space=space, mu=3.1, rel=STL.LESS, var_name='x1')
    pred5_ = STL(STL.PRED, space=space, mu=0.5, rel=STL.GREATER, var_name='x2')
    pred6_ = STL(STL.PRED, space=space, mu=1, rel=STL.LESS, var_name='x2')
    conj2_ = STL(STL.AND, space=space, subformulae=[pred5_, pred6_,pred55_, pred66_])
    always1_ = STL(STL.ALWAYS, space=space, low=5, high=6, subformula=conj2_)
    

    # Obstacle
    pred7 = STL(STL.PRED, space=space, mu=2.0, rel=STL.LESS, var_name='x1')
    pred8 = STL(STL.PRED, space=space, mu=3.0, rel=STL.GREATER, var_name='x1')
    pred9 = STL(STL.PRED, space=space, mu=-0.5, rel=STL.LESS, var_name='x2')
    pred10 = STL(STL.PRED, space=space, mu=0.5, rel=STL.GREATER, var_name='x2')
    disj1 = STL(STL.OR, space=space, subformulae=[pred7, pred8, pred9, pred10])
    always2 = STL(STL.ALWAYS, space=space, low=0, high=15, subformula=disj1)
    
    spec = STL(STL.AND, space=space, subformulae=[event1,always1_, always1,always2])

    
    
    # # Old Specs: 
    # space = SignalSpace(var_names=['x1', 'x2'], bounds=[[0, 4], [-1, 1]])
    
    # pred1 = STL(STL.PRED, space=space, mu=3.5, rel=STL.GREATER, var_name='x1')
    # pred2 = STL(STL.PRED, space=space, mu=4, rel=STL.LESS, var_name='x1')
    # pred3 = STL(STL.PRED, space=space, mu=-0.2, rel=STL.GREATER, var_name='x2')
    # pred4 = STL(STL.PRED, space=space, mu=0.2, rel=STL.LESS, var_name='x2')
    # conj1 = STL(STL.AND, space=space, subformulae=[pred1, pred2, pred3, pred4])
    # event1 = STL(STL.EVENTUALLY, space=space, low=2, high=10, subformula=conj1)
    
    # pred5 = STL(STL.PRED, space=space, mu=-0.5, rel=STL.GREATER, var_name='x2')
    # pred6 = STL(STL.PRED, space=space, mu=0.5, rel=STL.LESS, var_name='x2')
    # conj2 = STL(STL.AND, space=space, subformulae=[pred5, pred6])
    # always1 = STL(STL.ALWAYS, space=space, low=0, high=2, subformula=conj2)
    
    # pred7 = STL(STL.PRED, space=space, mu=2, rel=STL.LESS, var_name='x1')
    # pred8 = STL(STL.PRED, space=space, mu=3, rel=STL.GREATER, var_name='x1')
    # pred9 = STL(STL.PRED, space=space, mu=-0.5, rel=STL.LESS, var_name='x2')
    # pred10 = STL(STL.PRED, space=space, mu=0.5, rel=STL.GREATER, var_name='x2')
    # disj1 = STL(STL.OR, space=space, subformulae=[pred7, pred8, pred9, pred10])
    # always2 = STL(STL.ALWAYS, space=space, low=0, high=10, subformula=disj1)
    
    # spec = STL(STL.AND, space=space, subformulae=[event1, always1, always2])
    #------------------------------------------------------------------------

    # space = SignalSpace(var_names=['x1', 'x2'], bounds=[[0, 4], [-1, 1]])
    
    # # Goal region: 
    # pred1 = STL(STL.PRED, space=space, mu=3.5, rel=STL.GREATER, var_name='x1')
    # pred2 = STL(STL.PRED, space=space, mu=4, rel=STL.LESS, var_name='x1')
    # pred3 = STL(STL.PRED, space=space, mu=-0.2, rel=STL.GREATER, var_name='x2')
    # pred4 = STL(STL.PRED, space=space, mu=0.2, rel=STL.LESS, var_name='x2')
    # conj1 = STL(STL.AND, space=space, subformulae=[pred1, pred2, pred3, pred4])
    # event1 = STL(STL.EVENTUALLY, space=space, low=2, high=10, subformula=conj1)

    # # 1st region to visit
    # pred5 = STL(STL.PRED, space=space, mu=0, rel=STL.GREATER, var_name='x1')
    # pred6 = STL(STL.PRED, space=space, mu=1, rel=STL.LESS, var_name='x1')
    # pred7 = STL(STL.PRED, space=space, mu=0, rel=STL.GREATER, var_name='x2')
    # pred8 = STL(STL.PRED, space=space, mu=0.5, rel=STL.LESS, var_name='x2')
    # conj2 = STL(STL.AND, space=space, subformulae=[pred5, pred6, pred7, pred8])
    # always1 = STL(STL.ALWAYS, space=space, low=0, high=2, subformula=conj2)

    # # 2nd region to visit
    # pred9 = STL(STL.PRED, space=space, mu=1.5, rel=STL.GREATER, var_name='x1')
    # pred10 = STL(STL.PRED, space=space, mu=3.25, rel=STL.LESS, var_name='x1')
    # pred11 = STL(STL.PRED, space=space, mu=0.5, rel=STL.GREATER, var_name='x2')
    # pred12 = STL(STL.PRED, space=space, mu=1, rel=STL.LESS, var_name='x2')
    # conj3 = STL(STL.AND, space=space, subformulae=[pred9, pred10, pred11, pred12])
    # always2 = STL(STL.ALWAYS, space=space, low=3, high=7, subformula=conj3)

    # # Avoid obstacles region for all time:
    # pred13 = STL(STL.PRED, space=space, mu=3, rel=STL.GREATER, var_name='x1')
    # pred14 = STL(STL.PRED, space=space, mu=1.5, rel=STL.LESS, var_name='x1')
    # pred15 = STL(STL.PRED, space=space, mu=0.4, rel=STL.GREATER, var_name='x2')
    # pred16 = STL(STL.PRED, space=space, mu=-0.25, rel=STL.LESS, var_name='x2')
    # disj1 = STL(STL.OR, space=space, subformulae=[pred13, pred14, pred15, pred16])
    # always3 = STL(STL.ALWAYS, space=space, low=0, high=10, subformula=disj1)

    # pred16 = STL(STL.PRED, space=space, mu=-0.5, rel=STL.GREATER, var_name='x2')
    # pred17 = STL(STL.PRED, space=space, mu=0.5, rel=STL.LESS, var_name='x2')
    # conj5 = STL(STL.AND, space=space, subformulae=[pred16, pred17])
    # always4 = STL(STL.ALWAYS, space=space, low=0, high=2, subformula=conj5)  

    # spec = STL(STL.AND, space=space, subformulae=[event1, always1, always2,always3])
    
    return spec

def rear_wheel_car_spec():
    space = SignalSpace(var_names=['x1', 'x2', 'theta', 'v', 'omega'],
                        bounds=[[0, 4], [0, 4], [-3.14, 3.14],
                                [-0.3, 0.3], [-1, 1]])
    
    pred1 = STL(STL.PRED, space=space, mu=3, rel=STL.GREATER, var_name='x1')
    pred2 = STL(STL.PRED, space=space, mu=4, rel=STL.LESS, var_name='x1')
    pred3 = STL(STL.PRED, space=space, mu=2, rel=STL.GREATER, var_name='x2')
    pred4 = STL(STL.PRED, space=space, mu=3, rel=STL.LESS, var_name='x2')
    conj1 = STL(STL.AND, space=space, subformulae=[pred1, pred2, pred3, pred4])
    event1 = STL(STL.EVENTUALLY, space=space, low=0, high=18, subformula=conj1)
    
    pred5 = STL(STL.PRED, space=space, mu=3, rel=STL.GREATER, var_name='x2')
    pred6 = STL(STL.PRED, space=space, mu=2, rel=STL.LESS, var_name='x2')
    pred7 = STL(STL.PRED, space=space, mu=2, rel=STL.GREATER, var_name='x1')
    pred8 = STL(STL.PRED, space=space, mu=1, rel=STL.LESS, var_name='x1')
    disj2 = STL(STL.OR, space=space, subformulae=[pred5, pred6, pred7, pred8])
    always2 = STL(STL.ALWAYS, space=space, low=0, high=6, subformula=disj2)
    
#     pred9  = STL(STL.PRED, space=space, mu=0, rel=STL.GREATER, var_name='x1')
#     pred10 = STL(STL.PRED, space=space, mu=1, rel=STL.LESS, var_name='x1')
#     pred11 = STL(STL.PRED, space=space, mu=0, rel=STL.GREATER, var_name='x2')
#     pred12 = STL(STL.PRED, space=space, mu=1, rel=STL.LESS, var_name='x2')
#     conj3 = STL(STL.AND, space=space, subformulae=[pred9, pred10, pred11,
#                                                    pred12])
#     event3 = STL(STL.EVENTUALLY, space=space, low=30, high=40, subformula=conj3)
#     
#     pred13 = STL(STL.PRED, space=space, mu=2, rel=STL.GREATER, var_name='x1')
#     pred14 = STL(STL.PRED, space=space, mu=1, rel=STL.LESS, var_name='x1')
#     pred15 = STL(STL.PRED, space=space, mu=1, rel=STL.GREATER, var_name='x2')
#     pred16 = STL(STL.PRED, space=space, mu=0, rel=STL.LESS, var_name='x2')
#     disj4 = STL(STL.OR, space=space, subformulae=[pred13, pred14, pred15,
#                                                   pred16])
#     always4 = STL(STL.ALWAYS, space=space, low=18, high=24, subformula=disj4)
    
#     spec = event1 #STL(STL.AND, space=space, subformulae=[event1, always1])
    spec = STL(STL.AND, space=space, subformulae=[event1, always2])
#                                                  event3, always4])
    
    return spec

if __name__ == '__main__':
    # if len(sys.argv)>0:
    #     phi = {'di': double_integrator_spec,'rwc': rear_wheel_car_spec,}[sys.argv[0]]()
    # else:
    #     phi = double_integrator_spec()
    phi = double_integrator_spec()
    
    print(phi)
    print(phi.bound)
    
    for t in range(phi.bound+1):
        print('t:', t, 'active:', phi.active(t))
    
