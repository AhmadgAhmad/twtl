'''
Created on Aug 20, 2016

@author: cristi
'''

import functools as fn
import itertools as it
# from collections import namedtuple

import numpy as np

# STL = namedtuple('STL', ['op', 'bound', 'space'])

class SignalSpace(object):
    '''Class representing the bounded space where the signals live. '''
    __slots__ = 'var_names', 'bounds', 'rho_max'
    
    def __init__(self, var_names, bounds):
        self.var_names = tuple(var_names)
        self.bounds = tuple([tuple(b) for b in bounds])
        self.rho_max = max([b[1]- b[0] for b in self.bounds])
    
    def var_idx(self, var_name):
        return self.var_names.index(var_name)
    
    def __hash__(self):
        return hash((self.var_names, self.bounds))
    
    def __eq__(self, other):
        return self.var_names == other.var_names and self.bounds == other.bounds


class STL(object):
    '''Class for STL formulae.'''
    BOOLEAN, PRED, AND, OR, NOT, ALWAYS, EVENTUALLY, UNTIL = range(8)
    op_names = [None, None, '&', '|', '!', 'G', 'F', 'U']
    LESS, GREATER = range(2)
    rel_names = ['<=', '>']

    def __init__(self, op, space, **kwargs):
        '''Constructor'''
        self.op = op
        self.space = space
        
        if self.op == STL.BOOLEAN:
            self.value = kwargs['value']
            self.bound = 0
            self.__hash = hash((self.op, self.value))
        elif self.op == STL.PRED:
            self.mu = kwargs['mu']
            self.rel = kwargs['rel']
            self.var_idx = self.space.var_idx(kwargs['var_name'])
            self.bound = 0
            self.__hash = hash((self.op, self.var_idx, self.rel, self.mu))
        elif self.op == STL.NOT:
            self.subformula = kwargs['subformula']
            self.bound = self.subformula.bound
            self.__hash = hash((self.op, hash(self.subformula)))
            # FIXME: add method to translate formula to positive normal form
            assert False, 'We assume that the formula is in positive normal form!'
        elif self.op in (STL.AND, STL.OR):
            self.subformulae = tuple(kwargs['subformulae'])
            self.bound = max([f.bound for f in self.subformulae])
            self.__hash = hash((self.op, ) + tuple([str(f) for f in self.subformulae]))
        elif self.op in (STL.ALWAYS, STL.EVENTUALLY):
            self.low = kwargs['low']
            self.high = kwargs['high']
            self.subformula = kwargs['subformula']
            self.bound = self.high + self.subformula.bound
            self.__hash = hash((self.op, self.low, self.high, hash(self.subformula)))
        elif self.op == STL.UNTIL:
            self.low = kwargs['low']
            self.high = kwargs['high']
            self.hold_subformula = kwargs['hold_subformula']
            self.detect_subformula = kwargs['detect_subformula']
            self.bound = self.high + max(self.hold_subformula.bound,
                                         self.detect_subformula.bound)
            self.__hash = hash((self.op, self.low, self.high,
                                hash(self.hold_subformula),
                                hash(self.detect_subformula)))
        else:
            raise ValueError('Unknown operation, opcode: %d!', self.op)
        
        # FIXME: this might not be the best implementation
        self.parent = None
    
    def mutually_exclusive(self, other):
        '''Returns true if self and other are mutually exclusive predicates.'''
        assert self.op == STL.PRED and other.op == STL.PRED
        if self.var_idx != other.var_idx:
            return False
        if self.rel == other.rel:
            return False
        if self.rel == STL.LESS and other.rel == STL.GREATER:
            return self.mu <= other.mu
        elif self.rel == STL.GREATER and other.rel == STL.LESS:
            return self.mu >= other.mu
        assert False, 'Code should be unreachable!'
    
    def robustness(self,traj, shift = 0):
        ''' TODO [code_opt] This method could be incorporated with rosi (see below) method 
            TODO [code_opt] This function performs duplicated computation, we might as well use the computation of RoSI when 
        @input traj: A torch tensor with size([dim,n]), dim: dimension of the signal (typically 2) n: the number of discrete points of the trajectory 
        
        
        

        '''
        
        assert len(traj[0][0]) == len(traj[1]), traj
        if self.op == STL.BOOLEAN:
            assert self.value
            rho = self.space.rho_max
        elif self.op == STL.PRED:
            s, t = traj
            if t[0] <= shift <= t[-1]:
                # s_i = [x[self.var_idx] for x in s]
                s_i = s[self.var_idx,:]
                x = np.interp(shift, t, s_i)
                if self.rel == STL.LESS:
                    rho = self.mu - x
                else: # self.rel == STL.GREATER
                    rho = x - self.mu
            else:
                rho = -self.space.rho_max 
        elif self.op in (STL.OR, STL.AND):
            rho = [f.robustness(traj, shift) for f in self.subformulae]
            if self.op == STL.OR:
                rho = max(rho)
            else: # self.op == STL.AND
                rho = min(rho)
        elif self.op == STL.NOT:
            rho = self.subformula.robustness(traj)
            rho = -rho
        elif self.op in (STL.EVENTUALLY, STL.ALWAYS):
            times = [t for t in traj[1] if self.low <= t <= self.high]
            rho = [self.subformula.robustness(traj, t) for t in times]
            if self.op == STL.EVENTUALLY:
                rho = max(rho)
            else: # self.op == STL.ALWAYS
                rho = min(rho)
            a = 1
        elif self.op == STL.UNTIL:
            raise NotImplementedError
        else:
            raise ValueError('Unknown operation, opcode: %d!', self.op)
            
        
        return rho
    # def robustness(self,traj, shift = 0):
    #     ''' TODO [code_opt] This method could be incorporated with rosi (see below) method 
    #         TODO [code_opt] This function performs duplicated computation, we might as well use the computation of RoSI when 
    #     '''
        
    #     assert len(traj[0]) == len(traj[1]), traj
    #     if self.op == STL.BOOLEAN:
    #         assert self.value
    #         rho = self.space.rho_max
    #     elif self.op == STL.PRED:
    #         s, t = traj
    #         if t[0] <= shift <= t[-1]:
    #             s_i = [x[self.var_idx] for x in s]
    #             x = np.interp(shift, t, s_i)
    #             if self.rel == STL.LESS:
    #                 rho = self.mu - x
    #             else: # self.rel == STL.GREATER
    #                 rho = x - self.mu
    #         else:
    #             rho = -self.space.rho_max 
    #     elif self.op in (STL.OR, STL.AND):
    #         rho = [f.robustness(traj, shift) for f in self.subformulae]
    #         if self.op == STL.OR:
    #             rho = max(rho)
    #         else: # self.op == STL.AND
    #             rho = min(rho)
    #     elif self.op == STL.NOT:
    #         rho = self.subformula.robustness(traj)
    #         rho = -rho
    #     elif self.op in (STL.EVENTUALLY, STL.ALWAYS):
    #         times = [t for t in traj[1] if self.low <= t <= self.high]
    #         rho = [self.subformula.robustness(traj, t) for t in times]
    #         if self.op == STL.EVENTUALLY:
    #             rho = max(rho)
    #         else: # self.op == STL.ALWAYS
    #             rho = min(rho)
    #         a = 1
    #     elif self.op == STL.UNTIL:
    #         raise NotImplementedError
    #     else:
    #         raise ValueError('Unknown operation, opcode: %d!', self.op)
            
        
    #     return rho

    def rosi(self, traj, shift=0.0):
        '''FIXME: call online monitor
        assumes traj is sorted w.r.t. times
        '''
        assert len(traj[0]) == len(traj[1]), traj
        assert shift >= 0.0
        
        if self.op == STL.BOOLEAN:
            assert self.value
            rho_low, rho_high = self.space.rho_max, self.space.rho_max
        elif self.op == STL.PRED:
            s, t = traj
            if t[0] <= shift <= t[-1]:
                s_i = [x[self.var_idx] for x in s]
                x = np.interp(shift, t, s_i)
                if self.rel == STL.LESS:
                    rho = self.mu - x
                else: # self.rel == STL.GREATER
                    rho = x - self.mu
                rho_low, rho_high = rho, rho
            else:
                rho_low, rho_high = -self.space.rho_max, self.space.rho_max
        elif self.op in (STL.OR, STL.AND):
            rho_low, rho_high = zip(*[f.rosi(traj, shift)
                                                     for f in self.subformulae])
            if self.op == STL.OR:
                rho_low, rho_high = max(rho_low), max(rho_high)
            else: # self.op == STL.AND
                rho_low, rho_high = min(rho_low), min(rho_high)
        elif self.op == STL.NOT:
            rho_low, rho_high = self.subformula.rosi(traj)
            rho_low, rho_high = -rho_high, -rho_low
        elif self.op in (STL.EVENTUALLY, STL.ALWAYS):
            times = set([t for t in traj[1] if self.low <= t <= self.high]
                        + [self.low, self.high]) 
            rho_low, rho_high = zip(*[self.subformula.rosi(traj, t)
                                                                for t in times])
            if self.op == STL.EVENTUALLY:
                rho_low, rho_high = max(rho_low), max(rho_high)
            else: # self.op == STL.ALWAYS
                rho_low, rho_high = min(rho_low), min(rho_high)
        elif self.op == STL.UNTIL:
            raise NotImplementedError
        else:
            raise ValueError('Unknown operation, opcode: %d!', self.op)
        
        assert rho_low <= rho_high
        return rho_low, rho_high
    
    def rosi_twtl(self, traj, shift=0.0):
        '''FIXME: call online monitor
        assumes traj is sorted w.r.t. times
        '''
        assert len(traj[0]) == len(traj[1]), traj
        assert shift >= 0.0
        
        if self.op == STL.BOOLEAN:
            assert self.value
            rho_low, rho_high = self.space.rho_max, self.space.rho_max
        elif self.op == STL.PRED:
            s, t = traj
            if t[0] <= shift <= t[-1]:
                s_i = [x[self.var_idx] for x in s]
                x = np.interp(shift, t, s_i)
                if self.rel == STL.LESS:
                    rho = self.mu - x
                else: # self.rel == STL.GREATER
                    rho = x - self.mu
                rho_low, rho_high = rho, rho
            else:
                rho_low, rho_high = -self.space.rho_max, self.space.rho_max
        elif self.op in (STL.OR, STL.AND):
            rho_low, rho_high = zip(*[f.rosi(traj, shift)
                                                    for f in self.subformulae])
            if self.op == STL.OR:
                rho_low, rho_high = max(rho_low), max(rho_high)
            else: # self.op == STL.AND
                rho_low, rho_high = min(rho_low), min(rho_high)
        elif self.op == STL.NOT:
            rho_low, rho_high = self.subformula.rosi(traj)
            rho_low, rho_high = -rho_high, -rho_low
        elif self.op in (STL.EVENTUALLY, STL.ALWAYS):
            times = set([t for t in traj[1] if self.low <= t <= self.high]
                        + [self.low, self.high]) 
            rho_low, rho_high = zip(*[self.subformula.rosi(traj, t)
                                                                for t in times])
            if self.op == STL.EVENTUALLY:
                rho_low, rho_high = max(rho_low), max(rho_high)
            else: # self.op == STL.ALWAYS
                rho_low, rho_high = min(rho_low), min(rho_high)
        elif self.op == STL.UNTIL:
            raise NotImplementedError
        else:
            raise ValueError('Unknown operation, opcode: %d!', self.op)
        
        assert rho_low <= rho_high
        return rho_low, rho_high
    
    
    def simplify(self, ch_traj, t):
        '''FIXME: maybe there is a better implementation.
        '''
        f, upd = self._simplify(ch_traj, t)
        if upd:
            f.parent = self
        else:
            assert f == self
        return f
    
    def _simplify(self, ch_traj, t):
        '''FIXME: does not use ch_traj, can simplify further
        '''
        if self.op == STL.BOOLEAN:
            return self, False
        elif self.op == STL.PRED:
            return self, False
        elif self.op in (STL.OR, STL.AND):
            subfs, upds = zip(*[f._simplify(ch_traj, t) for f in self.subformulae])
            if any(upds):
                assert not [f for f in subfs if f.op == STL.BOOLEAN and not f.value]
                subfs = [f for f in subfs if f.op != STL.BOOLEAN]
                if len(subfs) == 0:
                    return STL(STL.BOOLEAN, self.space, value=True), True
                elif len(subfs) == 1:
                    return subfs[0], True
                else:
                    return STL(self.op, self.space, subformulae=subfs), True
            else:
                if self.op == STL.AND:
                    return self, False
                else: # self.op == STL.OR
                    if np.random.uniform() < 0.5:
                        idx = np.random.randint(0, len(subfs))
                        subfs = list(subfs)
                        del subfs[idx]
                        return STL(self.op, self.space, subformulae=subfs), True
                    else:
                        return self, False
        elif self.op == STL.NOT:
            raise ValueError('The specification is not in PNF!')
        elif self.op in (STL.EVENTUALLY, STL.ALWAYS):
            if t <= self.high:
                subf, upd = self.subformula._simplify(ch_traj, t)
                if upd:
                    if subf.op == STL.BOOLEAN:
                        assert subf.value
                        return subf, True
                    else:
                        return STL(self.op, self.space, low=self.low,
                                   high=self.high, subformula=self.subformula), True
                else:
                    return self, False
            else:
                return STL(STL.BOOLEAN, self.space, value=True), True
        elif  self.op == STL.UNTIL:
            raise NotImplementedError
        else:
            raise ValueError('Unknown operation, opcode: %d!', self.op)
    
    def compatible(self, other):
        '''Check compatibility with other by looking for self in the ancestor
        line to the original specification. Ancestor lines are generated by
        successive simplifications.
        '''
        while other.parent is not None:
            if self == other.parent:
                return True
        return False
    
    def active(self, t):
        '''Returns the set of active predicates at time t.'''
        if t < 0:
            return set([])
        
        if self.op == STL.BOOLEAN:
            return set([])
        elif self.op == STL.PRED:
            if t == 0:
                return set([self])
            else:
                return set([])
        elif self.op in (STL.AND, STL.OR):
            return fn.reduce(set.union, [f.active(t) for f in self.subformulae])
        elif self.op in (STL.ALWAYS, STL.EVENTUALLY):
            if self.low <= t <= self.bound:
                return fn.reduce(set.union,
                                 [self.subformula.active(t - tp)
                                        for tp in range(self.low, self.high+1)])
            else:
                return set([])
        elif self.op == STL.UNTIL:
            if t <= self.bound:
                Pset_hold = fn.reduce(set.union,
                                      [self.hold_subformula.active(t - tp)
                                               for tp in range(0, self.high+1)])
                Pset_detect = fn.reduce(set.union,
                                        [self.detect_subformula.active(t - tp)
                                        for tp in range(self.low, self.high+1)])
                return Pset_hold | Pset_detect
            else:
                return set([])
        raise ValueError('Unknown operation, opcode: %d!', self.op)
    
    def __str__(self):
        opname = STL.op_names[self.op]
        if self.op == STL.BOOLEAN:
            return str(self.value)
        elif self.op == STL.PRED:
            return '({var} {rel} {mu})'.format(
                        var=self.space.var_names[self.var_idx],
                        rel=STL.rel_names[self.rel],
                        mu=self.mu)
        elif self.op in (STL.AND, STL.OR):
            opstr = ' {} '.format(opname)
            return opstr.join([str(f) for f in self.subformulae])
        elif self.op == STL.NOT:
            return '{op} ({subformula})'.format(opname, str(self.subformula))
        elif self.op in (STL.ALWAYS, STL.EVENTUALLY):
            return '{op}_[{low}, {high}] ({subformula})'.format(
                        op=opname,
                        low=self.low,
                        high=self.high,
                        subformula=str(self.subformula))
        elif self.op == STL.UNTIL:
            return '({hold_formula}) {op}_[{low}, {high}] ({detect_formula})'.format(
                        op=opname,
                        low=self.low,
                        high=self.high,
                        hold_formula=self.hold_subformula,
                        detect_subformula=self.detect_subformula)
        raise ValueError('Unknown operation, opcode: %d!', self.op)
    
    __repr__ = __str__
    
    def __hash__(self):
        return self.__hash
    
    def __eq__(self, other):
        if self.op != other.op:
            return False
        
        if self.op == STL.BOOLEAN:
            return self.value == other.value
        elif self.op == STL.PRED:
            return (self.var_idx == other.var_idx and self.rel == other.rel
                    and self.mu == other.mu)
        elif self.op in (STL.AND, STL.OR):
            if len(self.subformulae) != len(other.subformulae):
                return False
            #FIXME: very restrictive - dependent on the order of the subformulae
            # TODO [PY2]>>>
            # return all([f1 == f2 for f1, f2 in it.izip(self.subformulae, other.subformulae)])
            # TODO [PY2]<<<
            # TODO [PY3]>>>
            return all([f1 == f2 for f1, f2 in it.zip_longest(self.subformulae, other.subformulae)])
            # TODO [PY3]<<<


        elif self.op == STL.NOT:
            return self.subformula == other.subformula
        elif self.op in (STL.ALWAYS, STL.EVENTUALLY):
            return (self.low == other.low and self.high == other.high
                    and self.subformula == other.subformula)
        elif self.op == STL.UNTIL:
            return (self.low == other.low and self.high == other.high
                    and self.hold_subformula == other.hold_subformula
                    and self.detect_subformula == other.detect_subformula)
        raise ValueError('Unknown operation, opcode: %d!', self.op)
        


if __name__ == '__main__':
    pass