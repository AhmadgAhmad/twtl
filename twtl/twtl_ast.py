license_text='''
    Module implements API for TWTL Abstract Syntax Trees (AST).
    Copyright (C) 2020  Cristian Ioan Vasile <cvasile@lehigh.edu>
    Explainable Robotics Lab (ERL), Autonomous and Intelligent Robotics Lab
    Lehigh University

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
'''
.. module:: twtl.py
   :synopsis: Module implements API for TWTL Abstract Syntax Trees (AST).

.. moduleauthor:: Cristian Ioan Vasile <cvasile@lehigh.edu>

'''

from concurrent.futures import thread
from antlr4 import InputStream, CommonTokenStream
from torch import threshold

from twtlLexer import twtlLexer
from twtlParser import twtlParser
from twtlVisitor import twtlVisitor
from twtlListener import twtlListener


class Operation(object):
    '''TWTL operations'''
    NOP, NOT, OR, AND, HOLD, CONCAT, WITHIN, LT, LE, GT, GE, EQ, NQ = range(13)
    opnames = [None, '!', '||', '&&', 'H', '.', 'W', '<', '<=', '>', '>=', '=', '!=']
    opcodes = {'!': NOT, '&&': AND, '||' : OR, 'H': HOLD, '.': CONCAT,
               'W': WITHIN, '<': LT, '<=': LE, '>' : GT, '>=': GE, '=': EQ, '!=': NQ}

    @classmethod
    def getCode(cls, text):
        ''' Gets the code corresponding to the string representation.'''
        return cls.opcodes.get(text, cls.NOP)

    @classmethod
    def getString(cls, op):
        '''Gets custom string representation for each operation.'''
        return cls.opnames[op]


class TWTLFormula(object):
    '''Abstract Syntax Tree representation of an TWTL formula'''

    def __init__(self, operation, **kwargs):
        '''Constructor'''
        self.op = operation
        self.props = []
        if self.op in (Operation.AND, Operation.OR, Operation.CONCAT):
            self.left = kwargs['left']
            self.right = kwargs['right']
        elif self.op == Operation.NOT:
            self.child = kwargs['child']
        elif self.op == Operation.HOLD:
            self.duration = kwargs['duration']
            self.proposition = kwargs['proposition']
            self.negated = kwargs['negated']
            self.predicate = kwargs['predicate']
            self.relation = kwargs['relation']
            self.variable = kwargs['variable']
            self.threshold = kwargs['threshold']
        elif self.op == Operation.WITHIN:
            self.low = kwargs['low']
            self.high = kwargs['high']
            self.child = kwargs['child']

        self.__string = None
        self.__hash = None

    def bounds(self):
        '''Computes the bounds of the TWTL formula.'''
        if self.op == Operation.AND:
            left_bounds = self.left.bounds()
            right_bounds = self.right.bounds()
            low_bound = max(left_bounds[0], right_bounds[0])
            upper_bound = max(left_bounds[1], right_bounds[1])
            return [low_bound, upper_bound]
        elif self.op == Operation.OR:
            left_bounds = self.left.bounds()
            right_bounds = self.right.bounds()
            low_bound = min(left_bounds[0], right_bounds[0])
            upper_bound = max(left_bounds[1], right_bounds[1])
            return [low_bound, upper_bound]
        elif self.op == Operation.NOT:
            return self.child.bounds()
        elif self.op == Operation.HOLD:
            return [self.duration, self.duration]
        elif self.op == Operation.CONCAT:
            left_bounds = self.left.bounds()
            right_bounds = self.right.bounds()
            low_bound = 1 + left_bounds[0] + right_bounds[0]
            upper_bound = 1 + left_bounds[1] + right_bounds[1]
            return [low_bound, upper_bound]
        elif self.op == Operation.WITHIN:
            child_bounds = self.child.bounds()
            assert child_bounds[0] <= self.high - self.low, \
                'Child formula is unfeasible within time window'
            return [self.low + child_bounds[0], self.high]

    def propositions(self):
        '''Computes the set of predicates/booleanExpr that are involved in the TWTL formula'''
        if self.op == Operation.HOLD:
            if self.proposition is None and self.predicate is not None:
                # TODO [Fix] when 
                predFormctx = self
                return predFormctx
            else: 
                props = self.props
                if self.proposition not in self.props:
                    props.append(self.proposition)
                    self.props = props
                    return
                else:
                    return        
        elif self.op in (Operation.AND, Operation.OR, Operation.CONCAT):
            left_p = self.left.propositions()
            right_p = self.right.propositions()
            return left_p, right_p
        elif self.op in (Operation.NOT, Operation.WITHIN):
            child_p = self.child.propositions()
            return child_p
        
        
        
        '''Computes the set of propositions involved in the TWTL formula.'''
        # if self.op == Operation.HOLD:
        #     return {self.proposition}
        # elif self.op in (Operation.AND, Operation.OR, Operation.CONCAT):
        #     return self.left.variables() | self.right.variables()
        # elif self.op in (Operation.NOT, Operation.WITHIN):
        #     return self.child.variables()

    def identifier(self):
        h = hash(self)
        if h < 0:
            h = hex(ord('-'))[2:] + hex(-h)[1:]
        else:
            h = hex(ord('+'))[2:] + hex(h)[1:]
        return h

    def __hash__(self):
        if self.__hash is None:
            self.__hash = hash(str(self))
        return self.__hash

    def __eq__(self, other):
        return str(self) == str(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        if self.__string is not None:
            return self.__string

        opname = Operation.getString(self.op)
        if self.op in (Operation.AND, Operation.OR, Operation.CONCAT):
            s = '{left} {op} {right}'.format(left=self.left, op=opname,
                                             right=self.right)
        elif self.op == Operation.NOT:
            s = '{op} {child}'.format(op=opname, child=self.child)
        elif self.op == Operation.HOLD:
            if self.negated:
                prop = '!{prop}'.format(prop=self.proposition)
            else:
                prop = '{prop}'.format(prop=self.proposition)
            s = '{op}^{duration} {prop}'.format(op=opname, prop=prop,
                                                duration=self.duration)
        elif self.op == Operation.WITHIN:
            s = '[{child}]^[{low}, {high}]'.format(child=self.child,
                                                   low=self.low, high=self.high)
        self.__string = s
        return self.__string


class TWTLAbstractSyntaxTreeExtractor(twtlVisitor):
    '''Parse Tree visitor that constructs the AST of an TWTL formula'''

    def visitFormula(self, ctx):
        if ctx.op is None:
            op = Operation.HOLD
        elif ctx.op.text == '(':
            op = Operation.NOP
        elif ctx.op.text == '[':
            op = Operation.WITHIN
        else:
            op = Operation.getCode(ctx.op.text)
        ret = None
        low = -1
        high = -1
        if op in (Operation.AND, Operation.OR, Operation.CONCAT):
            ret = TWTLFormula(op, left=self.visit(ctx.left),
                              right=self.visit(ctx.right))
        elif op == Operation.NOT:
            ret = TWTLFormula(op, child=self.visit(ctx.child))
        elif op == Operation.HOLD: # The base case: 
            duration = 0 if ctx.duration is None else int(ctx.duration.text)
            
            # If the proposition is based off predicate: 
            pred_flg = len(ctx.children[3].children) == 3
            if pred_flg: 

                pred = ctx.children[3]
                pred_variable = pred.children[0].children[0].symbol.text 
                pred_relation = pred.children[1].symbol.text
                pred_threshold = float(pred.children[2].children[0].symbol.text)
                negated = ctx.negated is not None
                ret = TWTLFormula(op, duration=duration, predicate=pred,
                                relation = pred_relation,
                                variable = pred_variable,
                                threshold = pred_threshold,
                                proposition = None,
                                negated=negated)
            else: #Atomic proposition: 
                prop = ctx.children[3].children[0].symbol.text
                if prop in ('true', 'false'):
                    prop = ctx.prop.text.lower() == 'true'
                negated = ctx.negated is not None
                ret = TWTLFormula(op, duration=duration, predicate=None,
                                relation = None,
                                variable = None,
                                threshold = None,
                                proposition = prop,
                                negated=negated)
        elif op == Operation.WITHIN:
            print((ctx.op.text, op))
            low = int(ctx.low.text)
            high = int(ctx.high.text)
            ret = TWTLFormula(Operation.WITHIN, child=self.visit(ctx.child),
                              low=low, high=high)
        elif op == Operation.NOP:
            ret = self.visit(ctx.child)
        else:
            print('Error: unknown operation!')
        return ret


if __name__ == '__main__':
    lexer = twtlLexer(InputStream("[A && !B]^[0, 3] && [H^2 C * H^6 E]^[1, 2]|| [H^2 !D]^[1, 7]"))
    tokens = CommonTokenStream(lexer)

    parser = twtlParser(tokens)
    t = parser.formula()
    print(t.toStringTree())

    ast = TWTLAbstractSyntaxTreeExtractor().visit(t)
    print('AST:', str(ast))
