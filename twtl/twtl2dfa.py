license_text='''
    Module converts a TWTL formula to an automaton.
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


# from twtl import twtl
# from twtl import twtl_ast
from twtl_ast import Operation as TWTLOperation
import twtl_ast
from dfa import complement, concatenation, hold, intersection, union, within


def twtl2dfa(formula_ast, props):
    '''TODO:
    '''
    dfa = None

    if formula_ast.op in (TWTLOperation.OR, TWTLOperation.AND,
                          TWTLOperation.CONCAT):
        dfa_left = twtl2dfa(formula_ast.left,props=props)
        dfa_right = twtl2dfa(formula_ast.right, props= props)
        if formula_ast.op == TWTLOperation.OR:
            dfa = union(dfa_left, dfa_right)
        elif formula_ast.op == TWTLOperation.AND:
            dfa = intersection(dfa_left, dfa_right)
        elif formula_ast.op == TWTLOperation.CONCAT:
            dfa = concatenation(dfa_left, dfa_right)
    elif formula_ast.op == TWTLOperation.NOT:
        dfa = complement(twtl2dfa(formula_ast.child))
    elif formula_ast.op == TWTLOperation.HOLD: # This is the rho_hold function
        # FIXME: does not work if ast_formula.proposition is a Boolean constant
        if formula_ast.proposition is None:
            dfa = hold(props=props,prop='A',duration=formula_ast.duration,negation=formula_ast.negated)
            # dfa = hold(props, formula_ast.predicate, formula_ast.duration,
            #         negation=formula_ast.negated)
        else:
            dfa = hold(props, formula_ast.proposition, formula_ast.duration,
                    negation=formula_ast.negated)

    elif formula_ast.op == TWTLOperation.WITHIN:
        low,high = int(formula_ast.low), int(formula_ast.high)
        childDFA = twtl2dfa(formula_ast=formula_ast.child,props=props)
        dfa = within(childDFA, low, high)

    return dfa
