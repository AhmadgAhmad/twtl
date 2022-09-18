# Generated from twtl.g4 by ANTLR 4.7.1
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .twtlParser import twtlParser
else:
    from twtlParser import twtlParser

license_text='''
    Copyright (C) 2015-2020  Cristian Ioan Vasile <cvasile@lehigh.edu>
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


# This class defines a complete generic visitor for a parse tree produced by twtlParser.

class twtlVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by twtlParser#formula.
    def visitFormula(self, ctx:twtlParser.FormulaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by twtlParser#expr.
    def visitExpr(self, ctx:twtlParser.ExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by twtlParser#booleanExpr.
    def visitBooleanExpr(self, ctx:twtlParser.BooleanExprContext):
        return self.visitChildren(ctx)



del twtlParser