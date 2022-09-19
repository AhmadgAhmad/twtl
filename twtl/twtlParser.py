# Generated from twtl.g4 by ANTLR 4.7.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys


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

def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\35")
        buf.write("X\4\2\t\2\4\3\t\3\4\4\t\4\3\2\3\2\3\2\3\2\3\2\3\2\3\2")
        buf.write("\3\2\5\2\21\n\2\3\2\5\2\24\n\2\3\2\3\2\3\2\3\2\3\2\3\2")
        buf.write("\3\2\3\2\3\2\3\2\3\2\3\2\3\2\5\2#\n\2\3\2\3\2\3\2\3\2")
        buf.write("\3\2\3\2\3\2\3\2\3\2\7\2.\n\2\f\2\16\2\61\13\2\3\3\3\3")
        buf.write("\3\3\3\3\3\3\3\3\3\3\5\3:\n\3\3\4\3\4\3\4\3\4\3\4\3\4")
        buf.write("\3\4\3\4\3\4\3\4\3\4\3\4\5\4H\n\4\3\4\3\4\3\4\3\4\3\4")
        buf.write("\3\4\3\4\3\4\3\4\7\4S\n\4\f\4\16\4V\13\4\3\4\2\4\2\6\5")
        buf.write("\2\4\6\2\6\3\2\t\r\4\2\3\3\16\16\3\2\17\20\3\2\21\22\2")
        buf.write("e\2\"\3\2\2\2\49\3\2\2\2\6G\3\2\2\2\b\t\b\2\1\2\t\n\7")
        buf.write("\3\2\2\n\13\5\2\2\2\13\f\7\4\2\2\f#\3\2\2\2\r\16\7\26")
        buf.write("\2\2\16\17\7\5\2\2\17\21\7\30\2\2\20\r\3\2\2\2\20\21\3")
        buf.write("\2\2\2\21\23\3\2\2\2\22\24\7\25\2\2\23\22\3\2\2\2\23\24")
        buf.write("\3\2\2\2\24\25\3\2\2\2\25#\5\4\3\2\26\27\7\25\2\2\27#")
        buf.write("\5\2\2\4\30\31\7\6\2\2\31\32\5\2\2\2\32\33\7\7\2\2\33")
        buf.write("\34\7\5\2\2\34\35\7\6\2\2\35\36\7\30\2\2\36\37\7\b\2\2")
        buf.write("\37 \7\30\2\2 !\7\7\2\2!#\3\2\2\2\"\b\3\2\2\2\"\20\3\2")
        buf.write("\2\2\"\26\3\2\2\2\"\30\3\2\2\2#/\3\2\2\2$%\f\b\2\2%&\7")
        buf.write("\24\2\2&.\5\2\2\t\'(\f\7\2\2()\7\23\2\2).\5\2\2\b*+\f")
        buf.write("\6\2\2+,\7\27\2\2,.\5\2\2\7-$\3\2\2\2-\'\3\2\2\2-*\3\2")
        buf.write("\2\2.\61\3\2\2\2/-\3\2\2\2/\60\3\2\2\2\60\3\3\2\2\2\61")
        buf.write("/\3\2\2\2\62:\7\32\2\2\63:\7\31\2\2\64:\7\33\2\2\65\66")
        buf.write("\5\6\4\2\66\67\t\2\2\2\678\5\6\4\28:\3\2\2\29\62\3\2\2")
        buf.write("\29\63\3\2\2\29\64\3\2\2\29\65\3\2\2\2:\5\3\2\2\2;<\b")
        buf.write("\4\1\2<=\t\3\2\2=>\5\6\4\2>?\7\4\2\2?H\3\2\2\2@A\7\33")
        buf.write("\2\2AB\7\3\2\2BC\5\6\4\2CD\7\4\2\2DH\3\2\2\2EH\7\30\2")
        buf.write("\2FH\7\33\2\2G;\3\2\2\2G@\3\2\2\2GE\3\2\2\2GF\3\2\2\2")
        buf.write("HT\3\2\2\2IJ\f\b\2\2JK\7\5\2\2KS\5\6\4\bLM\f\6\2\2MN\t")
        buf.write("\4\2\2NS\5\6\4\7OP\f\5\2\2PQ\t\5\2\2QS\5\6\4\6RI\3\2\2")
        buf.write("\2RL\3\2\2\2RO\3\2\2\2SV\3\2\2\2TR\3\2\2\2TU\3\2\2\2U")
        buf.write("\7\3\2\2\2VT\3\2\2\2\13\20\23\"-/9GRT")
        return buf.getvalue()


class twtlParser ( Parser ):

    grammarFileName = "twtl.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'('", "')'", "'^'", "'['", "']'", "','", 
                     "'<'", "'<='", "'='", "'>='", "'>'", "'-('", "'*'", 
                     "'/'", "'+'", "'-'", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "'H'", "'.'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "AND", "OR", "NOT", "HOLD", "CONCAT", 
                      "RATIONAL", "TRUE", "FALSE", "VARIABLE", "LINECMT", 
                      "WS" ]

    RULE_formula = 0
    RULE_booleanExpr = 1
    RULE_expr = 2

    ruleNames =  [ "formula", "booleanExpr", "expr" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    T__7=8
    T__8=9
    T__9=10
    T__10=11
    T__11=12
    T__12=13
    T__13=14
    T__14=15
    T__15=16
    AND=17
    OR=18
    NOT=19
    HOLD=20
    CONCAT=21
    RATIONAL=22
    TRUE=23
    FALSE=24
    VARIABLE=25
    LINECMT=26
    WS=27

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.7.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None



    class FormulaContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.left = None # FormulaContext
            self.op = None # Token
            self.child = None # FormulaContext
            self.duration = None # Token
            self.negated = None # Token
            self.prop = None # BooleanExprContext
            self.low = None # Token
            self.high = None # Token
            self.right = None # FormulaContext

        def formula(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(twtlParser.FormulaContext)
            else:
                return self.getTypedRuleContext(twtlParser.FormulaContext,i)


        def booleanExpr(self):
            return self.getTypedRuleContext(twtlParser.BooleanExprContext,0)


        def HOLD(self):
            return self.getToken(twtlParser.HOLD, 0)

        def RATIONAL(self, i:int=None):
            if i is None:
                return self.getTokens(twtlParser.RATIONAL)
            else:
                return self.getToken(twtlParser.RATIONAL, i)

        def NOT(self):
            return self.getToken(twtlParser.NOT, 0)

        def OR(self):
            return self.getToken(twtlParser.OR, 0)

        def AND(self):
            return self.getToken(twtlParser.AND, 0)

        def CONCAT(self):
            return self.getToken(twtlParser.CONCAT, 0)

        def getRuleIndex(self):
            return twtlParser.RULE_formula

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFormula" ):
                listener.enterFormula(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFormula" ):
                listener.exitFormula(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFormula" ):
                return visitor.visitFormula(self)
            else:
                return visitor.visitChildren(self)



    def formula(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = twtlParser.FormulaContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 0
        self.enterRecursionRule(localctx, 0, self.RULE_formula, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 32
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
            if la_ == 1:
                self.state = 7
                localctx.op = self.match(twtlParser.T__0)
                self.state = 8
                localctx.child = self.formula(0)
                self.state = 9
                self.match(twtlParser.T__1)
                pass

            elif la_ == 2:
                self.state = 14
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==twtlParser.HOLD:
                    self.state = 11
                    localctx.op = self.match(twtlParser.HOLD)
                    self.state = 12
                    self.match(twtlParser.T__2)
                    self.state = 13
                    localctx.duration = self.match(twtlParser.RATIONAL)


                self.state = 17
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==twtlParser.NOT:
                    self.state = 16
                    localctx.negated = self.match(twtlParser.NOT)


                self.state = 19
                localctx.prop = self.booleanExpr()
                pass

            elif la_ == 3:
                self.state = 20
                localctx.op = self.match(twtlParser.NOT)
                self.state = 21
                localctx.child = self.formula(2)
                pass

            elif la_ == 4:
                self.state = 22
                localctx.op = self.match(twtlParser.T__3)
                self.state = 23
                localctx.child = self.formula(0)
                self.state = 24
                self.match(twtlParser.T__4)
                self.state = 25
                self.match(twtlParser.T__2)
                self.state = 26
                self.match(twtlParser.T__3)
                self.state = 27
                localctx.low = self.match(twtlParser.RATIONAL)
                self.state = 28
                self.match(twtlParser.T__5)
                self.state = 29
                localctx.high = self.match(twtlParser.RATIONAL)
                self.state = 30
                self.match(twtlParser.T__4)
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 45
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,4,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 43
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
                    if la_ == 1:
                        localctx = twtlParser.FormulaContext(self, _parentctx, _parentState)
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_formula)
                        self.state = 34
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 6)")
                        self.state = 35
                        localctx.op = self.match(twtlParser.OR)
                        self.state = 36
                        localctx.right = self.formula(7)
                        pass

                    elif la_ == 2:
                        localctx = twtlParser.FormulaContext(self, _parentctx, _parentState)
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_formula)
                        self.state = 37
                        if not self.precpred(self._ctx, 5):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 5)")
                        self.state = 38
                        localctx.op = self.match(twtlParser.AND)
                        self.state = 39
                        localctx.right = self.formula(6)
                        pass

                    elif la_ == 3:
                        localctx = twtlParser.FormulaContext(self, _parentctx, _parentState)
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_formula)
                        self.state = 40
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 4)")
                        self.state = 41
                        localctx.op = self.match(twtlParser.CONCAT)
                        self.state = 42
                        localctx.right = self.formula(5)
                        pass

             
                self.state = 47
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,4,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx

    class BooleanExprContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.op = None # Token
            self.left = None # ExprContext
            self.right = None # ExprContext

        def FALSE(self):
            return self.getToken(twtlParser.FALSE, 0)

        def TRUE(self):
            return self.getToken(twtlParser.TRUE, 0)

        def VARIABLE(self):
            return self.getToken(twtlParser.VARIABLE, 0)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(twtlParser.ExprContext)
            else:
                return self.getTypedRuleContext(twtlParser.ExprContext,i)


        def getRuleIndex(self):
            return twtlParser.RULE_booleanExpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBooleanExpr" ):
                listener.enterBooleanExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBooleanExpr" ):
                listener.exitBooleanExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBooleanExpr" ):
                return visitor.visitBooleanExpr(self)
            else:
                return visitor.visitChildren(self)




    def booleanExpr(self):

        localctx = twtlParser.BooleanExprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_booleanExpr)
        self._la = 0 # Token type
        try:
            self.state = 55
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 48
                localctx.op = self.match(twtlParser.FALSE)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 49
                localctx.op = self.match(twtlParser.TRUE)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 50
                localctx.op = self.match(twtlParser.VARIABLE)
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 51
                localctx.left = self.expr(0)
                self.state = 52
                localctx.op = self._input.LT(1)
                _la = self._input.LA(1)
                if not((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << twtlParser.T__6) | (1 << twtlParser.T__7) | (1 << twtlParser.T__8) | (1 << twtlParser.T__9) | (1 << twtlParser.T__10))) != 0)):
                    localctx.op = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 53
                localctx.right = self.expr(0)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx

    class ExprContext(ParserRuleContext):

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(twtlParser.ExprContext)
            else:
                return self.getTypedRuleContext(twtlParser.ExprContext,i)


        def VARIABLE(self):
            return self.getToken(twtlParser.VARIABLE, 0)

        def RATIONAL(self):
            return self.getToken(twtlParser.RATIONAL, 0)

        def getRuleIndex(self):
            return twtlParser.RULE_expr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExpr" ):
                listener.enterExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExpr" ):
                listener.exitExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExpr" ):
                return visitor.visitExpr(self)
            else:
                return visitor.visitChildren(self)



    def expr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = twtlParser.ExprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 4
        self.enterRecursionRule(localctx, 4, self.RULE_expr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 69
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,6,self._ctx)
            if la_ == 1:
                self.state = 58
                _la = self._input.LA(1)
                if not(_la==twtlParser.T__0 or _la==twtlParser.T__11):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 59
                self.expr(0)
                self.state = 60
                self.match(twtlParser.T__1)
                pass

            elif la_ == 2:
                self.state = 62
                self.match(twtlParser.VARIABLE)
                self.state = 63
                self.match(twtlParser.T__0)
                self.state = 64
                self.expr(0)
                self.state = 65
                self.match(twtlParser.T__1)
                pass

            elif la_ == 3:
                self.state = 67
                self.match(twtlParser.RATIONAL)
                pass

            elif la_ == 4:
                self.state = 68
                self.match(twtlParser.VARIABLE)
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 82
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,8,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 80
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,7,self._ctx)
                    if la_ == 1:
                        localctx = twtlParser.ExprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 71
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 6)")
                        self.state = 72
                        self.match(twtlParser.T__2)
                        self.state = 73
                        self.expr(6)
                        pass

                    elif la_ == 2:
                        localctx = twtlParser.ExprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 74
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 4)")
                        self.state = 75
                        _la = self._input.LA(1)
                        if not(_la==twtlParser.T__12 or _la==twtlParser.T__13):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 76
                        self.expr(5)
                        pass

                    elif la_ == 3:
                        localctx = twtlParser.ExprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 77
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 3)")
                        self.state = 78
                        _la = self._input.LA(1)
                        if not(_la==twtlParser.T__14 or _la==twtlParser.T__15):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 79
                        self.expr(4)
                        pass

             
                self.state = 84
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,8,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[0] = self.formula_sempred
        self._predicates[2] = self.expr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def formula_sempred(self, localctx:FormulaContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 6)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 5)
         

            if predIndex == 2:
                return self.precpred(self._ctx, 4)
         

    def expr_sempred(self, localctx:ExprContext, predIndex:int):
            if predIndex == 3:
                return self.precpred(self._ctx, 6)
         

            if predIndex == 4:
                return self.precpred(self._ctx, 4)
         

            if predIndex == 5:
                return self.precpred(self._ctx, 3)
         




