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
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\36")
        buf.write("V\4\2\t\2\4\3\t\3\4\4\t\4\3\2\3\2\3\2\3\2\3\2\3\2\3\2")
        buf.write("\3\2\3\2\3\2\5\2\23\n\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3")
        buf.write("\2\3\2\3\2\3\2\3\2\3\2\5\2\"\n\2\3\2\3\2\3\2\3\2\3\2\3")
        buf.write("\2\3\2\3\2\3\2\7\2-\n\2\f\2\16\2\60\13\2\3\3\3\3\3\3\3")
        buf.write("\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\5\3>\n\3\3\3\3\3\3")
        buf.write("\3\3\3\3\3\3\3\3\3\3\3\3\3\7\3I\n\3\f\3\16\3L\13\3\3\4")
        buf.write("\3\4\3\4\3\4\3\4\3\4\5\4T\n\4\3\4\2\4\2\4\5\2\4\6\2\6")
        buf.write("\4\2\3\3\t\t\3\2\n\13\3\2\f\r\3\2\16\22\2a\2!\3\2\2\2")
        buf.write("\4=\3\2\2\2\6S\3\2\2\2\b\t\b\2\1\2\t\n\7\3\2\2\n\13\5")
        buf.write("\2\2\2\13\f\7\4\2\2\f\"\3\2\2\2\r\16\7\31\2\2\16\17\7")
        buf.write("\5\2\2\17\20\7\34\2\2\20\22\3\2\2\2\21\23\7\30\2\2\22")
        buf.write("\21\3\2\2\2\22\23\3\2\2\2\23\24\3\2\2\2\24\"\5\6\4\2\25")
        buf.write("\26\7\30\2\2\26\"\5\2\2\5\27\30\7\6\2\2\30\31\5\2\2\2")
        buf.write("\31\32\7\7\2\2\32\33\7\5\2\2\33\34\7\6\2\2\34\35\7\34")
        buf.write("\2\2\35\36\7\b\2\2\36\37\7\34\2\2\37 \7\7\2\2 \"\3\2\2")
        buf.write("\2!\b\3\2\2\2!\r\3\2\2\2!\25\3\2\2\2!\27\3\2\2\2\".\3")
        buf.write("\2\2\2#$\f\7\2\2$%\7\26\2\2%-\5\2\2\b&\'\f\6\2\2\'(\7")
        buf.write("\27\2\2(-\5\2\2\7)*\f\4\2\2*+\7\32\2\2+-\5\2\2\5,#\3\2")
        buf.write("\2\2,&\3\2\2\2,)\3\2\2\2-\60\3\2\2\2.,\3\2\2\2./\3\2\2")
        buf.write("\2/\3\3\2\2\2\60.\3\2\2\2\61\62\b\3\1\2\62\63\t\2\2\2")
        buf.write("\63\64\5\4\3\2\64\65\7\4\2\2\65>\3\2\2\2\66\67\7\24\2")
        buf.write("\2\678\7\3\2\289\5\4\3\29:\7\4\2\2:>\3\2\2\2;>\7\25\2")
        buf.write("\2<>\7\24\2\2=\61\3\2\2\2=\66\3\2\2\2=;\3\2\2\2=<\3\2")
        buf.write("\2\2>J\3\2\2\2?@\f\b\2\2@A\7\5\2\2AI\5\4\3\bBC\f\6\2\2")
        buf.write("CD\t\3\2\2DI\5\4\3\7EF\f\5\2\2FG\t\4\2\2GI\5\4\3\6H?\3")
        buf.write("\2\2\2HB\3\2\2\2HE\3\2\2\2IL\3\2\2\2JH\3\2\2\2JK\3\2\2")
        buf.write("\2K\5\3\2\2\2LJ\3\2\2\2MN\5\4\3\2NO\t\5\2\2OP\5\4\3\2")
        buf.write("PT\3\2\2\2QT\7\23\2\2RT\7\33\2\2SM\3\2\2\2SQ\3\2\2\2S")
        buf.write("R\3\2\2\2T\7\3\2\2\2\n\22!,.=HJS")
        return buf.getvalue()


class twtlParser ( Parser ):

    grammarFileName = "twtl.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'('", "')'", "'^'", "'['", "']'", "','", 
                     "'-('", "'prd'", "'/'", "'+'", "'-'", "'<'", "'<='", 
                     "'='", "'>='", "'>'", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "<INVALID>", "<INVALID>", "'H'", "'*'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "BOOLEAN", "VARIABLE", "RATIONAL", "AND", 
                      "OR", "NOT", "HOLD", "CONCAT", "PROP", "INT", "LINECMT", 
                      "WS" ]

    RULE_formula = 0
    RULE_expr = 1
    RULE_booleanExpr = 2

    ruleNames =  [ "formula", "expr", "booleanExpr" ]

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
    BOOLEAN=17
    VARIABLE=18
    RATIONAL=19
    AND=20
    OR=21
    NOT=22
    HOLD=23
    CONCAT=24
    PROP=25
    INT=26
    LINECMT=27
    WS=28

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
            self.child = None # FormulaContext
            self.op = None # Token
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

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(twtlParser.INT)
            else:
                return self.getToken(twtlParser.INT, i)

        def NOT(self):
            return self.getToken(twtlParser.NOT, 0)

        def AND(self):
            return self.getToken(twtlParser.AND, 0)

        def OR(self):
            return self.getToken(twtlParser.OR, 0)

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
            self.state = 31
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [twtlParser.T__0]:
                self.state = 7
                self.match(twtlParser.T__0)
                self.state = 8
                localctx.child = self.formula(0)
                self.state = 9
                self.match(twtlParser.T__1)
                pass
            elif token in [twtlParser.HOLD]:
                self.state = 11
                localctx.op = self.match(twtlParser.HOLD)
                self.state = 12
                self.match(twtlParser.T__2)
                self.state = 13
                localctx.duration = self.match(twtlParser.INT)
                self.state = 16
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==twtlParser.NOT:
                    self.state = 15
                    localctx.negated = self.match(twtlParser.NOT)


                self.state = 18
                localctx.prop = self.booleanExpr()
                pass
            elif token in [twtlParser.NOT]:
                self.state = 19
                localctx.op = self.match(twtlParser.NOT)
                self.state = 20
                localctx.child = self.formula(3)
                pass
            elif token in [twtlParser.T__3]:
                self.state = 21
                localctx.op = self.match(twtlParser.T__3)
                self.state = 22
                localctx.child = self.formula(0)
                self.state = 23
                self.match(twtlParser.T__4)
                self.state = 24
                self.match(twtlParser.T__2)
                self.state = 25
                self.match(twtlParser.T__3)
                self.state = 26
                localctx.low = self.match(twtlParser.INT)
                self.state = 27
                self.match(twtlParser.T__5)
                self.state = 28
                localctx.high = self.match(twtlParser.INT)
                self.state = 29
                self.match(twtlParser.T__4)
                pass
            else:
                raise NoViableAltException(self)

            self._ctx.stop = self._input.LT(-1)
            self.state = 44
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,3,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 42
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
                    if la_ == 1:
                        localctx = twtlParser.FormulaContext(self, _parentctx, _parentState)
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_formula)
                        self.state = 33
                        if not self.precpred(self._ctx, 5):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 5)")
                        self.state = 34
                        localctx.op = self.match(twtlParser.AND)
                        self.state = 35
                        localctx.right = self.formula(6)
                        pass

                    elif la_ == 2:
                        localctx = twtlParser.FormulaContext(self, _parentctx, _parentState)
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_formula)
                        self.state = 36
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 4)")
                        self.state = 37
                        localctx.op = self.match(twtlParser.OR)
                        self.state = 38
                        localctx.right = self.formula(5)
                        pass

                    elif la_ == 3:
                        localctx = twtlParser.FormulaContext(self, _parentctx, _parentState)
                        localctx.left = _prevctx
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_formula)
                        self.state = 39
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                        self.state = 40
                        localctx.op = self.match(twtlParser.CONCAT)
                        self.state = 41
                        localctx.right = self.formula(3)
                        pass

             
                self.state = 46
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,3,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
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
        _startState = 2
        self.enterRecursionRule(localctx, 2, self.RULE_expr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 59
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,4,self._ctx)
            if la_ == 1:
                self.state = 48
                _la = self._input.LA(1)
                if not(_la==twtlParser.T__0 or _la==twtlParser.T__6):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 49
                self.expr(0)
                self.state = 50
                self.match(twtlParser.T__1)
                pass

            elif la_ == 2:
                self.state = 52
                self.match(twtlParser.VARIABLE)
                self.state = 53
                self.match(twtlParser.T__0)
                self.state = 54
                self.expr(0)
                self.state = 55
                self.match(twtlParser.T__1)
                pass

            elif la_ == 3:
                self.state = 57
                self.match(twtlParser.RATIONAL)
                pass

            elif la_ == 4:
                self.state = 58
                self.match(twtlParser.VARIABLE)
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 72
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,6,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 70
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
                    if la_ == 1:
                        localctx = twtlParser.ExprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 61
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 6)")
                        self.state = 62
                        self.match(twtlParser.T__2)
                        self.state = 63
                        self.expr(6)
                        pass

                    elif la_ == 2:
                        localctx = twtlParser.ExprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 64
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 4)")
                        self.state = 65
                        _la = self._input.LA(1)
                        if not(_la==twtlParser.T__7 or _la==twtlParser.T__8):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 66
                        self.expr(5)
                        pass

                    elif la_ == 3:
                        localctx = twtlParser.ExprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                        self.state = 67
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 3)")
                        self.state = 68
                        _la = self._input.LA(1)
                        if not(_la==twtlParser.T__9 or _la==twtlParser.T__10):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 69
                        self.expr(4)
                        pass

             
                self.state = 74
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,6,self._ctx)

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
            self.left = None # ExprContext
            self.op = None # Token
            self.right = None # ExprContext

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(twtlParser.ExprContext)
            else:
                return self.getTypedRuleContext(twtlParser.ExprContext,i)


        def BOOLEAN(self):
            return self.getToken(twtlParser.BOOLEAN, 0)

        def PROP(self):
            return self.getToken(twtlParser.PROP, 0)

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
        self.enterRule(localctx, 4, self.RULE_booleanExpr)
        self._la = 0 # Token type
        try:
            self.state = 81
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [twtlParser.T__0, twtlParser.T__6, twtlParser.VARIABLE, twtlParser.RATIONAL]:
                self.enterOuterAlt(localctx, 1)
                self.state = 75
                localctx.left = self.expr(0)
                self.state = 76
                localctx.op = self._input.LT(1)
                _la = self._input.LA(1)
                if not((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << twtlParser.T__11) | (1 << twtlParser.T__12) | (1 << twtlParser.T__13) | (1 << twtlParser.T__14) | (1 << twtlParser.T__15))) != 0)):
                    localctx.op = self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 77
                localctx.right = self.expr(0)
                pass
            elif token in [twtlParser.BOOLEAN]:
                self.enterOuterAlt(localctx, 2)
                self.state = 79
                localctx.op = self.match(twtlParser.BOOLEAN)
                pass
            elif token in [twtlParser.PROP]:
                self.enterOuterAlt(localctx, 3)
                self.state = 80
                localctx.op = self.match(twtlParser.PROP)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[0] = self.formula_sempred
        self._predicates[1] = self.expr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def formula_sempred(self, localctx:FormulaContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 5)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 4)
         

            if predIndex == 2:
                return self.precpred(self._ctx, 2)
         

    def expr_sempred(self, localctx:ExprContext, predIndex:int):
            if predIndex == 3:
                return self.precpred(self._ctx, 6)
         

            if predIndex == 4:
                return self.precpred(self._ctx, 4)
         

            if predIndex == 5:
                return self.precpred(self._ctx, 3)
         




