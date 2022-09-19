# Generated from twtl.g4 by ANTLR 4.7.1
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
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2\35")
        buf.write("\u00d3\b\1\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7")
        buf.write("\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\4\f\t\f\4\r\t\r")
        buf.write("\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22\4\23")
        buf.write("\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30")
        buf.write("\4\31\t\31\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36")
        buf.write("\t\36\4\37\t\37\4 \t \4!\t!\3\2\3\2\3\3\3\3\3\4\3\4\3")
        buf.write("\5\3\5\3\6\3\6\3\7\3\7\3\b\3\b\3\t\3\t\3\t\3\n\3\n\3\13")
        buf.write("\3\13\3\13\3\f\3\f\3\r\3\r\3\r\3\16\3\16\3\17\3\17\3\20")
        buf.write("\3\20\3\21\3\21\3\22\3\22\3\23\3\23\3\24\3\24\3\25\5\25")
        buf.write("n\n\25\3\26\3\26\5\26r\n\26\3\27\3\27\3\27\5\27w\n\27")
        buf.write("\3\30\3\30\3\30\5\30|\n\30\3\31\3\31\3\32\3\32\3\33\3")
        buf.write("\33\3\34\5\34\u0085\n\34\3\34\7\34\u0088\n\34\f\34\16")
        buf.write("\34\u008b\13\34\3\34\5\34\u008e\n\34\3\34\6\34\u0091\n")
        buf.write("\34\r\34\16\34\u0092\3\34\3\34\3\34\5\34\u0098\n\34\3")
        buf.write("\34\7\34\u009b\n\34\f\34\16\34\u009e\13\34\3\35\3\35\3")
        buf.write("\35\3\35\3\35\3\35\3\35\3\35\5\35\u00a8\n\35\3\36\3\36")
        buf.write("\3\36\3\36\3\36\3\36\3\36\3\36\3\36\3\36\5\36\u00b4\n")
        buf.write("\36\3\37\3\37\5\37\u00b8\n\37\3\37\3\37\3\37\7\37\u00bd")
        buf.write("\n\37\f\37\16\37\u00c0\13\37\3 \3 \3 \3 \7 \u00c6\n \f")
        buf.write(" \16 \u00c9\13 \3 \3 \3!\6!\u00ce\n!\r!\16!\u00cf\3!\3")
        buf.write("!\2\2\"\3\3\5\4\7\5\t\6\13\7\r\b\17\t\21\n\23\13\25\f")
        buf.write("\27\r\31\16\33\17\35\20\37\21!\22#\2%\2\'\2)\2+\2-\23")
        buf.write("/\24\61\25\63\26\65\27\67\309\31;\32=\33?\34A\35\3\2\7")
        buf.write("\5\2CIKXZ\\\4\2##\u0080\u0080\3\2\62;\4\2\f\f\17\17\5")
        buf.write("\2\13\f\16\17\"\"\2\u00df\2\3\3\2\2\2\2\5\3\2\2\2\2\7")
        buf.write("\3\2\2\2\2\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3\2")
        buf.write("\2\2\2\21\3\2\2\2\2\23\3\2\2\2\2\25\3\2\2\2\2\27\3\2\2")
        buf.write("\2\2\31\3\2\2\2\2\33\3\2\2\2\2\35\3\2\2\2\2\37\3\2\2\2")
        buf.write("\2!\3\2\2\2\2-\3\2\2\2\2/\3\2\2\2\2\61\3\2\2\2\2\63\3")
        buf.write("\2\2\2\2\65\3\2\2\2\2\67\3\2\2\2\29\3\2\2\2\2;\3\2\2\2")
        buf.write("\2=\3\2\2\2\2?\3\2\2\2\2A\3\2\2\2\3C\3\2\2\2\5E\3\2\2")
        buf.write("\2\7G\3\2\2\2\tI\3\2\2\2\13K\3\2\2\2\rM\3\2\2\2\17O\3")
        buf.write("\2\2\2\21Q\3\2\2\2\23T\3\2\2\2\25V\3\2\2\2\27Y\3\2\2\2")
        buf.write("\31[\3\2\2\2\33^\3\2\2\2\35`\3\2\2\2\37b\3\2\2\2!d\3\2")
        buf.write("\2\2#f\3\2\2\2%h\3\2\2\2\'j\3\2\2\2)m\3\2\2\2+q\3\2\2")
        buf.write("\2-v\3\2\2\2/{\3\2\2\2\61}\3\2\2\2\63\177\3\2\2\2\65\u0081")
        buf.write("\3\2\2\2\67\u0084\3\2\2\29\u00a7\3\2\2\2;\u00b3\3\2\2")
        buf.write("\2=\u00b7\3\2\2\2?\u00c1\3\2\2\2A\u00cd\3\2\2\2CD\7*\2")
        buf.write("\2D\4\3\2\2\2EF\7+\2\2F\6\3\2\2\2GH\7`\2\2H\b\3\2\2\2")
        buf.write("IJ\7]\2\2J\n\3\2\2\2KL\7_\2\2L\f\3\2\2\2MN\7.\2\2N\16")
        buf.write("\3\2\2\2OP\7>\2\2P\20\3\2\2\2QR\7>\2\2RS\7?\2\2S\22\3")
        buf.write("\2\2\2TU\7?\2\2U\24\3\2\2\2VW\7@\2\2WX\7?\2\2X\26\3\2")
        buf.write("\2\2YZ\7@\2\2Z\30\3\2\2\2[\\\7/\2\2\\]\7*\2\2]\32\3\2")
        buf.write("\2\2^_\7,\2\2_\34\3\2\2\2`a\7\61\2\2a\36\3\2\2\2bc\7-")
        buf.write("\2\2c \3\2\2\2de\7/\2\2e\"\3\2\2\2fg\4\62;\2g$\3\2\2\2")
        buf.write("hi\4c|\2i&\3\2\2\2jk\4C\\\2k(\3\2\2\2ln\t\2\2\2ml\3\2")
        buf.write("\2\2n*\3\2\2\2or\5%\23\2pr\5\'\24\2qo\3\2\2\2qp\3\2\2")
        buf.write("\2r,\3\2\2\2sw\7(\2\2tu\7(\2\2uw\7(\2\2vs\3\2\2\2vt\3")
        buf.write("\2\2\2w.\3\2\2\2x|\7~\2\2yz\7~\2\2z|\7~\2\2{x\3\2\2\2")
        buf.write("{y\3\2\2\2|\60\3\2\2\2}~\t\3\2\2~\62\3\2\2\2\177\u0080")
        buf.write("\7J\2\2\u0080\64\3\2\2\2\u0081\u0082\7\60\2\2\u0082\66")
        buf.write("\3\2\2\2\u0083\u0085\7/\2\2\u0084\u0083\3\2\2\2\u0084")
        buf.write("\u0085\3\2\2\2\u0085\u0089\3\2\2\2\u0086\u0088\t\4\2\2")
        buf.write("\u0087\u0086\3\2\2\2\u0088\u008b\3\2\2\2\u0089\u0087\3")
        buf.write("\2\2\2\u0089\u008a\3\2\2\2\u008a\u008d\3\2\2\2\u008b\u0089")
        buf.write("\3\2\2\2\u008c\u008e\7\60\2\2\u008d\u008c\3\2\2\2\u008d")
        buf.write("\u008e\3\2\2\2\u008e\u0090\3\2\2\2\u008f\u0091\t\4\2\2")
        buf.write("\u0090\u008f\3\2\2\2\u0091\u0092\3\2\2\2\u0092\u0090\3")
        buf.write("\2\2\2\u0092\u0093\3\2\2\2\u0093\u0097\3\2\2\2\u0094\u0098")
        buf.write("\7G\2\2\u0095\u0096\7G\2\2\u0096\u0098\7/\2\2\u0097\u0094")
        buf.write("\3\2\2\2\u0097\u0095\3\2\2\2\u0097\u0098\3\2\2\2\u0098")
        buf.write("\u009c\3\2\2\2\u0099\u009b\t\4\2\2\u009a\u0099\3\2\2\2")
        buf.write("\u009b\u009e\3\2\2\2\u009c\u009a\3\2\2\2\u009c\u009d\3")
        buf.write("\2\2\2\u009d8\3\2\2\2\u009e\u009c\3\2\2\2\u009f\u00a0")
        buf.write("\7V\2\2\u00a0\u00a1\7t\2\2\u00a1\u00a2\7w\2\2\u00a2\u00a8")
        buf.write("\7g\2\2\u00a3\u00a4\7v\2\2\u00a4\u00a5\7t\2\2\u00a5\u00a6")
        buf.write("\7w\2\2\u00a6\u00a8\7g\2\2\u00a7\u009f\3\2\2\2\u00a7\u00a3")
        buf.write("\3\2\2\2\u00a8:\3\2\2\2\u00a9\u00aa\7H\2\2\u00aa\u00ab")
        buf.write("\7c\2\2\u00ab\u00ac\7n\2\2\u00ac\u00ad\7u\2\2\u00ad\u00b4")
        buf.write("\7g\2\2\u00ae\u00af\7h\2\2\u00af\u00b0\7c\2\2\u00b0\u00b1")
        buf.write("\7n\2\2\u00b1\u00b2\7u\2\2\u00b2\u00b4\7g\2\2\u00b3\u00a9")
        buf.write("\3\2\2\2\u00b3\u00ae\3\2\2\2\u00b4<\3\2\2\2\u00b5\u00b8")
        buf.write("\5%\23\2\u00b6\u00b8\5)\25\2\u00b7\u00b5\3\2\2\2\u00b7")
        buf.write("\u00b6\3\2\2\2\u00b8\u00be\3\2\2\2\u00b9\u00bd\7a\2\2")
        buf.write("\u00ba\u00bd\5+\26\2\u00bb\u00bd\5#\22\2\u00bc\u00b9\3")
        buf.write("\2\2\2\u00bc\u00ba\3\2\2\2\u00bc\u00bb\3\2\2\2\u00bd\u00c0")
        buf.write("\3\2\2\2\u00be\u00bc\3\2\2\2\u00be\u00bf\3\2\2\2\u00bf")
        buf.write(">\3\2\2\2\u00c0\u00be\3\2\2\2\u00c1\u00c2\7\61\2\2\u00c2")
        buf.write("\u00c3\7\61\2\2\u00c3\u00c7\3\2\2\2\u00c4\u00c6\n\5\2")
        buf.write("\2\u00c5\u00c4\3\2\2\2\u00c6\u00c9\3\2\2\2\u00c7\u00c5")
        buf.write("\3\2\2\2\u00c7\u00c8\3\2\2\2\u00c8\u00ca\3\2\2\2\u00c9")
        buf.write("\u00c7\3\2\2\2\u00ca\u00cb\b \2\2\u00cb@\3\2\2\2\u00cc")
        buf.write("\u00ce\t\6\2\2\u00cd\u00cc\3\2\2\2\u00ce\u00cf\3\2\2\2")
        buf.write("\u00cf\u00cd\3\2\2\2\u00cf\u00d0\3\2\2\2\u00d0\u00d1\3")
        buf.write("\2\2\2\u00d1\u00d2\b!\2\2\u00d2B\3\2\2\2\24\2mqv{\u0084")
        buf.write("\u0089\u008d\u0092\u0097\u009c\u00a7\u00b3\u00b7\u00bc")
        buf.write("\u00be\u00c7\u00cf\3\b\2\2")
        return buf.getvalue()


class twtlLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    T__0 = 1
    T__1 = 2
    T__2 = 3
    T__3 = 4
    T__4 = 5
    T__5 = 6
    T__6 = 7
    T__7 = 8
    T__8 = 9
    T__9 = 10
    T__10 = 11
    T__11 = 12
    T__12 = 13
    T__13 = 14
    T__14 = 15
    T__15 = 16
    AND = 17
    OR = 18
    NOT = 19
    HOLD = 20
    CONCAT = 21
    RATIONAL = 22
    TRUE = 23
    FALSE = 24
    VARIABLE = 25
    LINECMT = 26
    WS = 27

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
            "'('", "')'", "'^'", "'['", "']'", "','", "'<'", "'<='", "'='", 
            "'>='", "'>'", "'-('", "'*'", "'/'", "'+'", "'-'", "'H'", "'.'" ]

    symbolicNames = [ "<INVALID>",
            "AND", "OR", "NOT", "HOLD", "CONCAT", "RATIONAL", "TRUE", "FALSE", 
            "VARIABLE", "LINECMT", "WS" ]

    ruleNames = [ "T__0", "T__1", "T__2", "T__3", "T__4", "T__5", "T__6", 
                  "T__7", "T__8", "T__9", "T__10", "T__11", "T__12", "T__13", 
                  "T__14", "T__15", "DIGIT", "LWLETTER", "HGLETTER", "HGLETTERALL", 
                  "LETTER", "AND", "OR", "NOT", "HOLD", "CONCAT", "RATIONAL", 
                  "TRUE", "FALSE", "VARIABLE", "LINECMT", "WS" ]

    grammarFileName = "twtl.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.7.1")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


