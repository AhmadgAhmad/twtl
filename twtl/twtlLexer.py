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
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2\36")
        buf.write("\u00ef\b\1\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7")
        buf.write("\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\4\f\t\f\4\r\t\r")
        buf.write("\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22\4\23")
        buf.write("\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30")
        buf.write("\4\31\t\31\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36")
        buf.write("\t\36\4\37\t\37\4 \t \4!\t!\4\"\t\"\3\2\3\2\3\3\3\3\3")
        buf.write("\4\3\4\3\5\3\5\3\6\3\6\3\7\3\7\3\b\3\b\3\b\3\t\3\t\3\t")
        buf.write("\3\t\3\n\3\n\3\13\3\13\3\f\3\f\3\r\3\r\3\16\3\16\3\16")
        buf.write("\3\17\3\17\3\20\3\20\3\20\3\21\3\21\3\22\3\22\3\22\3\22")
        buf.write("\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3\22")
        buf.write("\3\22\3\22\3\22\5\22}\n\22\3\23\3\23\5\23\u0081\n\23\3")
        buf.write("\23\3\23\3\23\7\23\u0086\n\23\f\23\16\23\u0089\13\23\3")
        buf.write("\24\5\24\u008c\n\24\3\24\7\24\u008f\n\24\f\24\16\24\u0092")
        buf.write("\13\24\3\24\5\24\u0095\n\24\3\24\6\24\u0098\n\24\r\24")
        buf.write("\16\24\u0099\3\24\3\24\3\24\5\24\u009f\n\24\3\24\7\24")
        buf.write("\u00a2\n\24\f\24\16\24\u00a5\13\24\3\25\3\25\3\25\3\25")
        buf.write("\3\25\5\25\u00ac\n\25\3\26\3\26\3\26\3\26\3\26\5\26\u00b3")
        buf.write("\n\26\3\27\3\27\3\30\3\30\3\31\3\31\3\32\3\32\5\32\u00bd")
        buf.write("\n\32\3\32\3\32\3\32\7\32\u00c2\n\32\f\32\16\32\u00c5")
        buf.write("\13\32\3\33\3\33\3\34\3\34\3\35\3\35\3\36\5\36\u00ce\n")
        buf.write("\36\3\37\3\37\5\37\u00d2\n\37\3 \3 \3 \7 \u00d7\n \f ")
        buf.write("\16 \u00da\13 \5 \u00dc\n \3!\3!\3!\3!\7!\u00e2\n!\f!")
        buf.write("\16!\u00e5\13!\3!\3!\3\"\6\"\u00ea\n\"\r\"\16\"\u00eb")
        buf.write("\3\"\3\"\2\2#\3\3\5\4\7\5\t\6\13\7\r\b\17\t\21\n\23\13")
        buf.write("\25\f\27\r\31\16\33\17\35\20\37\21!\22#\23%\24\'\25)\26")
        buf.write("+\27-\30/\31\61\32\63\33\65\2\67\29\2;\2=\2?\34A\35C\36")
        buf.write("\3\2\7\3\2\62;\4\2##\u0080\u0080\5\2CIKXZ\\\4\2\f\f\17")
        buf.write("\17\5\2\13\f\16\17\"\"\2\u0104\2\3\3\2\2\2\2\5\3\2\2\2")
        buf.write("\2\7\3\2\2\2\2\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2\2\2\17")
        buf.write("\3\2\2\2\2\21\3\2\2\2\2\23\3\2\2\2\2\25\3\2\2\2\2\27\3")
        buf.write("\2\2\2\2\31\3\2\2\2\2\33\3\2\2\2\2\35\3\2\2\2\2\37\3\2")
        buf.write("\2\2\2!\3\2\2\2\2#\3\2\2\2\2%\3\2\2\2\2\'\3\2\2\2\2)\3")
        buf.write("\2\2\2\2+\3\2\2\2\2-\3\2\2\2\2/\3\2\2\2\2\61\3\2\2\2\2")
        buf.write("\63\3\2\2\2\2?\3\2\2\2\2A\3\2\2\2\2C\3\2\2\2\3E\3\2\2")
        buf.write("\2\5G\3\2\2\2\7I\3\2\2\2\tK\3\2\2\2\13M\3\2\2\2\rO\3\2")
        buf.write("\2\2\17Q\3\2\2\2\21T\3\2\2\2\23X\3\2\2\2\25Z\3\2\2\2\27")
        buf.write("\\\3\2\2\2\31^\3\2\2\2\33`\3\2\2\2\35c\3\2\2\2\37e\3\2")
        buf.write("\2\2!h\3\2\2\2#|\3\2\2\2%\u0080\3\2\2\2\'\u008b\3\2\2")
        buf.write("\2)\u00ab\3\2\2\2+\u00b2\3\2\2\2-\u00b4\3\2\2\2/\u00b6")
        buf.write("\3\2\2\2\61\u00b8\3\2\2\2\63\u00bc\3\2\2\2\65\u00c6\3")
        buf.write("\2\2\2\67\u00c8\3\2\2\29\u00ca\3\2\2\2;\u00cd\3\2\2\2")
        buf.write("=\u00d1\3\2\2\2?\u00db\3\2\2\2A\u00dd\3\2\2\2C\u00e9\3")
        buf.write("\2\2\2EF\7*\2\2F\4\3\2\2\2GH\7+\2\2H\6\3\2\2\2IJ\7`\2")
        buf.write("\2J\b\3\2\2\2KL\7]\2\2L\n\3\2\2\2MN\7_\2\2N\f\3\2\2\2")
        buf.write("OP\7.\2\2P\16\3\2\2\2QR\7/\2\2RS\7*\2\2S\20\3\2\2\2TU")
        buf.write("\7r\2\2UV\7t\2\2VW\7f\2\2W\22\3\2\2\2XY\7\61\2\2Y\24\3")
        buf.write("\2\2\2Z[\7-\2\2[\26\3\2\2\2\\]\7/\2\2]\30\3\2\2\2^_\7")
        buf.write(">\2\2_\32\3\2\2\2`a\7>\2\2ab\7?\2\2b\34\3\2\2\2cd\7?\2")
        buf.write("\2d\36\3\2\2\2ef\7@\2\2fg\7?\2\2g \3\2\2\2hi\7@\2\2i\"")
        buf.write("\3\2\2\2jk\7v\2\2kl\7t\2\2lm\7w\2\2m}\7g\2\2no\7V\2\2")
        buf.write("op\7t\2\2pq\7w\2\2q}\7g\2\2rs\7h\2\2st\7c\2\2tu\7n\2\2")
        buf.write("uv\7u\2\2v}\7g\2\2wx\7H\2\2xy\7c\2\2yz\7n\2\2z{\7u\2\2")
        buf.write("{}\7g\2\2|j\3\2\2\2|n\3\2\2\2|r\3\2\2\2|w\3\2\2\2}$\3")
        buf.write("\2\2\2~\u0081\5\67\34\2\177\u0081\5;\36\2\u0080~\3\2\2")
        buf.write("\2\u0080\177\3\2\2\2\u0081\u0087\3\2\2\2\u0082\u0086\7")
        buf.write("a\2\2\u0083\u0086\5=\37\2\u0084\u0086\5\65\33\2\u0085")
        buf.write("\u0082\3\2\2\2\u0085\u0083\3\2\2\2\u0085\u0084\3\2\2\2")
        buf.write("\u0086\u0089\3\2\2\2\u0087\u0085\3\2\2\2\u0087\u0088\3")
        buf.write("\2\2\2\u0088&\3\2\2\2\u0089\u0087\3\2\2\2\u008a\u008c")
        buf.write("\7/\2\2\u008b\u008a\3\2\2\2\u008b\u008c\3\2\2\2\u008c")
        buf.write("\u0090\3\2\2\2\u008d\u008f\t\2\2\2\u008e\u008d\3\2\2\2")
        buf.write("\u008f\u0092\3\2\2\2\u0090\u008e\3\2\2\2\u0090\u0091\3")
        buf.write("\2\2\2\u0091\u0094\3\2\2\2\u0092\u0090\3\2\2\2\u0093\u0095")
        buf.write("\7\60\2\2\u0094\u0093\3\2\2\2\u0094\u0095\3\2\2\2\u0095")
        buf.write("\u0097\3\2\2\2\u0096\u0098\t\2\2\2\u0097\u0096\3\2\2\2")
        buf.write("\u0098\u0099\3\2\2\2\u0099\u0097\3\2\2\2\u0099\u009a\3")
        buf.write("\2\2\2\u009a\u009e\3\2\2\2\u009b\u009f\7G\2\2\u009c\u009d")
        buf.write("\7G\2\2\u009d\u009f\7/\2\2\u009e\u009b\3\2\2\2\u009e\u009c")
        buf.write("\3\2\2\2\u009e\u009f\3\2\2\2\u009f\u00a3\3\2\2\2\u00a0")
        buf.write("\u00a2\t\2\2\2\u00a1\u00a0\3\2\2\2\u00a2\u00a5\3\2\2\2")
        buf.write("\u00a3\u00a1\3\2\2\2\u00a3\u00a4\3\2\2\2\u00a4(\3\2\2")
        buf.write("\2\u00a5\u00a3\3\2\2\2\u00a6\u00ac\7(\2\2\u00a7\u00a8")
        buf.write("\7(\2\2\u00a8\u00ac\7(\2\2\u00a9\u00aa\7\61\2\2\u00aa")
        buf.write("\u00ac\7^\2\2\u00ab\u00a6\3\2\2\2\u00ab\u00a7\3\2\2\2")
        buf.write("\u00ab\u00a9\3\2\2\2\u00ac*\3\2\2\2\u00ad\u00b3\7~\2\2")
        buf.write("\u00ae\u00af\7~\2\2\u00af\u00b3\7~\2\2\u00b0\u00b1\7^")
        buf.write("\2\2\u00b1\u00b3\7\61\2\2\u00b2\u00ad\3\2\2\2\u00b2\u00ae")
        buf.write("\3\2\2\2\u00b2\u00b0\3\2\2\2\u00b3,\3\2\2\2\u00b4\u00b5")
        buf.write("\t\3\2\2\u00b5.\3\2\2\2\u00b6\u00b7\7J\2\2\u00b7\60\3")
        buf.write("\2\2\2\u00b8\u00b9\7,\2\2\u00b9\62\3\2\2\2\u00ba\u00bd")
        buf.write("\5\67\34\2\u00bb\u00bd\5;\36\2\u00bc\u00ba\3\2\2\2\u00bc")
        buf.write("\u00bb\3\2\2\2\u00bd\u00c3\3\2\2\2\u00be\u00c2\7a\2\2")
        buf.write("\u00bf\u00c2\5=\37\2\u00c0\u00c2\5\65\33\2\u00c1\u00be")
        buf.write("\3\2\2\2\u00c1\u00bf\3\2\2\2\u00c1\u00c0\3\2\2\2\u00c2")
        buf.write("\u00c5\3\2\2\2\u00c3\u00c1\3\2\2\2\u00c3\u00c4\3\2\2\2")
        buf.write("\u00c4\64\3\2\2\2\u00c5\u00c3\3\2\2\2\u00c6\u00c7\4\62")
        buf.write(";\2\u00c7\66\3\2\2\2\u00c8\u00c9\4c|\2\u00c98\3\2\2\2")
        buf.write("\u00ca\u00cb\4C\\\2\u00cb:\3\2\2\2\u00cc\u00ce\t\4\2\2")
        buf.write("\u00cd\u00cc\3\2\2\2\u00ce<\3\2\2\2\u00cf\u00d2\5\67\34")
        buf.write("\2\u00d0\u00d2\59\35\2\u00d1\u00cf\3\2\2\2\u00d1\u00d0")
        buf.write("\3\2\2\2\u00d2>\3\2\2\2\u00d3\u00dc\7\62\2\2\u00d4\u00d8")
        buf.write("\4\63;\2\u00d5\u00d7\5\65\33\2\u00d6\u00d5\3\2\2\2\u00d7")
        buf.write("\u00da\3\2\2\2\u00d8\u00d6\3\2\2\2\u00d8\u00d9\3\2\2\2")
        buf.write("\u00d9\u00dc\3\2\2\2\u00da\u00d8\3\2\2\2\u00db\u00d3\3")
        buf.write("\2\2\2\u00db\u00d4\3\2\2\2\u00dc@\3\2\2\2\u00dd\u00de")
        buf.write("\7\61\2\2\u00de\u00df\7\61\2\2\u00df\u00e3\3\2\2\2\u00e0")
        buf.write("\u00e2\n\5\2\2\u00e1\u00e0\3\2\2\2\u00e2\u00e5\3\2\2\2")
        buf.write("\u00e3\u00e1\3\2\2\2\u00e3\u00e4\3\2\2\2\u00e4\u00e6\3")
        buf.write("\2\2\2\u00e5\u00e3\3\2\2\2\u00e6\u00e7\b!\2\2\u00e7B\3")
        buf.write("\2\2\2\u00e8\u00ea\t\6\2\2\u00e9\u00e8\3\2\2\2\u00ea\u00eb")
        buf.write("\3\2\2\2\u00eb\u00e9\3\2\2\2\u00eb\u00ec\3\2\2\2\u00ec")
        buf.write("\u00ed\3\2\2\2\u00ed\u00ee\b\"\2\2\u00eeD\3\2\2\2\30\2")
        buf.write("|\u0080\u0085\u0087\u008b\u0090\u0094\u0099\u009e\u00a3")
        buf.write("\u00ab\u00b2\u00bc\u00c1\u00c3\u00cd\u00d1\u00d8\u00db")
        buf.write("\u00e3\u00eb\3\b\2\2")
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
    BOOLEAN = 17
    VARIABLE = 18
    RATIONAL = 19
    AND = 20
    OR = 21
    NOT = 22
    HOLD = 23
    CONCAT = 24
    PROP = 25
    INT = 26
    LINECMT = 27
    WS = 28

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
            "'('", "')'", "'^'", "'['", "']'", "','", "'-('", "'prd'", "'/'", 
            "'+'", "'-'", "'<'", "'<='", "'='", "'>='", "'>'", "'H'", "'*'" ]

    symbolicNames = [ "<INVALID>",
            "BOOLEAN", "VARIABLE", "RATIONAL", "AND", "OR", "NOT", "HOLD", 
            "CONCAT", "PROP", "INT", "LINECMT", "WS" ]

    ruleNames = [ "T__0", "T__1", "T__2", "T__3", "T__4", "T__5", "T__6", 
                  "T__7", "T__8", "T__9", "T__10", "T__11", "T__12", "T__13", 
                  "T__14", "T__15", "BOOLEAN", "VARIABLE", "RATIONAL", "AND", 
                  "OR", "NOT", "HOLD", "CONCAT", "PROP", "DIGIT", "LWLETTER", 
                  "HGLETTER", "HGLETTERALL", "LETTER", "INT", "LINECMT", 
                  "WS" ]

    grammarFileName = "twtl.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.7.1")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


