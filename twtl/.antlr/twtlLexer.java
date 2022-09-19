// Generated from /home/ahmad/Desktop/twtl/twtl/twtl__.g4 by ANTLR 4.9.2

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

import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class twtlLexer extends Lexer {
	static { RuntimeMetaData.checkVersion("4.9.2", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		T__0=1, T__1=2, T__2=3, T__3=4, T__4=5, T__5=6, AND=7, OR=8, NOT=9, HOLD=10, 
		CONCAT=11, INT=12, TRUE=13, FALSE=14, PROP=15, LINECMT=16, WS=17;
	public static String[] channelNames = {
		"DEFAULT_TOKEN_CHANNEL", "HIDDEN"
	};

	public static String[] modeNames = {
		"DEFAULT_MODE"
	};

	private static String[] makeRuleNames() {
		return new String[] {
			"T__0", "T__1", "T__2", "T__3", "T__4", "T__5", "DIGIT", "LWLETTER", 
			"HGLETTER", "HGLETTERALL", "LETTER", "AND", "OR", "NOT", "HOLD", "CONCAT", 
			"INT", "TRUE", "FALSE", "PROP", "LINECMT", "WS"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'('", "')'", "'^'", "'['", "']'", "','", null, null, null, "'H'", 
			"'*'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, null, null, null, null, null, null, "AND", "OR", "NOT", "HOLD", 
			"CONCAT", "INT", "TRUE", "FALSE", "PROP", "LINECMT", "WS"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}


	public twtlLexer(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "twtl__.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public String[] getChannelNames() { return channelNames; }

	@Override
	public String[] getModeNames() { return modeNames; }

	@Override
	public ATN getATN() { return _ATN; }

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2\23\u0096\b\1\4\2"+
		"\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4"+
		"\13\t\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22"+
		"\t\22\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\3\2\3\2\3\3\3"+
		"\3\3\4\3\4\3\5\3\5\3\6\3\6\3\7\3\7\3\b\3\b\3\t\3\t\3\n\3\n\3\13\5\13C"+
		"\n\13\3\f\3\f\5\fG\n\f\3\r\3\r\3\r\5\rL\n\r\3\16\3\16\3\16\5\16Q\n\16"+
		"\3\17\3\17\3\20\3\20\3\21\3\21\3\22\3\22\3\22\7\22\\\n\22\f\22\16\22_"+
		"\13\22\5\22a\n\22\3\23\3\23\3\23\3\23\3\23\3\23\3\23\3\23\5\23k\n\23\3"+
		"\24\3\24\3\24\3\24\3\24\3\24\3\24\3\24\3\24\3\24\5\24w\n\24\3\25\3\25"+
		"\5\25{\n\25\3\25\3\25\3\25\7\25\u0080\n\25\f\25\16\25\u0083\13\25\3\26"+
		"\3\26\3\26\3\26\7\26\u0089\n\26\f\26\16\26\u008c\13\26\3\26\3\26\3\27"+
		"\6\27\u0091\n\27\r\27\16\27\u0092\3\27\3\27\2\2\30\3\3\5\4\7\5\t\6\13"+
		"\7\r\b\17\2\21\2\23\2\25\2\27\2\31\t\33\n\35\13\37\f!\r#\16%\17\'\20)"+
		"\21+\22-\23\3\2\6\5\2CIKXZ\\\4\2##\u0080\u0080\4\2\f\f\17\17\5\2\13\f"+
		"\16\17\"\"\2\u009d\2\3\3\2\2\2\2\5\3\2\2\2\2\7\3\2\2\2\2\t\3\2\2\2\2\13"+
		"\3\2\2\2\2\r\3\2\2\2\2\31\3\2\2\2\2\33\3\2\2\2\2\35\3\2\2\2\2\37\3\2\2"+
		"\2\2!\3\2\2\2\2#\3\2\2\2\2%\3\2\2\2\2\'\3\2\2\2\2)\3\2\2\2\2+\3\2\2\2"+
		"\2-\3\2\2\2\3/\3\2\2\2\5\61\3\2\2\2\7\63\3\2\2\2\t\65\3\2\2\2\13\67\3"+
		"\2\2\2\r9\3\2\2\2\17;\3\2\2\2\21=\3\2\2\2\23?\3\2\2\2\25B\3\2\2\2\27F"+
		"\3\2\2\2\31K\3\2\2\2\33P\3\2\2\2\35R\3\2\2\2\37T\3\2\2\2!V\3\2\2\2#`\3"+
		"\2\2\2%j\3\2\2\2\'v\3\2\2\2)z\3\2\2\2+\u0084\3\2\2\2-\u0090\3\2\2\2/\60"+
		"\7*\2\2\60\4\3\2\2\2\61\62\7+\2\2\62\6\3\2\2\2\63\64\7`\2\2\64\b\3\2\2"+
		"\2\65\66\7]\2\2\66\n\3\2\2\2\678\7_\2\28\f\3\2\2\29:\7.\2\2:\16\3\2\2"+
		"\2;<\4\62;\2<\20\3\2\2\2=>\4c|\2>\22\3\2\2\2?@\4C\\\2@\24\3\2\2\2AC\t"+
		"\2\2\2BA\3\2\2\2C\26\3\2\2\2DG\5\21\t\2EG\5\23\n\2FD\3\2\2\2FE\3\2\2\2"+
		"G\30\3\2\2\2HL\7(\2\2IJ\7(\2\2JL\7(\2\2KH\3\2\2\2KI\3\2\2\2L\32\3\2\2"+
		"\2MQ\7~\2\2NO\7~\2\2OQ\7~\2\2PM\3\2\2\2PN\3\2\2\2Q\34\3\2\2\2RS\t\3\2"+
		"\2S\36\3\2\2\2TU\7J\2\2U \3\2\2\2VW\7,\2\2W\"\3\2\2\2Xa\7\62\2\2Y]\4\63"+
		";\2Z\\\5\17\b\2[Z\3\2\2\2\\_\3\2\2\2][\3\2\2\2]^\3\2\2\2^a\3\2\2\2_]\3"+
		"\2\2\2`X\3\2\2\2`Y\3\2\2\2a$\3\2\2\2bc\7V\2\2cd\7t\2\2de\7w\2\2ek\7g\2"+
		"\2fg\7v\2\2gh\7t\2\2hi\7w\2\2ik\7g\2\2jb\3\2\2\2jf\3\2\2\2k&\3\2\2\2l"+
		"m\7H\2\2mn\7c\2\2no\7n\2\2op\7u\2\2pw\7g\2\2qr\7h\2\2rs\7c\2\2st\7n\2"+
		"\2tu\7u\2\2uw\7g\2\2vl\3\2\2\2vq\3\2\2\2w(\3\2\2\2x{\5\21\t\2y{\5\25\13"+
		"\2zx\3\2\2\2zy\3\2\2\2{\u0081\3\2\2\2|\u0080\7a\2\2}\u0080\5\27\f\2~\u0080"+
		"\5\17\b\2\177|\3\2\2\2\177}\3\2\2\2\177~\3\2\2\2\u0080\u0083\3\2\2\2\u0081"+
		"\177\3\2\2\2\u0081\u0082\3\2\2\2\u0082*\3\2\2\2\u0083\u0081\3\2\2\2\u0084"+
		"\u0085\7\61\2\2\u0085\u0086\7\61\2\2\u0086\u008a\3\2\2\2\u0087\u0089\n"+
		"\4\2\2\u0088\u0087\3\2\2\2\u0089\u008c\3\2\2\2\u008a\u0088\3\2\2\2\u008a"+
		"\u008b\3\2\2\2\u008b\u008d\3\2\2\2\u008c\u008a\3\2\2\2\u008d\u008e\b\26"+
		"\2\2\u008e,\3\2\2\2\u008f\u0091\t\5\2\2\u0090\u008f\3\2\2\2\u0091\u0092"+
		"\3\2\2\2\u0092\u0090\3\2\2\2\u0092\u0093\3\2\2\2\u0093\u0094\3\2\2\2\u0094"+
		"\u0095\b\27\2\2\u0095.\3\2\2\2\20\2BFKP]`jvz\177\u0081\u008a\u0092\3\b"+
		"\2\2";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}