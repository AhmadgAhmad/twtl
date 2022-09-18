// Generated from /home/ahmad/Desktop/twtl/twtl/twtl.g4 by ANTLR 4.9.2

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

import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class twtlParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.9.2", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		T__0=1, T__1=2, T__2=3, T__3=4, T__4=5, T__5=6, T__6=7, T__7=8, T__8=9, 
		T__9=10, T__10=11, T__11=12, T__12=13, T__13=14, T__14=15, T__15=16, BOOLEAN=17, 
		VARIABLE=18, RATIONAL=19, AND=20, OR=21, NOT=22, HOLD=23, CONCAT=24, PROP=25, 
		INT=26, LINECMT=27, WS=28;
	public static final int
		RULE_formula = 0, RULE_expr = 1, RULE_booleanExpr = 2;
	private static String[] makeRuleNames() {
		return new String[] {
			"formula", "expr", "booleanExpr"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'('", "')'", "'^'", "'['", "']'", "','", "'-('", "'prd'", "'/'", 
			"'+'", "'-'", "'<'", "'<='", "'='", "'>='", "'>'", null, null, null, 
			null, null, null, "'H'", "'*'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, "BOOLEAN", "VARIABLE", "RATIONAL", "AND", 
			"OR", "NOT", "HOLD", "CONCAT", "PROP", "INT", "LINECMT", "WS"
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

	@Override
	public String getGrammarFileName() { return "twtl.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public twtlParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	public static class FormulaContext extends ParserRuleContext {
		public FormulaContext left;
		public FormulaContext child;
		public Token op;
		public Token duration;
		public Token negated;
		public BooleanExprContext prop;
		public Token low;
		public Token high;
		public FormulaContext right;
		public List<FormulaContext> formula() {
			return getRuleContexts(FormulaContext.class);
		}
		public FormulaContext formula(int i) {
			return getRuleContext(FormulaContext.class,i);
		}
		public BooleanExprContext booleanExpr() {
			return getRuleContext(BooleanExprContext.class,0);
		}
		public TerminalNode HOLD() { return getToken(twtlParser.HOLD, 0); }
		public List<TerminalNode> INT() { return getTokens(twtlParser.INT); }
		public TerminalNode INT(int i) {
			return getToken(twtlParser.INT, i);
		}
		public TerminalNode NOT() { return getToken(twtlParser.NOT, 0); }
		public TerminalNode AND() { return getToken(twtlParser.AND, 0); }
		public TerminalNode OR() { return getToken(twtlParser.OR, 0); }
		public TerminalNode CONCAT() { return getToken(twtlParser.CONCAT, 0); }
		public FormulaContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_formula; }
	}

	public final FormulaContext formula() throws RecognitionException {
		return formula(0);
	}

	private FormulaContext formula(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		FormulaContext _localctx = new FormulaContext(_ctx, _parentState);
		FormulaContext _prevctx = _localctx;
		int _startState = 0;
		enterRecursionRule(_localctx, 0, RULE_formula, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(32);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,2,_ctx) ) {
			case 1:
				{
				setState(7);
				match(T__0);
				setState(8);
				((FormulaContext)_localctx).child = formula(0);
				setState(9);
				match(T__1);
				}
				break;
			case 2:
				{
				setState(14);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==HOLD) {
					{
					setState(11);
					((FormulaContext)_localctx).op = match(HOLD);
					setState(12);
					match(T__2);
					setState(13);
					((FormulaContext)_localctx).duration = match(INT);
					}
				}

				setState(17);
				_errHandler.sync(this);
				_la = _input.LA(1);
				if (_la==NOT) {
					{
					setState(16);
					((FormulaContext)_localctx).negated = match(NOT);
					}
				}

				setState(19);
				((FormulaContext)_localctx).prop = booleanExpr();
				}
				break;
			case 3:
				{
				setState(20);
				((FormulaContext)_localctx).op = match(NOT);
				setState(21);
				((FormulaContext)_localctx).child = formula(3);
				}
				break;
			case 4:
				{
				setState(22);
				((FormulaContext)_localctx).op = match(T__3);
				setState(23);
				((FormulaContext)_localctx).child = formula(0);
				setState(24);
				match(T__4);
				setState(25);
				match(T__2);
				setState(26);
				match(T__3);
				setState(27);
				((FormulaContext)_localctx).low = match(INT);
				setState(28);
				match(T__5);
				setState(29);
				((FormulaContext)_localctx).high = match(INT);
				setState(30);
				match(T__4);
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(45);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,4,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(43);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,3,_ctx) ) {
					case 1:
						{
						_localctx = new FormulaContext(_parentctx, _parentState);
						_localctx.left = _prevctx;
						_localctx.left = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_formula);
						setState(34);
						if (!(precpred(_ctx, 5))) throw new FailedPredicateException(this, "precpred(_ctx, 5)");
						setState(35);
						((FormulaContext)_localctx).op = match(AND);
						setState(36);
						((FormulaContext)_localctx).right = formula(6);
						}
						break;
					case 2:
						{
						_localctx = new FormulaContext(_parentctx, _parentState);
						_localctx.left = _prevctx;
						_localctx.left = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_formula);
						setState(37);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(38);
						((FormulaContext)_localctx).op = match(OR);
						setState(39);
						((FormulaContext)_localctx).right = formula(5);
						}
						break;
					case 3:
						{
						_localctx = new FormulaContext(_parentctx, _parentState);
						_localctx.left = _prevctx;
						_localctx.left = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_formula);
						setState(40);
						if (!(precpred(_ctx, 2))) throw new FailedPredicateException(this, "precpred(_ctx, 2)");
						setState(41);
						((FormulaContext)_localctx).op = match(CONCAT);
						setState(42);
						((FormulaContext)_localctx).right = formula(3);
						}
						break;
					}
					} 
				}
				setState(47);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,4,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class ExprContext extends ParserRuleContext {
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode VARIABLE() { return getToken(twtlParser.VARIABLE, 0); }
		public TerminalNode RATIONAL() { return getToken(twtlParser.RATIONAL, 0); }
		public ExprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expr; }
	}

	public final ExprContext expr() throws RecognitionException {
		return expr(0);
	}

	private ExprContext expr(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		ExprContext _localctx = new ExprContext(_ctx, _parentState);
		ExprContext _prevctx = _localctx;
		int _startState = 2;
		enterRecursionRule(_localctx, 2, RULE_expr, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(60);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,5,_ctx) ) {
			case 1:
				{
				setState(49);
				_la = _input.LA(1);
				if ( !(_la==T__0 || _la==T__6) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(50);
				expr(0);
				setState(51);
				match(T__1);
				}
				break;
			case 2:
				{
				setState(53);
				match(VARIABLE);
				setState(54);
				match(T__0);
				setState(55);
				expr(0);
				setState(56);
				match(T__1);
				}
				break;
			case 3:
				{
				setState(58);
				match(RATIONAL);
				}
				break;
			case 4:
				{
				setState(59);
				match(VARIABLE);
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(73);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,7,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(71);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,6,_ctx) ) {
					case 1:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(62);
						if (!(precpred(_ctx, 6))) throw new FailedPredicateException(this, "precpred(_ctx, 6)");
						setState(63);
						match(T__2);
						setState(64);
						expr(6);
						}
						break;
					case 2:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(65);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(66);
						_la = _input.LA(1);
						if ( !(_la==T__7 || _la==T__8) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(67);
						expr(5);
						}
						break;
					case 3:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(68);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(69);
						_la = _input.LA(1);
						if ( !(_la==T__9 || _la==T__10) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(70);
						expr(4);
						}
						break;
					}
					} 
				}
				setState(75);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,7,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class BooleanExprContext extends ParserRuleContext {
		public ExprContext left;
		public Token op;
		public ExprContext right;
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode BOOLEAN() { return getToken(twtlParser.BOOLEAN, 0); }
		public TerminalNode PROP() { return getToken(twtlParser.PROP, 0); }
		public BooleanExprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_booleanExpr; }
	}

	public final BooleanExprContext booleanExpr() throws RecognitionException {
		BooleanExprContext _localctx = new BooleanExprContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_booleanExpr);
		int _la;
		try {
			setState(82);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case T__0:
			case T__6:
			case VARIABLE:
			case RATIONAL:
				enterOuterAlt(_localctx, 1);
				{
				setState(76);
				((BooleanExprContext)_localctx).left = expr(0);
				setState(77);
				((BooleanExprContext)_localctx).op = _input.LT(1);
				_la = _input.LA(1);
				if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << T__11) | (1L << T__12) | (1L << T__13) | (1L << T__14) | (1L << T__15))) != 0)) ) {
					((BooleanExprContext)_localctx).op = (Token)_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(78);
				((BooleanExprContext)_localctx).right = expr(0);
				}
				break;
			case BOOLEAN:
				enterOuterAlt(_localctx, 2);
				{
				setState(80);
				((BooleanExprContext)_localctx).op = match(BOOLEAN);
				}
				break;
			case PROP:
				enterOuterAlt(_localctx, 3);
				{
				setState(81);
				((BooleanExprContext)_localctx).op = match(PROP);
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
		switch (ruleIndex) {
		case 0:
			return formula_sempred((FormulaContext)_localctx, predIndex);
		case 1:
			return expr_sempred((ExprContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean formula_sempred(FormulaContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0:
			return precpred(_ctx, 5);
		case 1:
			return precpred(_ctx, 4);
		case 2:
			return precpred(_ctx, 2);
		}
		return true;
	}
	private boolean expr_sempred(ExprContext _localctx, int predIndex) {
		switch (predIndex) {
		case 3:
			return precpred(_ctx, 6);
		case 4:
			return precpred(_ctx, 4);
		case 5:
			return precpred(_ctx, 3);
		}
		return true;
	}

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\36W\4\2\t\2\4\3\t"+
		"\3\4\4\t\4\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\5\2\21\n\2\3\2\5\2\24\n\2\3"+
		"\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\5\2#\n\2\3\2\3\2\3"+
		"\2\3\2\3\2\3\2\3\2\3\2\3\2\7\2.\n\2\f\2\16\2\61\13\2\3\3\3\3\3\3\3\3\3"+
		"\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\5\3?\n\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3"+
		"\3\3\3\7\3J\n\3\f\3\16\3M\13\3\3\4\3\4\3\4\3\4\3\4\3\4\5\4U\n\4\3\4\2"+
		"\4\2\4\5\2\4\6\2\6\4\2\3\3\t\t\3\2\n\13\3\2\f\r\3\2\16\22\2c\2\"\3\2\2"+
		"\2\4>\3\2\2\2\6T\3\2\2\2\b\t\b\2\1\2\t\n\7\3\2\2\n\13\5\2\2\2\13\f\7\4"+
		"\2\2\f#\3\2\2\2\r\16\7\31\2\2\16\17\7\5\2\2\17\21\7\34\2\2\20\r\3\2\2"+
		"\2\20\21\3\2\2\2\21\23\3\2\2\2\22\24\7\30\2\2\23\22\3\2\2\2\23\24\3\2"+
		"\2\2\24\25\3\2\2\2\25#\5\6\4\2\26\27\7\30\2\2\27#\5\2\2\5\30\31\7\6\2"+
		"\2\31\32\5\2\2\2\32\33\7\7\2\2\33\34\7\5\2\2\34\35\7\6\2\2\35\36\7\34"+
		"\2\2\36\37\7\b\2\2\37 \7\34\2\2 !\7\7\2\2!#\3\2\2\2\"\b\3\2\2\2\"\20\3"+
		"\2\2\2\"\26\3\2\2\2\"\30\3\2\2\2#/\3\2\2\2$%\f\7\2\2%&\7\26\2\2&.\5\2"+
		"\2\b\'(\f\6\2\2()\7\27\2\2).\5\2\2\7*+\f\4\2\2+,\7\32\2\2,.\5\2\2\5-$"+
		"\3\2\2\2-\'\3\2\2\2-*\3\2\2\2.\61\3\2\2\2/-\3\2\2\2/\60\3\2\2\2\60\3\3"+
		"\2\2\2\61/\3\2\2\2\62\63\b\3\1\2\63\64\t\2\2\2\64\65\5\4\3\2\65\66\7\4"+
		"\2\2\66?\3\2\2\2\678\7\24\2\289\7\3\2\29:\5\4\3\2:;\7\4\2\2;?\3\2\2\2"+
		"<?\7\25\2\2=?\7\24\2\2>\62\3\2\2\2>\67\3\2\2\2><\3\2\2\2>=\3\2\2\2?K\3"+
		"\2\2\2@A\f\b\2\2AB\7\5\2\2BJ\5\4\3\bCD\f\6\2\2DE\t\3\2\2EJ\5\4\3\7FG\f"+
		"\5\2\2GH\t\4\2\2HJ\5\4\3\6I@\3\2\2\2IC\3\2\2\2IF\3\2\2\2JM\3\2\2\2KI\3"+
		"\2\2\2KL\3\2\2\2L\5\3\2\2\2MK\3\2\2\2NO\5\4\3\2OP\t\5\2\2PQ\5\4\3\2QU"+
		"\3\2\2\2RU\7\23\2\2SU\7\33\2\2TN\3\2\2\2TR\3\2\2\2TS\3\2\2\2U\7\3\2\2"+
		"\2\13\20\23\"-/>IKT";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}