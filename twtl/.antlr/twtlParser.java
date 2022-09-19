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
		T__9=10, T__10=11, T__11=12, T__12=13, T__13=14, T__14=15, T__15=16, AND=17, 
		OR=18, NOT=19, HOLD=20, CONCAT=21, RATIONAL=22, TRUE=23, FALSE=24, VARIABLE=25, 
		LINECMT=26, WS=27;
	public static final int
		RULE_formula = 0, RULE_booleanExpr = 1, RULE_expr = 2;
	private static String[] makeRuleNames() {
		return new String[] {
			"formula", "booleanExpr", "expr"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, "'('", "')'", "'^'", "'['", "']'", "','", "'<'", "'<='", "'='", 
			"'>='", "'>'", "'-('", "'*'", "'/'", "'+'", "'-'", null, null, null, 
			"'H'", "'.'"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, null, null, null, null, null, null, null, null, null, null, null, 
			null, null, null, null, null, "AND", "OR", "NOT", "HOLD", "CONCAT", "RATIONAL", 
			"TRUE", "FALSE", "VARIABLE", "LINECMT", "WS"
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
		public Token op;
		public FormulaContext child;
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
		public List<TerminalNode> RATIONAL() { return getTokens(twtlParser.RATIONAL); }
		public TerminalNode RATIONAL(int i) {
			return getToken(twtlParser.RATIONAL, i);
		}
		public TerminalNode NOT() { return getToken(twtlParser.NOT, 0); }
		public TerminalNode OR() { return getToken(twtlParser.OR, 0); }
		public TerminalNode AND() { return getToken(twtlParser.AND, 0); }
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
				((FormulaContext)_localctx).op = match(T__0);
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
					((FormulaContext)_localctx).duration = match(RATIONAL);
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
				((FormulaContext)_localctx).child = formula(2);
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
				((FormulaContext)_localctx).low = match(RATIONAL);
				setState(28);
				match(T__5);
				setState(29);
				((FormulaContext)_localctx).high = match(RATIONAL);
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
						if (!(precpred(_ctx, 6))) throw new FailedPredicateException(this, "precpred(_ctx, 6)");
						setState(35);
						((FormulaContext)_localctx).op = match(OR);
						setState(36);
						((FormulaContext)_localctx).right = formula(7);
						}
						break;
					case 2:
						{
						_localctx = new FormulaContext(_parentctx, _parentState);
						_localctx.left = _prevctx;
						_localctx.left = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_formula);
						setState(37);
						if (!(precpred(_ctx, 5))) throw new FailedPredicateException(this, "precpred(_ctx, 5)");
						setState(38);
						((FormulaContext)_localctx).op = match(AND);
						setState(39);
						((FormulaContext)_localctx).right = formula(6);
						}
						break;
					case 3:
						{
						_localctx = new FormulaContext(_parentctx, _parentState);
						_localctx.left = _prevctx;
						_localctx.left = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_formula);
						setState(40);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(41);
						((FormulaContext)_localctx).op = match(CONCAT);
						setState(42);
						((FormulaContext)_localctx).right = formula(5);
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

	public static class BooleanExprContext extends ParserRuleContext {
		public Token op;
		public ExprContext left;
		public ExprContext right;
		public TerminalNode FALSE() { return getToken(twtlParser.FALSE, 0); }
		public TerminalNode TRUE() { return getToken(twtlParser.TRUE, 0); }
		public TerminalNode VARIABLE() { return getToken(twtlParser.VARIABLE, 0); }
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public BooleanExprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_booleanExpr; }
	}

	public final BooleanExprContext booleanExpr() throws RecognitionException {
		BooleanExprContext _localctx = new BooleanExprContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_booleanExpr);
		int _la;
		try {
			setState(55);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,5,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(48);
				((BooleanExprContext)_localctx).op = match(FALSE);
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(49);
				((BooleanExprContext)_localctx).op = match(TRUE);
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(50);
				((BooleanExprContext)_localctx).op = match(VARIABLE);
				}
				break;
			case 4:
				enterOuterAlt(_localctx, 4);
				{
				setState(51);
				((BooleanExprContext)_localctx).left = expr(0);
				setState(52);
				((BooleanExprContext)_localctx).op = _input.LT(1);
				_la = _input.LA(1);
				if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << T__6) | (1L << T__7) | (1L << T__8) | (1L << T__9) | (1L << T__10))) != 0)) ) {
					((BooleanExprContext)_localctx).op = (Token)_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(53);
				((BooleanExprContext)_localctx).right = expr(0);
				}
				break;
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
		int _startState = 4;
		enterRecursionRule(_localctx, 4, RULE_expr, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(69);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,6,_ctx) ) {
			case 1:
				{
				setState(58);
				_la = _input.LA(1);
				if ( !(_la==T__0 || _la==T__11) ) {
				_errHandler.recoverInline(this);
				}
				else {
					if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
					_errHandler.reportMatch(this);
					consume();
				}
				setState(59);
				expr(0);
				setState(60);
				match(T__1);
				}
				break;
			case 2:
				{
				setState(62);
				match(VARIABLE);
				setState(63);
				match(T__0);
				setState(64);
				expr(0);
				setState(65);
				match(T__1);
				}
				break;
			case 3:
				{
				setState(67);
				match(RATIONAL);
				}
				break;
			case 4:
				{
				setState(68);
				match(VARIABLE);
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(82);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,8,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(80);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,7,_ctx) ) {
					case 1:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(71);
						if (!(precpred(_ctx, 6))) throw new FailedPredicateException(this, "precpred(_ctx, 6)");
						setState(72);
						match(T__2);
						setState(73);
						expr(6);
						}
						break;
					case 2:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(74);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(75);
						_la = _input.LA(1);
						if ( !(_la==T__12 || _la==T__13) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(76);
						expr(5);
						}
						break;
					case 3:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(77);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(78);
						_la = _input.LA(1);
						if ( !(_la==T__14 || _la==T__15) ) {
						_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(79);
						expr(4);
						}
						break;
					}
					} 
				}
				setState(84);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,8,_ctx);
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

	public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
		switch (ruleIndex) {
		case 0:
			return formula_sempred((FormulaContext)_localctx, predIndex);
		case 2:
			return expr_sempred((ExprContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean formula_sempred(FormulaContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0:
			return precpred(_ctx, 6);
		case 1:
			return precpred(_ctx, 5);
		case 2:
			return precpred(_ctx, 4);
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
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\35X\4\2\t\2\4\3\t"+
		"\3\4\4\t\4\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\5\2\21\n\2\3\2\5\2\24\n\2\3"+
		"\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\3\2\5\2#\n\2\3\2\3\2\3"+
		"\2\3\2\3\2\3\2\3\2\3\2\3\2\7\2.\n\2\f\2\16\2\61\13\2\3\3\3\3\3\3\3\3\3"+
		"\3\3\3\3\3\5\3:\n\3\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\5"+
		"\4H\n\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\7\4S\n\4\f\4\16\4V\13\4\3"+
		"\4\2\4\2\6\5\2\4\6\2\6\3\2\t\r\4\2\3\3\16\16\3\2\17\20\3\2\21\22\2e\2"+
		"\"\3\2\2\2\49\3\2\2\2\6G\3\2\2\2\b\t\b\2\1\2\t\n\7\3\2\2\n\13\5\2\2\2"+
		"\13\f\7\4\2\2\f#\3\2\2\2\r\16\7\26\2\2\16\17\7\5\2\2\17\21\7\30\2\2\20"+
		"\r\3\2\2\2\20\21\3\2\2\2\21\23\3\2\2\2\22\24\7\25\2\2\23\22\3\2\2\2\23"+
		"\24\3\2\2\2\24\25\3\2\2\2\25#\5\4\3\2\26\27\7\25\2\2\27#\5\2\2\4\30\31"+
		"\7\6\2\2\31\32\5\2\2\2\32\33\7\7\2\2\33\34\7\5\2\2\34\35\7\6\2\2\35\36"+
		"\7\30\2\2\36\37\7\b\2\2\37 \7\30\2\2 !\7\7\2\2!#\3\2\2\2\"\b\3\2\2\2\""+
		"\20\3\2\2\2\"\26\3\2\2\2\"\30\3\2\2\2#/\3\2\2\2$%\f\b\2\2%&\7\24\2\2&"+
		".\5\2\2\t\'(\f\7\2\2()\7\23\2\2).\5\2\2\b*+\f\6\2\2+,\7\27\2\2,.\5\2\2"+
		"\7-$\3\2\2\2-\'\3\2\2\2-*\3\2\2\2.\61\3\2\2\2/-\3\2\2\2/\60\3\2\2\2\60"+
		"\3\3\2\2\2\61/\3\2\2\2\62:\7\32\2\2\63:\7\31\2\2\64:\7\33\2\2\65\66\5"+
		"\6\4\2\66\67\t\2\2\2\678\5\6\4\28:\3\2\2\29\62\3\2\2\29\63\3\2\2\29\64"+
		"\3\2\2\29\65\3\2\2\2:\5\3\2\2\2;<\b\4\1\2<=\t\3\2\2=>\5\6\4\2>?\7\4\2"+
		"\2?H\3\2\2\2@A\7\33\2\2AB\7\3\2\2BC\5\6\4\2CD\7\4\2\2DH\3\2\2\2EH\7\30"+
		"\2\2FH\7\33\2\2G;\3\2\2\2G@\3\2\2\2GE\3\2\2\2GF\3\2\2\2HT\3\2\2\2IJ\f"+
		"\b\2\2JK\7\5\2\2KS\5\6\4\bLM\f\6\2\2MN\t\4\2\2NS\5\6\4\7OP\f\5\2\2PQ\t"+
		"\5\2\2QS\5\6\4\6RI\3\2\2\2RL\3\2\2\2RO\3\2\2\2SV\3\2\2\2TR\3\2\2\2TU\3"+
		"\2\2\2U\7\3\2\2\2VT\3\2\2\2\13\20\23\"-/9GRT";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}