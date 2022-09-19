

grammar Expr; // The name of the folder must be the same 

/* the grammar name and file must be the same */

@header { 
    package antlr;
     } // Every java class will be put into ANTLR package 

/* Defineing the production rules */
// Tokens must be capitals or single quote channels
//

// start variable (look at the grammar definition)
prog: ( decl | expr)+ EOF // +: means that there must be at least one of them. EOF: means that the error handler should be on
    ;

decl: ID ':' INT_TYPE '=' NUM // declear Integer value with number
    ;

/* ANTLR resolve ambiguities in favor of the alternative given first. */
expr: expr '*' expr  // Will have higher precedance ()
    | expr '+' expr
    | ID  
    | NUM
    ;

/*Tokens */

ID : [a-z][a-zA-Z0-9_]*; // *:means manny of them. Identefiers
NUM : '0' | '-'?[1-9][0-9]* ; // ?: means there or not (meanning that the number could be signed or not)
INT_TYPE : 'INT'; // See regular expressions 
COMMENT : '--' ~[\r\n]* -> skip; // anything except \r or \n --*: as many as possible-- is skipped from to be compled (comment)    
WS : [\t\n]+ -> skip;