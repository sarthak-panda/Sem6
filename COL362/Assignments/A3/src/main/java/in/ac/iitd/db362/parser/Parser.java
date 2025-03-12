package in.ac.iitd.db362.parser;


import java.util.List;

// Recursive descent parsing following the grammar:
//
// Expression -> Term (OR Term)*
// Term       -> Factor (AND Factor)*
// Factor     -> NOT Factor | '(' Expression ')' | Predicate
// Predicate  -> RangePredicate | SimplePredicate
//
// RangePredicate  : NUMBER '<' IDENTIFIER '<' NUMBER
// SimplePredicate : IDENTIFIER ( '=' | '<' | '>' ) Literal
// Literal         : NUMBER | STRING | IDENTIFIER
public class Parser {

    List<Token> tokens;
    int pos = 0;

    public Parser(List<Token> tokens) {
        this.tokens = tokens;
    }

    Token peek() {
        return tokens.get(pos);
    }

    Token consume(TokenType type) {
        Token token = peek();
        if (token.type != type) {
            throw new RuntimeException("Expected token " + type + " but found " + token.type);
        }
        pos++;
        return token;
    }

    boolean match(TokenType type) {
        if (peek().type == type) {
            pos++;
            return true;
        }
        return false;
    }

    // Expression -> Term (OR Term)*
    public QueryNode parseExpression() {
        QueryNode node = parseTerm();
        while (peek().type == TokenType.OR) {
            consume(TokenType.OR);
            QueryNode right = parseTerm();
            node = new QueryNode(Operator.OR, node, right);
        }
        return node;
    }

    // Term -> Factor (AND Factor)*
    QueryNode parseTerm() {
        QueryNode node = parseFactor();
        while (peek().type == TokenType.AND) {
            consume(TokenType.AND);
            QueryNode right = parseFactor();
            node = new QueryNode(Operator.AND, node, right);
        }
        return node;
    }

    // Factor -> NOT Factor | '(' Expression ')' | Predicate
    QueryNode parseFactor() {
        if (peek().type == TokenType.NOT) {
            consume(TokenType.NOT);
            QueryNode node = parseFactor();
            return new QueryNode(Operator.NOT, node);
        } else if (peek().type == TokenType.LPAREN) {
            consume(TokenType.LPAREN);
            QueryNode node = parseExpression();
            consume(TokenType.RPAREN);
            return node;
        } else {
            return parsePredicate();
        }
    }

    // Predicate -> RangePredicate | SimplePredicate
    QueryNode parsePredicate() {
        // Check for range predicate: NUMBER '<' IDENTIFIER '<' NUMBER
        if (peek().type == TokenType.NUMBER) {
            Token numToken = consume(TokenType.NUMBER);
            if (peek().type == TokenType.LT) {
                consume(TokenType.LT);
                if (peek().type == TokenType.IDENTIFIER) {
                    Token attrToken = consume(TokenType.IDENTIFIER);
                    if (peek().type == TokenType.LT) {
                        consume(TokenType.LT);
                        if (peek().type == TokenType.NUMBER) {
                            Token numToken2 = consume(TokenType.NUMBER);
                            QueryNode node = new QueryNode(Operator.RANGE, attrToken.text, numToken.text);
                            node.secondValue = numToken2.text;
                            return node;
                        } else {
                            throw new RuntimeException("Expected NUMBER in range predicate");
                        }
                    } else {
                        throw new RuntimeException("Expected '<' after attribute in range predicate");
                    }
                } else {
                    throw new RuntimeException("Expected IDENTIFIER after '<'");
                }
            } else {
                throw new RuntimeException("Unexpected token after NUMBER. For range predicates, use the form: NUMBER < attribute < NUMBER");
            }
        } else if (peek().type == TokenType.IDENTIFIER) {
            // Simple predicate: IDENTIFIER ( '=' | '<' | '>' ) Literal
            Token attrToken = consume(TokenType.IDENTIFIER);
            Token opToken = peek();
            Operator op;
            if (opToken.type == TokenType.EQ) {
                op = Operator.EQUALS;
                consume(TokenType.EQ);
            } else if (opToken.type == TokenType.LT) {
                op = Operator.LT;
                consume(TokenType.LT);
            } else if (opToken.type == TokenType.GT) {
                op = Operator.GT;
                consume(TokenType.GT);
            } else {
                throw new RuntimeException("Expected a relational operator after attribute");
            }
            // Literal: NUMBER, STRING, or IDENTIFIER
            Token literalToken = peek();
            if (literalToken.type == TokenType.NUMBER) {
                consume(TokenType.NUMBER);
            } else if (literalToken.type == TokenType.STRING) {
                consume(TokenType.STRING);
            } else if (literalToken.type == TokenType.IDENTIFIER) {
                consume(TokenType.IDENTIFIER);
            } else {
                throw new RuntimeException("Expected a literal after relational operator");
            }
            return new QueryNode(op, attrToken.text, literalToken.text);
        } else {
            throw new RuntimeException("Unexpected token in predicate: " + peek());
        }
    }

    public static QueryNode parse(String query) {
        List<Token> tokens = Tokenizer.tokenize(query);
        Parser parser = new Parser(tokens);
        return parser.parseExpression();
    }


    // Utility method to print the parse tree.
    public static void printParseTree(QueryNode node, int depth) {
        if (node == null) return;
        String indent = "  ".repeat(depth);
        if (node.operator == Operator.RANGE) {
            System.out.println(indent + node.operator + " on " + node.attribute + ": " + node.value + " and " + node.secondValue);
        } else if (node.operator == Operator.AND || node.operator == Operator.OR || node.operator == Operator.NOT) {
            System.out.println(indent + node.operator);
            printParseTree(node.left, depth + 1);
            if (node.right != null) {
                printParseTree(node.right, depth + 1);
            }
        } else { // simple predicate
            System.out.println(indent + node.operator + " " + node.attribute + " " + node.value);
        }
    }
}
