package in.ac.iitd.db362.parser;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

// The tokenizer uses a regex with named capturing groups to extract tokens.
public class Tokenizer {

    // Regex for tokens. Note that the order matters.
    private static Pattern tokenPattern = Pattern.compile(
            "\\s*(?:(?<LPAREN>\\()|" +
                    "(?<RPAREN>\\))|" +
                    "(?<AND>AND)|" +
                    "(?<OR>OR)|" +
                    "(?<NOT>NOT)|" +
                    "(?<EQ>=)|" +
                    "(?<LT><)|" +
                    "(?<GT>>)|" +
                    "(?<NUMBER>\\d+(?:\\.\\d+)?)|" +
                    "(?<STRING>\"[^\"]*\")|" +
                    "(?<IDENTIFIER>[a-zA-Z_][a-zA-Z0-9_]*))"
    );

    public static List<Token> tokenize(String input) {
        List<Token> tokens = new ArrayList<>();
        Matcher matcher = tokenPattern.matcher(input);
        int pos = 0;
        while (pos < input.length()) {
            if (matcher.find(pos) && matcher.start() == pos) {
                if (matcher.group("LPAREN") != null) {
                    tokens.add(new Token(TokenType.LPAREN, "("));
                } else if (matcher.group("RPAREN") != null) {
                    tokens.add(new Token(TokenType.RPAREN, ")"));
                } else if (matcher.group("AND") != null) {
                    tokens.add(new Token(TokenType.AND, "AND"));
                } else if (matcher.group("OR") != null) {
                    tokens.add(new Token(TokenType.OR, "OR"));
                } else if (matcher.group("NOT") != null) {
                    tokens.add(new Token(TokenType.NOT, "NOT"));
                } else if (matcher.group("EQ") != null) {
                    tokens.add(new Token(TokenType.EQ, "="));
                } else if (matcher.group("LT") != null) {
                    tokens.add(new Token(TokenType.LT, "<"));
                } else if (matcher.group("GT") != null) {
                    tokens.add(new Token(TokenType.GT, ">"));
                } else if (matcher.group("NUMBER") != null) {
                    tokens.add(new Token(TokenType.NUMBER, matcher.group("NUMBER")));
                } else if (matcher.group("STRING") != null) {
                    tokens.add(new Token(TokenType.STRING, matcher.group("STRING")));
                } else if (matcher.group("IDENTIFIER") != null) {
                    tokens.add(new Token(TokenType.IDENTIFIER, matcher.group("IDENTIFIER")));
                }
                pos = matcher.end();
            } else {
                throw new RuntimeException("Unexpected character at position " + pos);
            }
        }
        tokens.add(new Token(TokenType.EOF, ""));
        return tokens;
    }

    // Token class: each token has a type and textual value.

    // Token types for the query language

}
