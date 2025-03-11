package in.ac.iitd.db362.parser;

public class Token {
    TokenType type;
    String text;

    Token(TokenType type, String text) {
        this.type = type;
        this.text = text;
    }

    public String toString() {
        return type + "(" + text + ")";
    }
}