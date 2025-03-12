package in.ac.iitd.db362.parser;

// Parse tree node.
public class QueryNode {
    public Operator operator;
    public String attribute;  // for predicates: the column name
    public String value;      // for simple predicates: literal value;
    // for range predicates: lower bound
    public String secondValue; // for range predicates: upper bound
    public QueryNode left;
    public QueryNode right;

    // For simple predicates
    QueryNode(Operator operator, String attribute, String value) {
        this.operator = operator;
        this.attribute = attribute;
        this.value = value;
    }

    // For binary boolean operators (AND, OR)
    QueryNode(Operator operator, QueryNode left, QueryNode right) {
        this.operator = operator;
        this.left = left;
        this.right = right;
    }

    // For unary boolean operator (NOT)
    QueryNode(Operator operator, QueryNode node) {
        this.operator = operator;
        this.left = node;
    }
}
