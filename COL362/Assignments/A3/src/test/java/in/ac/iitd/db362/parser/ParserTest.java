package in.ac.iitd.db362.parser;

import in.ac.iitd.db362.parser.*;
import org.junit.jupiter.api.Test;


import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

public class ParserTest {

    @Test
    public void testSimplePredicate() {
        String query = "age < 50";
        QueryNode node = Parser.parse(query);
        //Parser.printParseTree(node, 0);

        // Expect a simple predicate: operator LT, attribute "age", value "50"
        assertNotNull(node, "The parse tree should not be null.");
        assertEquals(Operator.LT, node.operator, "Operator should be LT.");
        assertEquals("age", node.attribute, "Attribute should be 'age'.");
        assertEquals("50", node.value, "Value should be '50'.");
        assertNull(node.left, "Left child should be null for a simple predicate.");
        assertNull(node.right, "Right child should be null for a simple predicate.");
    }

    @Test
    public void testRangePredicate() {
        String query = "5000 < salary < 10000";
        QueryNode node = Parser.parse(query);
        //Parser.printParseTree(node, 0);

        // Expect a range predicate: operator RANGE, attribute "salary", value "5000", secondValue "10000"
        assertNotNull(node, "The parse tree should not be null.");
        assertEquals(Operator.RANGE, node.operator, "Operator should be RANGE.");
        assertEquals("salary", node.attribute, "Attribute should be 'salary'.");
        assertEquals("5000", node.value, "Lower bound should be '5000'.");
        assertEquals("10000", node.secondValue, "Upper bound should be '10000'.");
        assertNull(node.left, "Left child should be null for a predicate node.");
        assertNull(node.right, "Right child should be null for a predicate node.");
    }

    @Test
    public void testBooleanCombination() {
        String query = "(salary > 10000 AND department = HR) OR salary < 10000";
        QueryNode node = Parser.parse(query);
        //Parser.printParseTree(node, 0);

        // Root should be an OR operator.
        assertNotNull(node, "The parse tree should not be null.");
        assertEquals(Operator.OR, node.operator, "Root operator should be OR.");

        // Left child should be an AND operator.
       QueryNode left = node.left;
        assertNotNull(left, "Left child of OR should not be null.");
        assertEquals(Operator.AND, left.operator, "Left child operator should be AND.");

        // In the left subtree, check the simple predicates.
        // Left of AND: salary > 10000
        QueryNode leftLeft = left.left;
        assertNotNull(leftLeft, "Left child of AND should not be null.");
        assertEquals(Operator.GT, leftLeft.operator, "Expected operator GT.");
        assertEquals("salary", leftLeft.attribute, "Attribute should be 'salary'.");
        assertEquals("10000", leftLeft.value, "Value should be '10000'.");

        // Right of AND: department = HR
        QueryNode leftRight = left.right;
        assertNotNull(leftRight, "Right child of AND should not be null.");
        assertEquals(Operator.EQUALS, leftRight.operator, "Expected operator EQUALS.");
        assertEquals("department", leftRight.attribute, "Attribute should be 'department'.");
        assertEquals("HR", leftRight.value, "Value should be 'HR'.");

        // Right child of OR: salary < 10000
        QueryNode right = node.right;
        assertNotNull(right, "Right child of OR should not be null.");
        assertEquals(Operator.LT, right.operator, "Expected operator LT.");
        assertEquals("salary", right.attribute, "Attribute should be 'salary'.");
        assertEquals("10000", right.value, "Value should be '10000'.");
    }

    @Test
    public void testNotPredicate() {
        String query = "NOT (department = HR)";
        QueryNode node = Parser.parse(query);
        //Parser.printParseTree(node, 0);

        // Root should be a NOT operator.
        assertNotNull(node, "The parse tree should not be null.");
        assertEquals(Operator.NOT, node.operator, "Root operator should be NOT.");

        // The child of the NOT operator should be the predicate: department = HR.
        QueryNode child = node.left;
        assertNotNull(child, "Child of NOT should not be null.");
        assertEquals(Operator.EQUALS, child.operator, "Expected operator EQUALS.");
        assertEquals("department", child.attribute, "Attribute should be 'department'.");
        assertEquals("HR", child.value, "Value should be 'HR'.");
    }

    @Test
    public void testComplexPredicate() {
        String query = "5000 < salary < 10000 AND NOT department = HR OR deparment = BO";
        QueryNode node = Parser.parse(query);
        //Parser.printParseTree(node, 0);
    }
}

