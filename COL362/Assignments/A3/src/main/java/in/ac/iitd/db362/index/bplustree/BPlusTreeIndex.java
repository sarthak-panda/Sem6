package in.ac.iitd.db362.index.bplustree;

import in.ac.iitd.db362.index.Index;
import in.ac.iitd.db362.parser.QueryNode;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;

/**
 * Starter code for BPlusTree Implementation
 * @param <T> The type of the key.
 */
public class BPlusTreeIndex<T extends Comparable<T>> implements Index<T> {

    protected static final Logger logger = LogManager.getLogger();

    // Note: Do not rename this variable; the test cases will set this when testing. You can however initialize it with a
    // different value for testing your code.
    public static int ORDER = 10;

    // The attribute being indexed
    private String attribute;

    // Our Values are all integers (rowIds)
    private Node<T, Integer> root;
    private final int order; // Maximum children per node

    /** Constructor to initialize the B+ Tree with a given order */
    public BPlusTreeIndex(String attribute) {
        this.attribute = attribute;
        this.order = ORDER;
        this.root = new Node<>();
        this.root.isLeaf = true;
    }

    @Override
    public List<Integer> evaluate(QueryNode node) {
        logger.info("Evaluating predicate using B+ Tree index on attribute " + attribute + " for operator " + node.operator);
        //TODO: Implement me!
        return null;
    }

    private void insertIntoLeaf(Node<T, Integer> leaf, T key, int rowId) {
        int pos = 0;
        while (pos < leaf.keys.size() && key.compareTo(leaf.keys.get(pos)) > 0) {
            pos++;
        }
        leaf.keys.add(pos, key);
        leaf.values.add(pos, rowId);
    }

    private Node<T, Integer> findParent(Node<T, Integer> current, Node<T, Integer> child) {
        if (current.isLeaf) {
            return null;
        }
        for (int i = 0; i < current.children.size(); i++) {
            if (current.children.get(i) == child) {
                return current;
            }
            Node<T, Integer> parent = findParent(current.children.get(i), child);
            if (parent != null) {
                return parent;
            }
        }
        return null;
    }

    private void splitInternalNode(Node<T, Integer> node) {
        int splitIndex = node.keys.size() / 2;
        T promoteKey = node.keys.get(splitIndex);

        List<T> leftKeys = new ArrayList<>(node.keys.subList(0, splitIndex));
        List<T> rightKeys = new ArrayList<>(node.keys.subList(splitIndex + 1, node.keys.size()));

        List<Node<T, Integer>> leftChildren = new ArrayList<>(node.children.subList(0, splitIndex + 1));
        List<Node<T, Integer>> rightChildren = new ArrayList<>(node.children.subList(splitIndex + 1, node.children.size()));

        Node<T, Integer> rightNode = new Node<>();
        rightNode.isLeaf = false;
        rightNode.keys = rightKeys;
        rightNode.children = rightChildren;

        node.keys = leftKeys;
        node.children = leftChildren;

        Node<T, Integer> parent = findParent(root, node);
        if (parent == null) {
            parent = new Node<>();
            parent.isLeaf = false;
            parent.keys = new ArrayList<>();
            parent.children = new ArrayList<>();
            parent.keys.add(promoteKey);
            parent.children.add(node);
            parent.children.add(rightNode);
            root = parent;
        } else {
            int pos = 0;
            while (pos < parent.keys.size() && promoteKey.compareTo(parent.keys.get(pos)) >= 0) {
                pos++;
            }
            parent.keys.add(pos, promoteKey);
            parent.children.add(pos + 1, rightNode);

            if (parent.keys.size() > order - 1) {
                splitInternalNode(parent);
            }
        }
    }    

    private void splitLeafNode(Node<T, Integer> leaf) {
        int splitIndex = leaf.keys.size() / 2;
        List<T> rightKeys = new ArrayList<>(leaf.keys.subList(splitIndex, leaf.keys.size()));
        List<Integer> rightValues = new ArrayList<>(leaf.values.subList(splitIndex, leaf.values.size()));

        Node<T, Integer> rightNode = new Node<>();
        rightNode.isLeaf = true;
        rightNode.keys = rightKeys;
        rightNode.values = rightValues;
        rightNode.next = leaf.next;
        leaf.next = rightNode;

        leaf.keys = new ArrayList<>(leaf.keys.subList(0, splitIndex));
        leaf.values = new ArrayList<>(leaf.values.subList(0, splitIndex));

        T promoteKey = rightKeys.get(0);
        Node<T, Integer> parent = findParent(root, leaf);

        if (parent == null) {
            parent = new Node<>();
            parent.isLeaf = false;
            parent.keys = new ArrayList<>();
            parent.children = new ArrayList<>();
            parent.keys.add(promoteKey);
            parent.children.add(leaf);
            parent.children.add(rightNode);
            root = parent;
        } else {
            int pos = 0;
            while (pos < parent.keys.size() && promoteKey.compareTo(parent.keys.get(pos)) >= 0) {
                pos++;
            }
            parent.keys.add(pos, promoteKey);
            parent.children.add(pos + 1, rightNode);

            if (parent.keys.size() > order - 1) {
                splitInternalNode(parent);
            }
        }
    }


    @Override
    public void insert(T key, int rowId) {
        //TODO: Implement me!
        Node<T, Integer> current = root;
        while (!current.isLeaf) {
            int i = 0;
            while (i < current.keys.size() && key.compareTo(current.keys.get(i)) > 0) {
                i++;
            }
            current = current.getChild(i);
        }
        insertIntoLeaf(current, key, rowId);
        if (current.keys.size() > order - 1) {
            splitLeafNode(current);
        }
    }

    @Override
    public boolean delete(T key) {
        //TODO: Bonus
        return false;
    }

    @Override
    public List<Integer> search(T key) {
        //TODO: Implement me!
        //Note: When searching for a key, use Node's getChild() and getNext() methods. Some test cases may fail otherwise!
        //let us start with naive implementation
        List<Integer> rowIds = new ArrayList<>();
        Node<T, Integer> current = root;
        if (current == null) {
            return rowIds;
        }
        while (!current.isLeaf){
            int i = 0;
            while (i < current.keys.size() && key.compareTo(current.keys.get(i)) > 0){
                i++;
            }
            current = current.getChild(i);
        }
        while (current != null){
            int i = 0;
            while (i < current.keys.size()){
                int cmp = key.compareTo(current.keys.get(i));
                if (cmp == 0){
                    rowIds.add(current.values.get(i));
                }
                else if (cmp < 0){
                    return rowIds;
                }
                i++;
            }
            current = current.getNext();
        }
        return rowIds;
    }

    /**
     * Function that evaluates a range query and returns a list of rowIds.
     * e.g., 50 < x <=75, then function can be called as rangeQuery(50, false, 75, true)
     * @param startKey
     * @param startInclusive
     * @param endKey
     * @param endInclusive
     * @return all rowIds that satisfy the range predicate
     */
    List<Integer> rangeQuery(T startKey, boolean startInclusive, T endKey, boolean endInclusive) {
        //TODO: Implement me!
        //Note: When searching, use Node's getChild() and getNext() methods. Some test cases may fail otherwise!
        List<Integer> result = new ArrayList<>();
        if (root == null || root.keys.isEmpty()) return result;
        Node<T, Integer> current = root;
        while (!current.isLeaf) {
            int i = 0;
            while (i < current.keys.size() && startKey.compareTo(current.keys.get(i)) > 0) {
                i++;
            }
            current = current.children.get(i);
        }
        boolean done = false;
        while (current != null && !done) {
            for (int i = 0; i < current.keys.size(); i++) {
                T currentKey = current.keys.get(i);
                boolean meetsStartCondition = startInclusive ? (currentKey.compareTo(startKey) >= 0) : (currentKey.compareTo(startKey) > 0);
                boolean meetsEndCondition = endInclusive ? (currentKey.compareTo(endKey) <= 0) : (currentKey.compareTo(endKey) < 0);
                if (meetsStartCondition && meetsEndCondition) {
                    result.add(current.values.get(i));
                } else if (!meetsEndCondition) {
                    done = true;
                    break;
                }
            }
            current = current.getNext();
        }
        return result;
    }

    /**
     * Traverse leaf nodes and collect all keys in sorted order
     * @return all Keys
     */
    public List<T> getAllKeys() {
        // TODO: Implement me!
        List<T> keys = new ArrayList<>();
        if (root == null || root.keys.isEmpty()) return keys;
        Node<T, Integer> current = root;
        while (!current.isLeaf){
            current = current.getChild(0);
        }
        while (current != null){
            keys.addAll(current.keys);
            current = current.getNext();
        }
        return keys;
    }

    /**
     * Compute tree height by traversing from root to leaf
     * @return Height of the b+ tree
     */
    public int getHeight() {
        // TODO: Implement me!
        int height = 0;
        Node<T, Integer> current = root;
        while (!current.isLeaf){
            height++;
            current = current.getChild(0);
        }
        return height;
    }

    /**
     * Funtion that returns the order of the BPlusTree
     * Note: Do not remove this function!
     * @return
     */
    public int getOrder() {
        return order;
    }


    @Override
    public String prettyName() {
        return "B+Tree Index";
    }
}
