package in.ac.iitd.db362.index.bplustree;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.List;

// B+ Tree node structure.
// Note: Do not modify this class
public class Node<K, V> {

    protected static final Logger logger = LogManager.getLogger();

    boolean isLeaf;
    List<K> keys;
    List<V> values; // should be null/empty for non-leaf nodes!
    List<Node<K, V>> children;
    Node<K, V> next; // For leaf node linking

    /**
     * The function returns a child node of this node.
     * @param offset
     * @return return child node
     */
    Node<K,V> getChild(int offset) {
        logger.trace("Called getChild( " + offset + " )");
        assert !isLeaf;
        return this.children.get(offset);
    }

    /**
     * The function returns the next node pointed by a leaf node.
     * @return next leaf node
     */
    Node<K,V> getNext() {
        logger.trace("Called getNext()");
        logger.trace("Last <K,V>: " + "<" + (K)this.keys.get(keys.size()-1) + "," + (V)this.values.get(keys.size()-1) + ">");
        assert isLeaf;
        return this.next;
    }
}