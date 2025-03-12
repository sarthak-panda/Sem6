package in.ac.iitd.db362.catalog;


import java.util.*;

import in.ac.iitd.db362.index.Index;
import in.ac.iitd.db362.index.bplustree.BPlusTreeIndex;
import in.ac.iitd.db362.index.hashindex.ExtendibleHashing;
import in.ac.iitd.db362.index.BitmapIndex;
import in.ac.iitd.db362.parser.Operator;

/**
 * A simple catalog that stores which indexes are available for which attributes
 * Catalog is implemented as a singleton class. For now, we only focus on processing one CSV file at a time!
 * Note: DO NOT CHANGE ANYTHING IN THIS FILE
 */
public class Catalog {

    private Catalog(){};

    private static Catalog instance = null;

    public static synchronized Catalog getInstance() {
        if (instance == null) {
            instance = new Catalog();
        }
        return instance;
    }

    // Map: attribute -> list of available indexes.
    private Map<String, List<Index>> catalogMap = new HashMap<>();

    /**
     * Registers an index for the given attribute.
     */
    public void addIndex(String attribute, Index index) {
        catalogMap.computeIfAbsent(attribute, k -> new ArrayList<>()).add(index);
    }

    /**
     * Retrieves all indexes available for the given attribute.
     */
    public List<Index> getIndexes(String attribute) {
        return catalogMap.getOrDefault(attribute, Collections.emptyList());
    }

    /**
     * Get an appropriate index for the given attribute and operator based on a simple heuristic:
     *
     * If the operator is LT, GT, or RANGE, then use a B+ tree index.
     * If the operator is EQUALS, then prefer a Bitmap index; if not available, then a Hash index;
     * if that isn’t available, then a B+ tree index.
     *
     */
    public Index getIndex(String attribute, Operator operator) {
        List<Index> indexes = getIndexes(attribute);
        if (operator == Operator.LT || operator == Operator.GT || operator == Operator.RANGE) {
            // Look for a BPlusTreeIndex.
            for (Index idx : indexes) {
                if (idx instanceof BPlusTreeIndex) {
                    return idx;
                }
            }
        } else if (operator == Operator.EQUALS) {
            // Prefer a BitmapIndex.
            for (Index idx : indexes) {
                if (idx instanceof BitmapIndex) {
                    return idx;
                }
            }
            // Then try a HashIndex.
            for (Index idx : indexes) {
                if (idx instanceof ExtendibleHashing) {
                    return idx;
                }
            }
            // Fall back to a BPlusTreeIndex if available.
            for (Index idx : indexes) {
                if (idx instanceof BPlusTreeIndex) {
                    return idx;
                }
            }
        }
        return null; // No appropriate index found.
    }
}

