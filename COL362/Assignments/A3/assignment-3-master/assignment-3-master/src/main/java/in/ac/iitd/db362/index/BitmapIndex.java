package in.ac.iitd.db362.index;

import in.ac.iitd.db362.parser.QueryNode;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;


import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Starter code for a BitMap Index
 * Bitmap indexes are typically used for equality queries and rely on a BitSet.
 *
 * @param <T> The type of the key.
 */
public class BitmapIndex<T> implements Index<T> {

    protected static final Logger logger = LogManager.getLogger();

    private String attribute;
    private int maxRowId;

    private Map<T, int[]> bitmaps;

    /**
     * Constructor
     * @param attribute
     * @param maxRowId
     */
    public BitmapIndex(String attribute, int maxRowId) {
        this.attribute = attribute;
        this.maxRowId = maxRowId;
        bitmaps = new HashMap<>();
    }

    /**
     * Create a empty bitmap for a given key
     * @param key
     */
    private void createBitmapForKey(T key) {
        int arraySize = (maxRowId + 31) / 32;
        bitmaps.putIfAbsent(key, new int[arraySize]);
    }


    /**
     * This has been done for you.
     * @param key The attribute value.
     * @param rowId The row ID associated with the key.
     */
    public void insert(T key, int rowId) {
        createBitmapForKey(key);
        int index = rowId / 32;
        int bitPosition = rowId % 32;
        bitmaps.get(key)[index] |= (1 << bitPosition);
    }


    @Override
    /**
     * This is only for completeness. Although one can delete a key, it will mess up rowIds
     * If a record is deleted, then an unset bit may lead to ambiguity (is false vs not exists)
     */
    public boolean delete(T key) {
        return false;
    }


    @Override
    public List<Integer> evaluate(QueryNode node) {
        logger.info("Evaluating predicate using Bitmap index on attribute " + attribute + " for operator " + node.operator);
        // TODO: implement me
        return null;
    }

    @Override
    public List<Integer> search(T key) {
    //TODO: Implement me!
        return null;
    }

    @Override
    public String prettyName() {
        return "BitMap Index";
    }
}