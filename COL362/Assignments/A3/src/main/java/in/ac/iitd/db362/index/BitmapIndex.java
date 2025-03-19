package in.ac.iitd.db362.index;

import in.ac.iitd.db362.parser.Operator;
import in.ac.iitd.db362.parser.QueryNode;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collections;
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

    @SuppressWarnings("unchecked")
    private T parseKey(String value) {
        // Try parsing as Integer
        try {
            Integer intValue = Integer.parseInt(value);
            return (T) intValue;
        } catch (NumberFormatException e) {
            // Proceed to next type
        }

        // Try parsing as Double
        try {
            Double doubleValue = Double.parseDouble(value);
            return (T) doubleValue;
        } catch (NumberFormatException e) {
            // Proceed to next type
        }

        // Try parsing as LocalDate (ISO format: yyyy-MM-dd)
        try {
            LocalDate dateValue = LocalDate.parse(value);
            return (T) dateValue;
        } catch (Exception e) {
            // Proceed to next type
        }

        // Return as String if all else fails
        return (T) value;
    }


    @Override
    public List<Integer> evaluate(QueryNode node) {
        logger.info("Evaluating predicate using Bitmap index on attribute " + attribute + " for operator " + node.operator);

        if (node.operator != Operator.EQUALS) {
            logger.error("Bitmap index only supports equality queries");
            return Collections.emptyList();
        }
        T key = parseKey(node.value);
        return search(key);
    }

    @Override
    public List<Integer> search(T key) {
        List<Integer> result = new ArrayList<>();
        int[] bitmap = bitmaps.get(key);
        if (bitmap == null) {
            return result;
        }
        for (int i = 0; i < bitmap.length; i++) {
            int bits = bitmap[i];
            for (int bitPos = 0; bitPos < 32; bitPos++) {
                int rowId = i * 32 + bitPos;
                if (rowId > maxRowId) break;
                if ((bits & (1 << bitPos)) != 0) {
                    result.add(rowId);
                }
            }
        }
        return result;
    }

    @Override
    public String prettyName() {
        return "BitMap Index";
    }
}