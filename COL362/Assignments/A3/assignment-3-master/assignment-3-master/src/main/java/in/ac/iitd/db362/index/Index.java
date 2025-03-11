package in.ac.iitd.db362.index;

import in.ac.iitd.db362.parser.QueryNode;

import java.util.List;


/**
 * Generic Index interface for an index on an attribute of type T.
 * Note: Do not change this file!
 * @param <T> The type of the key stored in the index.
 */
public interface Index<T> {

    /**
     * Evaluate the predicate represented by the query node using this index.
     *
     * @param node The query node containing the predicate details.
     * @return A list of row IDs that satisfy the predicate.
     */
    List<Integer> evaluate(QueryNode node);

    /**
     * Insert a new entry into the index.
     *
     * @param key The attribute value.
     * @param rowId The row ID associated with the key.
     */
    void insert(T key, int rowId);

    /**
     * Delete an entry from the index.
     *
     * @param key The attribute value.
     */
    boolean delete(T key);

    /**
     * Search for the given key in the index.
     *
     * @param key The attribute value.
     * @return A list of row IDs associated with the key.
     */
    List<Integer> search(T key);


    String prettyName();

}
