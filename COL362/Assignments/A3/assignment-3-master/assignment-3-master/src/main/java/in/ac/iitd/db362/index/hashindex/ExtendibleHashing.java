package in.ac.iitd.db362.index.hashindex;

import in.ac.iitd.db362.index.Index;
import in.ac.iitd.db362.parser.QueryNode;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.List;


/**
 * Starter code for Extendible Hashing
 * @param <T> The type of the key.
 */
public class ExtendibleHashing<T> implements Index<T> {

    protected static final Logger logger = LogManager.getLogger();
    private String attribute; // attribute that we are indexing

   // Note: Do not rename the variable! You can initialize it to a different value for testing your code.
    public static int INITIAL_GLOBAL_DEPTH = 10;


    // Note: Do not rename the variable! You can initialize it to a different value for testing your code.
    public static int BUCKET_SIZE = 4;

    private int globalDepth;

    // directory is the bucket address table backed by an array of bucket pointers
    // the array offset (can be computed using the provided hashing scheme) allows accessing the bucket
    private Bucket<T>[] directory;


    /** Constructor */
    @SuppressWarnings("unchecked")
    public ExtendibleHashing(String attribute) {
        this.globalDepth = INITIAL_GLOBAL_DEPTH;
        int directorySize = 1 << globalDepth;
        this.directory = new Bucket[directorySize];
        for (int i = 0; i < directorySize; i++) {
            directory[i] = new Bucket<>(globalDepth);
        }
        this.attribute = attribute;
    }


    @Override
    public List<Integer> evaluate(QueryNode node) {
        logger.info("Evaluating predicate using Hash index on attribute " + attribute + " for operator " + node.operator);
        // TODO: Implement me!
        return null;
    }

    @Override
    public void insert(T key, int rowId) {
        // TODO: Implement insertion logic with bucket splitting and/or doubling the address table
    }


    @Override
    public boolean delete(T key) {
        // TODO: (Bonus) Implement deletion logic with bucket merging and/or shrinking the address table
        return false;
    }


    @Override
    public List<Integer> search(T key) {
        // TODO: Implement search logic
        return null;
    }

    /**
     * Note: Do not remove this function!
     * @return
     */
    public int getGlobalDepth() {
        return globalDepth;
    }

    /**
     * Note: Do not remove this function!
     * @param bucketId
     * @return
     */
    public int getLocalDepth(int bucketId) {
        return directory[bucketId].localDepth;
    }

    /**
     * Note: Do not remove this function!
     * @return
     */
    public int getBucketCount() {
        return directory.length;
    }


    public void printTable() {
        // TODO: You don't have to, but its good to print for small scale debugging
    }

    @Override
    public String prettyName() {
        return "Hash Index";
    }

}