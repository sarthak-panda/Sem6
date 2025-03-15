package in.ac.iitd.db362.index.hashindex;

import in.ac.iitd.db362.index.Index;
import in.ac.iitd.db362.parser.QueryNode;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
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

    private boolean tryInsert(Bucket<T> bucket, T key, int rowId) {
        Bucket<T> current = bucket;
        while (current != null) {
            if (current.size < BUCKET_SIZE) {
                current.keys[current.size] = key;
                current.values[current.size] = rowId;
                current.size++;
                return true;
            }
            current = current.next;
        }
        return false;
    }

    @Override
    public void insert(T key, int rowId) {
        // TODO: Implement insertion logic with bucket splitting and/or doubling the address table
        int directoryIndex = HashingScheme.getDirectoryIndex(key, globalDepth);
        Bucket<T> bucket = directory[directoryIndex];
        if (tryInsert(bucket, key, rowId)) {
            return;
        } else {
            if (bucket.localDepth == globalDepth) {
                globalDepth++;
                int newDirectorySize = 1 << globalDepth;
                int oldSize = directory.length;
                Bucket<T>[] newDirectory = new Bucket[newDirectorySize];
                for (int i = 0; i < newDirectorySize; i++) {
                    newDirectory[i] = new Bucket<>(globalDepth);
                }
                for (int i = 0; i < oldSize; i++) {
                    newDirectory[i] = directory[i];
                    newDirectory[i + oldSize] = directory[i];
                }
                directory = newDirectory;
            }
            bucket.localDepth++;
            Bucket<T> newBucket = new Bucket<>(bucket.localDepth + 1);
            Bucket<T> current = bucket;
            // Determine which directory entries (pointing to bucketToSplit) should now point to the new bucket.
            // The bit position to check is the new bit (at position localDepth-1).
            int bitMask = 1 << (bucket.localDepth - 1);
            for (int i = 0; i < directory.length; i++) {
                if (directory[i] == bucket && ((i & bitMask) != 0)) {
                    directory[i] = newBucket;
                }
            }
            List<T> allKeys = new ArrayList<>();
            List<Integer> allValues = new ArrayList<>();
            // Save data from primary bucket and all overflow buckets in a single loop
            Bucket<T> currentBucket = bucket;
            while (currentBucket != null) {
                for (int i = 0; i < currentBucket.size; i++) {
                    allKeys.add(currentBucket.keys[i]);
                    allValues.add(currentBucket.values[i]);
                }
                currentBucket = currentBucket.next;
            }
            
            while (current != null) {
                for (int i = 0; i < current.size; i++) {
                    int newDirectoryIndex = HashingScheme.getDirectoryIndex(current.keys[i], globalDepth);
                    if (newDirectoryIndex == directoryIndex) {
                        newBucket.keys[newBucket.size] = current.keys[i];
                        newBucket.values[newBucket.size] = current.values[i];
                        newBucket.size++;
                    }
                }
                current = current.next;
            }
            current = bucket;
            while (current != null) {
                for (int i = 0; i < current.size; i++) {
                    int newDirectoryIndex = HashingScheme.getDirectoryIndex(current.keys[i], globalDepth);
                    if (newDirectoryIndex == directoryIndex) {
                        current.keys[i] = null;
                        current.values[i] = -1;
                        current.size--;
                    }
                }
                current = current.next;
            }
            newBucket.next = bucket.next;
            bucket.next = newBucket;
            insert(key, rowId);
        }
    }


    @Override
    public boolean delete(T key) {
        // TODO: (Bonus) Implement deletion logic with bucket merging and/or shrinking the address table
        return false;
    }


    @Override
    public List<Integer> search(T key) {
        // TODO: Implement search logic
        int directoryIndex = HashingScheme.getDirectoryIndex(key, globalDepth);
        List<Integer> results = new ArrayList<>();
        Bucket<T> current = directory[directoryIndex];
        while (current != null) {
            for (int i = 0; i < current.size; i++) {
                if (current.keys[i].equals(key)) {
                    results.add(current.values[i]);
                }
            }
            current = current.next;
        }
        return results;
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