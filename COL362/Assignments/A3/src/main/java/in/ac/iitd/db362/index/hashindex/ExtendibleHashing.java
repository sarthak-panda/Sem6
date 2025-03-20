package in.ac.iitd.db362.index.hashindex;

import in.ac.iitd.db362.index.Index;
import in.ac.iitd.db362.parser.Operator;
import in.ac.iitd.db362.parser.QueryNode;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collections;
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
        logger.info("Evaluating predicate using Hash index on attribute " + attribute + " for operator " + node.operator);

        if (node.operator != Operator.EQUALS) {
            logger.error("Hash index only supports equality queries");
            return Collections.emptyList();
        }
        T key = parseKey(node.value);
        return search(key);
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

    private int getDirectoryIndexForKey(T key, int globalDepth) {
    if (key instanceof Integer) {
        return HashingScheme.getDirectoryIndex((Integer) key, globalDepth);
    } else if (key instanceof Double) {
        return HashingScheme.getDirectoryIndex((Double) key, globalDepth);
    } else if (key instanceof String) {
        return HashingScheme.getDirectoryIndex((String) key, globalDepth);
    } else if (key instanceof LocalDate) {
        return HashingScheme.getDirectoryIndex((LocalDate) key, globalDepth);
    } else {
        throw new IllegalArgumentException("Unsupported key type: " + key.getClass());
    }
}

    @Override
    public void insert(T key, int rowId) {
        int directoryIndex = getDirectoryIndexForKey(key, globalDepth);
        Bucket<T> bucket = directory[directoryIndex];
        if (tryInsert(bucket, key, rowId)) {
            return;
        } else {
            if (bucket.localDepth == globalDepth) {
                globalDepth++;
                int newDirectorySize = 1 << globalDepth;
                int oldSize = directory.length;
                Bucket<T>[] newDirectory = (Bucket<T>[]) new Bucket[newDirectorySize];
                for (int i = 0; i < oldSize; i++) {
                    newDirectory[i] = directory[i];
                    newDirectory[i + oldSize] = directory[i];
                }
                directory = newDirectory;
            }
            bucket.localDepth++;
            Bucket<T> newBucket = new Bucket<>(bucket.localDepth);
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
            int oldBucketCount =0;
            int newBucketCount = 0;
            for (int i = 0; i < allKeys.size(); i++) {
                T currentKey = allKeys.get(i);
                int currentKeyIndex = getDirectoryIndexForKey(currentKey, globalDepth);
                if (directory[currentKeyIndex] == bucket) {
                    oldBucketCount++;
                }
                else {//i.e. if (directory[currentKeyIndex] == newBucket)
                    newBucketCount++;
                }
            }
            if(oldBucketCount == allKeys.size() || newBucketCount == allKeys.size()){
                //no meaning to split we will add overflow bucket and insert later into that
                //first let us undo the directory mapping
                for (int i = 0; i < directory.length; i++) {
                    if (directory[i] == bucket) {
                        directory[i] = newBucket;//we are planning to add overflow bucket at head of list
                    }
                }
                bucket.localDepth--;
                newBucket.localDepth--;
                newBucket.next = bucket;    //add overflow bucket at head of list
                insert(key, rowId); 
                return;
            }
            //doing a split is meaningful so we will redistribute the keys
            //let us first clear the old bucket and remove the overflow chains from it
            //we empty there keys and values array in each bucket
            bucket.size = 0;
            bucket.keys = (T[]) new Object[BUCKET_SIZE];
            bucket.values = new int[BUCKET_SIZE];
            bucket.next = null;//break the old overflow chain
            //now we will redistribute the keys
            for (int i = 0; i < allKeys.size(); i++) {
                T currentKey = allKeys.get(i);
                int currentRowId = allValues.get(i);
                int currentKeyIndex = getDirectoryIndexForKey(currentKey, globalDepth);
                Bucket<T> targetBucket = directory[currentKeyIndex];
                //we have to careflly add the overflow bucket in the new bucket/old bucket if needed
                Bucket<T> temp = targetBucket;
                while (temp != null) {
                    if (temp.size < BUCKET_SIZE) {
                        temp.keys[temp.size] = currentKey;
                        temp.values[temp.size] = currentRowId;
                        temp.size++;
                        break;
                    }
                    if(temp.next == null){
                        temp.next=new Bucket<>(targetBucket.localDepth);
                    }
                    temp = temp.next;
                }
            }
            insert(key, rowId);
            // int directoryIndexF = getDirectoryIndexForKey(key, globalDepth);
            // Bucket<T> bucketF = directory[directoryIndexF];
            // if (tryInsert(bucketF, key, rowId)) {
            //     return;
            // }
            // else {
            //     throw new RuntimeException("Failed to insert key after split/double");
            // }
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
        int directoryIndex = getDirectoryIndexForKey(key, globalDepth);
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