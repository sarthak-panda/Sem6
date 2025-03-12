package in.ac.iitd.db362.index.hashindex;

/**
 * Bucket(s) used in Extendible Hashing
 * Note: Do not change anything in this file!
 * @param <T>
 */
public class Bucket<T> {
    int localDepth;

    T[] keys;

    int[] values; //our values are rowIds so lets use integer

    int size; // current size of the bucket (number of entries)

    Bucket<T> next; // Pointer to the next bucket for overflow handling


    @SuppressWarnings("unchecked")
    Bucket(int localDepth) {

        this.localDepth = localDepth;

        this.keys = (T[]) new Object[ExtendibleHashing.BUCKET_SIZE];

        this.values = new int[ExtendibleHashing.BUCKET_SIZE];

        this.size = 0;

        this.next = null;
    }
}
