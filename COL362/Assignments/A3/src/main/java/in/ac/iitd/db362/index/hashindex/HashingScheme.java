package in.ac.iitd.db362.index.hashindex;

import java.time.LocalDate;

/**
 * Hashing Scheme for Extendible Hashing. It computes an offset in the bucket address table based on the global depth
 * higher order bits.
 * Note: Do not change this code! Use this hashing scheme in your Extendible Hashing implementation.
 */
public class HashingScheme {
    private static int getOffset(int hashValue, int globalDepth) {
        int mask = (1 << globalDepth) - 1; // Extract "globalDepth" higher order bits
        return hashValue & mask;
    }

    public static int getDirectoryIndex(int key, int globalDepth) {
        int hashValue = Integer.hashCode(key);
        return getOffset(hashValue, globalDepth);
    }

    public static int getDirectoryIndex(double key, int globalDepth) {
        int hashValue = Double.hashCode(key);
        return getOffset(hashValue, globalDepth);
    }

    public static int getDirectoryIndex(String key, int globalDepth) {
        int hashValue = key.hashCode();
        return getOffset(hashValue, globalDepth);
    }

    public static int getDirectoryIndex(LocalDate key, int globalDepth) {
        int hashValue = key.hashCode();
        return getOffset(hashValue, globalDepth);
    }
}
