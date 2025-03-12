package in.ac.iitd.db362.io;



import in.ac.iitd.db362.index.Index;
import in.ac.iitd.db362.catalog.Catalog;
import in.ac.iitd.db362.index.bplustree.BPlusTreeIndex;
import in.ac.iitd.db362.index.hashindex.ExtendibleHashing;
import in.ac.iitd.db362.index.BitmapIndex;
import org.junit.jupiter.api.*;
import java.nio.file.*;
import java.io.IOException;
import java.time.LocalDate;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class CSVParserTest {

    // Reset the catalog before each test.
    @BeforeEach
    void setUp() {
        // Reinitialize the catalog so that tests are independent.
        Catalog catalog = Catalog.getInstance();
    }

    //TODO: remove @Disabled after your implementation to test your code
    @Test
    @Disabled
    void testValidCSVParsing() throws IOException {
        // Create a temporary CSV file.
        Path tempFile = Files.createTempFile("test", ".csv");
        // CSV content with a header and three rows.
        // Header format: attributeName:attributeType
        // Attributes: salary (double), department (string), date (date)
        String csvContent = String.join("\n",
                "salary:double,department:string,date:date",
                "10000,HR,2025-01-01",
                "20000,Engineering,2025-01-02",
                "15000,HR,2025-01-03"
        );
        Files.write(tempFile, csvContent.getBytes());

        // Specify which indexes to create for each attribute.
        Map<String, List<String>> indexesToCreate = new HashMap<>();
        indexesToCreate.put("salary", Collections.singletonList("BPlusTree"));
        indexesToCreate.put("department", Arrays.asList("Bitmap", "Hash"));
        indexesToCreate.put("date", Collections.singletonList("BPlusTree"));

        Catalog catalog = Catalog.getInstance();

        //
        int maxRowId = 2;
        CSVParser.parseCSV(tempFile.toString(), ",", catalog, indexesToCreate, maxRowId);

        // Retrieve the indexes from the catalog.
        // For "salary" we expect a BPlusTreeIndex<Double>.
        List<Index> salaryIndexes = catalog.getIndexes("salary");
        assertNotNull(salaryIndexes);
        assertFalse(salaryIndexes.isEmpty());
        Index<?> salaryIndex = salaryIndexes.get(0);
        assertTrue(salaryIndex instanceof BPlusTreeIndex);

        // Cast and test a search for a known key.
        @SuppressWarnings("unchecked")
        BPlusTreeIndex<Double> salaryBPT = (BPlusTreeIndex<Double>) salaryIndex;
        // The CSV row 0 has salary 10000.
        List<Integer> rowsFor10000 = salaryBPT.search(10000.0);
        // Expect row id 0 to be present.
        assertTrue(rowsFor10000.contains(0));

        // For "department", we expect both BitmapIndex and HashIndex.
        List<Index> deptIndexes = catalog.getIndexes("department");
        assertEquals(2, deptIndexes.size());
        boolean hasBitmap = deptIndexes.stream().anyMatch(idx -> idx instanceof BitmapIndex);
        boolean hasHash = deptIndexes.stream().anyMatch(idx -> idx instanceof ExtendibleHashing);
        assertTrue(hasBitmap, "Bitmap index expected for department");
        assertTrue(hasHash, "Hash index expected for department");

        // For example, the CSV rows with department "HR" are row 0 and row 2.
        for (Index<?> idx : deptIndexes) {
            @SuppressWarnings("unchecked")
            Index<String> deptIndex = (Index<String>) idx;
            List<Integer> hrRows = deptIndex.search("HR");
            if (idx instanceof BitmapIndex) {
                // Bitmap index should return row ids as set bits.
                assertTrue(hrRows.contains(0));
                assertTrue(hrRows.contains(2));
            } else if (idx instanceof ExtendibleHashing) {
                assertTrue(hrRows.contains(0));
                assertTrue(hrRows.contains(2));
            }
        }

        // For "date", we expect a BPlusTreeIndex<LocalDate>.
        List<Index> dateIndexes = catalog.getIndexes("date");
        assertEquals(1, dateIndexes.size());
        Index<?> dateIndex = dateIndexes.get(0);
        assertTrue(dateIndex instanceof BPlusTreeIndex);
        @SuppressWarnings("unchecked")
        BPlusTreeIndex<LocalDate> dateBPT = (BPlusTreeIndex<LocalDate>) dateIndex;
        // The first row date is 2025-01-01.
        List<Integer> dateRows = dateBPT.search(LocalDate.parse("2025-01-01"));
        assertTrue(dateRows.contains(0));

        // Clean up the temporary file.
        Files.deleteIfExists(tempFile);
    }

    @Test
    void testEmptyCSVFile() throws IOException {
        // Create an empty temporary file.
        Path tempFile = Files.createTempFile("empty", ".csv");
        Files.write(tempFile, "".getBytes());

        Catalog catalog = Catalog.getInstance();

        Map<String, List<String>> indexesToCreate = new HashMap<>();
        // No indexes to create.
        CSVParser.parseCSV(tempFile.toString(), ",", catalog, indexesToCreate, 100);

        // The catalog should remain empty.
        assertTrue(catalog.getIndexes("any") == null ||
                catalog.getIndexes("any").isEmpty());
        Files.deleteIfExists(tempFile);
    }

    @Test
    void testInvalidHeaderFormat() throws IOException {
        // Create a temporary CSV file with an invalid header.
        Path tempFile = Files.createTempFile("invalidHeader", ".csv");
        // Header is missing the colon separator.
        String csvContent = String.join("\n",
                "salary,double,department:string",
                "10000,HR"
        );
        Files.write(tempFile, csvContent.getBytes());

        Map<String, List<String>> indexesToCreate = new HashMap<>();
        indexesToCreate.put("salary", Collections.singletonList("BPlusTree"));

        Catalog catalog = Catalog.getInstance();

        // This should print an error message for the invalid header token,
        // but still process remaining tokens.
        CSVParser.parseCSV(tempFile.toString(), ",", catalog, indexesToCreate, 100);

        // In this case, salary may not have been registered due to header parsing failure.
        List<?> salaryIndexes = catalog.getIndexes("salary");
        // We cannot guarantee registration so we check for null or empty.
        assertTrue(salaryIndexes == null || salaryIndexes.isEmpty());
        Files.deleteIfExists(tempFile);
    }
}
