# COL362/632 Assignment-3

[Assignment Document](https://docs.google.com/document/d/14St9wFD6lcLRQjLlQQo38yEBuI2Jo9BXFG3bqBuw2uo/edit?usp=sharing)

## Requirements
- Java 11
- Maven 3.9.x



## Clone the project
1. Create directory ````mkdir path/to/assignment_3/````
2. cd into the newly created directory by ````cd path/to/assignment_3/````
3. Run ```git clone git@git.iitd.ac.in:2402COL362/assignment-3.git .``` to clone the project on your local machine (note the dot when cloning into the newly created directory). You must have set your ssh keys for this.
4. Run ``mvn clean install -DskipTest``
6. Import the project into your favorite editor. I strongly recommend Intellj.
    
## Assignment Tasks

### 1. Implement B+Tree Index.
The starter code has been provided in ``in/ac/iitd/db362/index/bplustree/BPlusTreeIndex.java``

You should further develop this class by completing all TODOs. Your B+tree index should be **right bias**. The node structure used in the B+Tree is specified in ``in/ac/iitd/db362/index/bplustree/Node.java``
Duplicate Keys: To handle duplicate keys, exploit the given node structure and implement the duplicate handling approach that allows leaf nodes to spill into overflow nodes. 
Study the node structure carefully and do not modify it. Note that the test cases will be based on B+tree logic discussed in the class.

  
### 2. Implement Hash Index based on Extendible Hashing
The starter code has been provided in ``in/ac/iitd/db362/index/hashindex/ExtendibleHashing.java``

You should further develop this class by completing all TODOs. There is a hashing scheme provided in ``in/ac/iitd/db362/index/hashindex/HashingScheme.java``. The scheme includes a ``getDirectoryIndex`` method that returns the offset of the bucket address table based on globaldepth higher order bits. Your logic should use this hashing scheme. Do not modify the hashing scheme. Further, the index uses a bucket structure specified in ``in/ac/iitd/db362/index/hashindex/Bucket.java``. Study the bucket structure carefully and do not modify it. The test cases will be based on the hashing logic discussed in the class.

 
### 3. Implement Search using BitMap Index.
The starter code has been provided in ``in/ac/iitd/db362/index/BitmapIndex.java``

The BitMap index is backed by array of 32 bit integers, as Java does not natively support bit vectors. Study the insertion logic carefully to understand how this is done. You should implement the search functionality for this index.
 
### 4. Implement Query Processor
The starter code has been provided in ``in/ac/iitd/db362/processor/QueryEvaluator.java``.

You must complete this class by implementing the ``evaluateQuery`` method. **Note that you implementation must  use the** ``evaluatePredicate`` method for evaluating all predicates in a query. Do not change or remove this function.


>Important
>Make sure to carefully read all comments provided in the code. 
>There are certain classes, files, data structures and variable names that you should not modify. 
>Modifying code that you should not will lead to failing of test cases.

  

## Testing your code

The starter code has been developed using Java 11, which will also be the version that will be used for testing. Make sure that you have Java verion 11 before proceeding with the assignment. Also install maven for installing and testing your code

````
cd /path/to/assignment-3/
mvn test
````

There are two very simple tests that are provied in the ``src/test/java/in/ac/iitd/db362``. The ``CSVParserTest.java`` contains a ``@Disabled`` annotation for the ``testValidCSVParsing``  test. You can remove it after implementing the indexes to test parsing and index creation. To add new test cases, create new test files and follow similar syntax as already included ones. (should include a ``@Test`` annotation before the test function).


  

## How and What to submit

 Create a patch file of your submission
````
cd path/to/assignment_3
git diff [COMMITID] > [ENTRYNO].patch
 ````
>Repace [ENTRYNO] with your entry number.
>Replace [COMMITID] with the one that will be provided to you (this will be provided 2 days before the submission deadline)

**Upload your patch file on Moodle by 26th March 2025 11:59PM**
