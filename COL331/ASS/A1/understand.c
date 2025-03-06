#include <stdlib.h>
#include <stdio.h>

int main() {
    void *ptr1 = malloc(1024);         // 1 KB allocation
    void *ptr2 = malloc(1024 * 1024);    // 1 MB allocation
    if (ptr1 && ptr2) {
        printf("Allocations successful\n");
    }
    free(ptr1);
    free(ptr2);
    return 0;
}
