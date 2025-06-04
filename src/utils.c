#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

void check_null(void *ptr, const char *message) {
    if (ptr == NULL) {
        fprintf(stderr, "Error: %s\n", message);
        exit(EXIT_FAILURE);
    }
}