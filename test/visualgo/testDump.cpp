#include "dump.h"
#include<stdio.h>
#include<math.h>

float func(float x) {
    return x * x;
}

int main() {
    float y = func(4.0);
    int x = sqrt(y);
    sqrt(y);
    int z;
    z = x + y;
    dump_int("test", z);
    printf("x = %d", x);

    return 0;
}