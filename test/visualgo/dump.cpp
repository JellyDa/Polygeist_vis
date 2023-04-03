#include<iostream>
#include<string>

int dump_int(const char *var, int v) {
    std::string s(var);
    std::cout << s << "\t" << v << std::endl;
    return 0;
}