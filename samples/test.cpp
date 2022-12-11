#include "mycv.hpp"

#include <string>
#include <iostream>

int main()
{
    int code = mycv::kImageEmpty;
    std::string msg = "test error";
    MYCV_ERROR(code,msg);

    return 0;
}