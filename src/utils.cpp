#include "utils.h"

#include "stdio.h"

namespace mycv
{

void error(
    int error_code, 
    const std::string &error_msg, 
    const char* func_name, 
    const char* source_file, 
    int code_line
    )
    {
        printf("%s : %s. in %s function: %s , line : %d",
        SHOW_ERROR_CODE(error_code),error_msg.c_str(),source_file,func_name,code_line);
    }
}// end namespace mycv