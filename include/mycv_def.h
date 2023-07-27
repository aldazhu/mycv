/**
 * @file mycv_def.h
 * @author WuMing (hello@hello.com)
 * @brief 一些define
 * @version 0.1
 * @date 2022-12-10
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef MYCV_MYCV_DEF_H_
#define MYCV_MYCV_DEF_H_

#include "utils.h"

//copy from opencv
#ifdef mycv_func
// keep current value (through OpenCV port file)
#elif defined __GNUC__ || (defined (__cpluscplus) && (__cpluscplus >= 201103))
#define mycv_func __func__
#elif defined __clang__ && (__clang_minor__ * 100 + __clang_major__ >= 305)
#define mycv_func __func__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION >= 199901)
#define mycv_func __func__
#elif defined _MSC_VER
#define mycv_func __FUNCTION__
#elif defined(__INTEL_COMPILER) && (_INTEL_COMPILER >= 600)
#define mycv_func __FUNCTION__
#elif defined __IBMCPP__ && __IBMCPP__ >=500
#define mycv_func __FUNCTION__
#elif defined __BORLAND__ && (__BORLANDC__ >= 0x550)
#define mycv_func __FUNC__
#else
#define mycv_func "<unknown>"
#endif


constexpr double PI = 3.151492653;


#define MYCV_ERROR2(code,msg) mycv::error(code,msg,mycv_func,__FILE__,__LINE__)

#define MYCV_ERROR1(code) mycv::error(code,"",mycv_func,__FILE__,__LINE__)

#define CHECK_RET(code)		\
{							\
	if(code != mycv::error_code::kSuccess){		\
		MYCV_ERROR1(code);						\
		return code;							\
	}											\
}


#endif //!MYCV_MYCV_DEF_H_ 