/*****************************************************************//**
 * @file   halcon_fun.h
 * @copyright 2023.  All right sreserved.
 * @brief  
 * 
 * @author zsP
 * @date   2023/6/29 
 *********************************************************************/
#ifndef HALCON_FUN_H_
#define HALCON_FUN_H_

#ifdef HAS_HALCON

#include "halconcpp/HalconCpp.h"
#include <map>
#include <string>

namespace mycv {

	enum class ImageChannelMode
	{
		B,
		G,
		R,
		GRAY,
		BGR,
	};
	typedef struct 
	{
		byte* data;
		int width;
		int height;
		int stride;
	}BmpData;
	int  CreateHalconShapeModel(HalconCpp::HImage& temp, double start_angle, double end_angle, HalconCpp::HTuple& model_id);
	int FindHalconShapeModel(HalconCpp::HImage& img, const HalconCpp::HTuple& model_id, double start_angle, double end_angle, double rectResult[4]);

	



} //end namespace mycv

#endif //HAS_HALCON

#endif // !HALCON_FUN_H_

