#include "resize.h"


namespace mycv
{

	static int ResizeNearst(const cv::Mat& src, cv::Mat& dst, const float fx, const float fy)
	{
		CHECK_RET(CheckImageEmpty(src));
		CHECK_RET(CheckImageEmpty(dst));
		for (int row = 0; row < dst.rows; ++row)
		{
			uchar* pd = dst.ptr<uchar>(row);
			// *    * -->ps_row
			//   .
			// *    * -->ps_row + 1
			float ps_y_loc = row / fy;
			int ps_row = static_cast<int>(ps_y_loc);// 从目标图映射会原图时的整数行
			float delta_y = ps_y_loc - ps_row;
			ps_row = delta_y < 0.5 ? ps_row : ps_row + 1;
			ps_row = ps_row < src.rows ? ps_row : src.rows - 1;
			const uchar* ps = src.ptr<uchar>(ps_row);
			for (int col = 0; col < dst.cols; ++col)
			{
				float ps_x_loc = col / fx;
				int ps_col = static_cast<int>(ps_x_loc);
				float delta_x = ps_x_loc - ps_col;
				ps_col = delta_x < 0.5 ? ps_col : ps_col + 1;
				ps_col = ps_col < src.cols ? ps_col : src.cols - 1;
				*(pd + col) = *(ps + ps_col);
			}
		}
		return error_code::kSuccess;
	}

	static int ResizeBilinear(const cv::Mat& src, cv::Mat& dst, const float fx, const float fy)
	{
		CHECK_RET(CheckImageEmpty(src));
		CHECK_RET(CheckImageEmpty(dst));
		for (int row = 0; row < dst.rows; ++row)
		{
			uchar* pd = dst.ptr<uchar>(row);
			// *    * -->ps_row
			//   .
			// *    * -->ps_row + 1
			float ps_y_loc = row / fy;
			int ps_row = static_cast<int>(ps_y_loc);// 从目标图映射会原图时的整数行
			float delta_y = ps_y_loc - ps_row;
			ps_row = delta_y < 0.5 ? ps_row : ps_row + 1;
			ps_row = ps_row < src.rows ? ps_row : src.rows - 1;
			const uchar* ps = src.ptr<uchar>(ps_row);
			for (int col = 0; col < dst.cols; ++col)
			{
				float ps_x_loc = col / fx;
				int ps_col = static_cast<int>(ps_x_loc);
				float delta_x = ps_x_loc - ps_col;
				ps_col = delta_x < 0.5 ? ps_col : ps_col + 1;
				ps_col = ps_col < src.cols ? ps_col : src.cols - 1;
				*(pd + col) = *(ps + ps_col);
			}
		}
		return error_code::kSuccess;
		
	}

	int Resize(const cv::Mat& src, cv::Mat& dst, const int new_h, const int new_w, const int mode)
	{
		CHECK_RET(CheckImageEmpty(src));
		if (new_h <= 0 || new_w <= 0) return error_code::kOutOfRange;
		if (CV_8UC1 != src.type()) return error_code::kNotImplement;

		dst = cv::Mat::zeros(new_h, new_w, CV_8UC1);
		float fx = static_cast<float>(new_w) / src.cols;
		float fy = static_cast<float>(new_h) / src.rows;

		int status = error_code::kSuccess;
		switch (mode)
		{
		case 0:
			status = ResizeNearst(src, dst, new_h, new_w);
			break;
		case 1:
			status = ResizeBilinear(src, dst, new_h, new_w);
			break;
		default:
			status = ResizeNearst(src, dst, new_h, new_w);
			break;
		}

		return status;
	}
}// end namespace
