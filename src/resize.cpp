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

	/**
	 * @brief 由四个领域的像素插值得到最后的像素值.
	 * 
	 * @param [in] data : 四个领域像素的灰度值， 分别是左上，上右， 左下，右下
	 * @param [in] delta_x : 距离左上点的横坐标的距离，[0,1]
	 * @param [in] delte_y : 距离左上点的纵坐标的距离，[0,1]
	 * @return  : 双线性插值的最终结果
	 */
	static inline float CalculateBilinearInterpolation(const uchar data[], const float delta_x, const float delte_y)
	{
		// *    * 
		//   .
		// *    * 
		// 采用城区距离来作为权重，会比用欧式距离来得快， 因为欧式距离需要平方再开方；
		float tl = delta_x + delte_y;				// top left
		float tr = (1 - delta_x) + delte_y;			// top right
		float bl = delta_x + (1 - delte_y);			// botom left
		float br = (1 - delta_x) + (1 - delte_y);	// bottom right
		//float distance_all = tl + bl + tr + br;// 城区距离的和一直为4
		float val = static_cast<float>(data[0]) * tl + static_cast<float>(data[1]) * tr + static_cast<float>(data[2]) * bl + static_cast<float>(data[3]) * br;
		//distance_all = std::abs(distance_all) > 0.0000001f ? distance_all : 0.000001f;
		val /= 4.0f;
		return val;
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
			// *    * -->ps_row + 1 : ps_next_row
			float ps_y_loc = row / fy;
			int ps_row = static_cast<int>(ps_y_loc);// 从目标图映射会原图时的整数行
			float delta_y = ps_y_loc - ps_row;
			ps_row = delta_y < 0.5 ? ps_row : ps_row + 1;
			ps_row = ps_row < src.rows ? ps_row : src.rows - 1;
			const uchar* ps = src.ptr<uchar>(ps_row);
			int ps_next_row = ps_row + 1;
			ps_next_row = ps_next_row < src.rows ? ps_next_row : src.rows - 1;
			const uchar* ps_next = src.ptr<uchar>(ps_next_row);
			for (int col = 0; col < dst.cols; ++col)
			{
				float ps_x_loc = col / fx;
				int ps_col = static_cast<int>(ps_x_loc);
				float delta_x = ps_x_loc - ps_col;
				ps_col = delta_x < 0.5 ? ps_col : ps_col + 1;
				ps_col = ps_col < src.cols ? ps_col : src.cols - 1;
				int ps_next_col = ps_col + 1;
				ps_next_col = ps_next_col < src.cols ? ps_next_col : src.cols - 1;

				uchar neighbor4[4] = {  * (ps + ps_col) , *(ps + ps_next_col), *(ps_next + ps_col), *(ps_next + ps_next_col)};
				*(pd + col) = static_cast<uchar>(CalculateBilinearInterpolation(neighbor4, delta_x, delta_y));
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
			status = ResizeNearst(src, dst, fx, fy);
			break;
		case 1:
			status = ResizeBilinear(src, dst, fx, fy);
			break;
		default:
			status = ResizeNearst(src, dst, fx, fy);
			break;
		}

		return status;
	}
}// end namespace
