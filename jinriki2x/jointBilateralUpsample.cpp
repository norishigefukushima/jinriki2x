#include "util.h"
#include <opencv2/core/internal.hpp>
using namespace std;

template<class T>
inline T weightedinterpolation64fw_(T lt, T rt, T lb, T rb, double wlt, double wrt, double wlb, double  wrb)
{
	return (T)((wlt*lt + wrt*rt + wlb*lb + wrb*rb)/(wlt + wrt + wlb + wrb)+0.5);
}

template<class T>
inline T weightedinterpolation32fw_(T lt, T rt, T lb, T rb, float wlt, float wrt, float wlb, float  wrb)
{
	return (T)((wlt*lt + wrt*rt + wlb*lb + wrb*rb)/(wlt + wrt + wlb + wrb)+0.5);
}

template<class T>
void jointBilateralUpsample_(Mat& src, Mat& joint, Mat& dest, double sigma_c, double sigma_s)
{
	if(dest.empty())dest.create(joint.size(),src.type());
	Mat sim,jim,eim;
	copyMakeBorder(src,sim,0,1,0,1,cv::BORDER_REPLICATE);
	const int dw = (joint.cols)/(src.cols);
	const int dh = (joint.rows)/(src.rows);

	copyMakeBorder(joint,jim,0,1,0,1,cv::BORDER_REPLICATE);

	double lut[256*3];
	double gauss_c_coeff = -0.5/(sigma_c*sigma_c);
	for(int i=0;i<256*3;i++)
	{
		lut[i] = (double)std::exp(i*i*gauss_c_coeff);
	}
	vector<double> lut_(dw*dh);
	double* lut2 = &lut_[0];
	if(sigma_s<=0.0)
	{
		for(int i=0;i<dw*dh;i++)
		{
			lut2[i] = 1.0;
		}
	}
	else
	{
		double gauss_s_coeff = -0.5/(sigma_s*sigma_s);
		for(int i=0;i<dw*dh;i++)
		{
			lut2[i] = (double)std::exp(i*i*gauss_s_coeff);
		}
	}

	for(int j=0;j<src.rows;j++)
	{
		int n=j*dh;
		T* s = sim.ptr<T>(j);
		uchar* jnt_ = jim.ptr(n);

		for(int i=0,m=0;i<src.cols;i++,m+=dw)
		{	
			const uchar ltr = jnt_[3*m+0];
			const uchar ltg = jnt_[3*m+1];
			const uchar ltb = jnt_[3*m+2];

			const uchar rtr = jnt_[3*(m+dw)+0];
			const uchar rtg = jnt_[3*(m+dw)+1];
			const uchar rtb = jnt_[3*(m+dw)+2];

			const uchar lbr = jnt_[3*(m + jim.cols*dh) +0];
			const uchar lbg = jnt_[3*(m + jim.cols*dh) +1];
			const uchar lbb = jnt_[3*(m + jim.cols*dh) +2];

			const uchar rbr = jnt_[3*(m+dw + jim.cols*dh) +0];
			const uchar rbg = jnt_[3*(m+dw + jim.cols*dh) +1];
			const uchar rbb = jnt_[3*(m+dw + jim.cols*dh) +2];

			const T ltd = s[i];
			const T rtd = s[i+1];
			const T lbd = s[i+sim.cols];
			const T rbd = s[i+1+sim.cols];

			for(int l=0;l<dh;l++)
			{
				T* d = dest.ptr<T>(n+l);
				uchar* jnt = jim.ptr(n+l);

				for(int k=0;k<dw;k++)
				{
					const uchar r = jnt[3*(m+k)+0];
					const uchar g = jnt[3*(m+k)+1];
					const uchar b = jnt[3*(m+k)+2];

					//double
					double wlt = lut2[k+l]      *lut[abs(ltr-r)+abs(ltg-g)+abs(ltb-b)];
					double wrt = lut2[dw-k+l]   *lut[abs(rtr-r)+abs(rtg-g)+abs(rtb-b)];				
					double wlb = lut2[k+dh-l]   *lut[abs(lbr-r)+abs(lbg-g)+abs(lbb-b)];
					double wrb = lut2[dw-k+dh-l]*lut[abs(rbr-r)+abs(rbg-g)+abs(rbb-b)];
					d[m+k] = weightedinterpolation64fw_<T>(ltd,rtd,lbd,rbd,wlt,wrt,wlb,wrb);

					//float 
					/*
					float wlt = lut2[k+l]      *lut[abs(ltr-r)+abs(ltg-g)+abs(ltb-b)];
					float wrt = lut2[dw-k+l]   *lut[abs(rtr-r)+abs(rtg-g)+abs(rtb-b)];				
					float wlb = lut2[k+dh-l]   *lut[abs(lbr-r)+abs(lbg-g)+abs(lbb-b)];
					float wrb = lut2[dw-k+dh-l]*lut[abs(rbr-r)+abs(rbg-g)+abs(rbb-b)];
					d[m+k] = weightedinterpolation32fw_<T>(ltd,rtd,lbd,rbd,wlt,wrt,wlb,wrb);
					*/
				}
			}
		}
	}
}

void jointBilateralUpsample(InputArray src, InputArray joint, OutputArray dest, const double sigma_c, const double sigma_s)
{
	if(dest.empty())dest.create(joint.size(),src.type());
	if(joint.size()!=dest.size())dest.create(joint.size(),src.type());
	if(src.type()!=dest.type())dest.create(joint.size(),src.type());

	Mat s = src.getMat();
	Mat j = joint.getMat();
	Mat d = dest.getMat();

		 if(src.depth()==CV_8U)  jointBilateralUpsample_<uchar>(s, j, d, sigma_c, sigma_s);
	else if(src.depth()==CV_16S) jointBilateralUpsample_<short>(s, j, d, sigma_c, sigma_s);
	else if(src.depth()==CV_16U) jointBilateralUpsample_<ushort>(s, j, d, sigma_c, sigma_s);
	else if(src.depth()==CV_32S) jointBilateralUpsample_<int>(s, j, d, sigma_c, sigma_s);
	else if(src.depth()==CV_32F) jointBilateralUpsample_<float>(s, j, d, sigma_c, sigma_s);
	else if(src.depth()==CV_64F) jointBilateralUpsample_<double>(s, j, d, sigma_c, sigma_s);
}
