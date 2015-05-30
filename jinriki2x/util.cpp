#include "util.h"
#include <stdarg.h>
using namespace std;

void showMatInfo(InputArray src_, string name)
{
	Mat src = src_.getMat();

	cout<<name<<":"<<endl;
	if(src.empty())
	{
		cout<<"empty"<<endl;
		return;
	}
	cout<<"size    "<<src.size()<<endl;
	cout<<"channel "<<src.channels()<<endl;
	if(src.depth()==CV_8U)cout<<"8U"<<endl;
	else if(src.depth()==CV_16S)cout<<"16S"<<endl;
	else if(src.depth()==CV_16U)cout<<"16U"<<endl;
	else if(src.depth()==CV_32S)cout<<"32S"<<endl;
	else if(src.depth()==CV_32F)cout<<"32F"<<endl;
	else if(src.depth()==CV_64F)cout<<"64F"<<endl;	

	if(src.channels()==1)
	{
		Scalar v = mean(src);
		cout<<"mean  : "<<v.val[0]<<endl;
		double minv,maxv;
		minMaxLoc(src, &minv, &maxv);
		cout<<"minmax: "<<minv<<","<<maxv<<endl;
	}
	else if(src.channels()==3)
	{
		Scalar v = mean(src);
		cout<<"mean  : "<<v.val[0]<<","<<v.val[1]<<","<<v.val[2]<<endl;
		
		vector<Mat> vv;
		split(src,vv);
		double minv,maxv;
		minMaxLoc(vv[0], &minv, &maxv);
		cout<<"minmax0: "<<minv<<","<<maxv<<endl;
		minMaxLoc(vv[1], &minv, &maxv);
		cout<<"minmax1: "<<minv<<","<<maxv<<endl;
		minMaxLoc(vv[2], &minv, &maxv);
		cout<<"minmax2: "<<minv<<","<<maxv<<endl;
	}
}

double RMSE(InputArray src1, InputArray src2, InputArray mask)
{
	Mat a,b;
	Mat temp = src1.getMat();
	temp.convertTo(a,CV_32F);
	temp = src2.getMat();
	temp.convertTo(b,CV_32F);
	temp = mask.getMat();
	if(temp.empty()) temp = Mat::ones(src1.size(),CV_8U);

	int count = countNonZero(temp);
	subtract(a,b,a);
	multiply(a,a,a);
	a.setTo(0, ~temp);
		
	Scalar v = sum(a);
	return sqrt(v.val[0]/(double)count);
}

void alphaBlend(InputArray src1_, InputArray src2_, double alpha, OutputArray dest)
{
	Mat src1 = src1_.getMat();
	Mat src2 = src2_.getMat();
	int T;
	Mat s1,s2;
	if(src1.channels()<=src2.channels())T=src2.type();
	else T=src1.type();
	
	if(src1.channels()==src2.channels())
	{
		s1=src1;
		s2=src2;
	}
	else if(src2.channels()==3)
	{
		cvtColor(src1,s1,CV_GRAY2BGR);
		s2=src2;
	}
	else if(src1.channels()==3)
	{
		cvtColor(src2,s2,CV_GRAY2BGR);
		s1=src1;
	}
	else if(src2.channels()==4)
	{
		cvtColor(src1,s1,CV_GRAY2BGRA);
		s2=src2;
	}
	else if(src1.channels()==4)
	{
		cvtColor(src2,s2,CV_GRAY2BGRA);
		s1=src1;
	}
	cv::addWeighted(s1, alpha, s2, 1.0-alpha, 0.0, dest);
}

void alphaBlend(const Mat& src1, const Mat& src2, const Mat& alpha, Mat& dest)
{
	int T;
	Mat s1,s2;
	if(src1.channels()<=src2.channels())T=src2.type();
	else T=src1.type();
	if(dest.empty()) dest=Mat::zeros(src1.size(),T);
	if(dest.type()!=T)dest=Mat::zeros(src1.size(),T);
	if(src1.channels()==src2.channels())
	{
		s1=src1;
		s2=src2;
	}
	else if(src2.channels()==3)
	{
		cvtColor(src1,s1,CV_GRAY2BGR);
		s2=src2;
	}
	else
	{
		cvtColor(src2,s2,CV_GRAY2BGR);
		s1=src1;
	}
	Mat a;
	if(alpha.depth()==CV_8U)
	{
		alpha.convertTo(a,CV_32F,1.0/255.0);
	}
	else if(alpha.depth()==CV_32F)
	{
		alpha.copyTo(a);
	}
	else if(alpha.depth()==CV_64F)
	{
		alpha.convertTo(a,CV_32F);
	}

	if(dest.channels()==3)
	{
		vector<Mat> ss1(3),ss2(3);
		vector<Mat> ss1f(3),ss2f(3);
		split(s1,ss1);
		split(s2,ss2);	
		for(int c=0;c<3;c++)
		{
			ss1[c].convertTo(ss1f[c],CV_32F);
			ss2[c].convertTo(ss2f[c],CV_32F);
		}
		{
			float* s1r = ss1f[0].ptr<float>(0);
			float* s2r = ss2f[0].ptr<float>(0);

			float* s1g = ss1f[1].ptr<float>(0);
			float* s2g = ss2f[1].ptr<float>(0);

			float* s1b = ss1f[2].ptr<float>(0);
			float* s2b = ss2f[2].ptr<float>(0);


			float* al = a.ptr<float>(0);
			const int size = src1.size().area()/4;
			const int sizeRem = src1.size().area()-size*4;

			const __m128 ones = _mm_set1_ps(1.0f);

			for(int i=size;i--;)
			{
				const __m128 msa = _mm_load_ps(al);
				const __m128 imsa = _mm_sub_ps(ones,msa);
				__m128 ms1 = _mm_load_ps(s1r);
				__m128 ms2 = _mm_load_ps(s2r);
				ms1 = _mm_mul_ps(ms1,msa);
				ms2 = _mm_mul_ps(ms2,imsa);
				ms1 = _mm_add_ps(ms1,ms2);
				_mm_store_ps(s1r,ms1);//store ss1f

				ms1 = _mm_load_ps(s1g);
				ms2 = _mm_load_ps(s2g);
				ms1 = _mm_mul_ps(ms1,msa);
				ms2 = _mm_mul_ps(ms2,imsa);
				ms1 = _mm_add_ps(ms1,ms2);
				_mm_store_ps(s1g,ms1);//store ss1f

				ms1 = _mm_load_ps(s1b);
				ms2 = _mm_load_ps(s2b);
				ms1 = _mm_mul_ps(ms1,msa);
				ms2 = _mm_mul_ps(ms2,imsa);
				ms1 = _mm_add_ps(ms1,ms2);
				_mm_store_ps(s1b,ms1);//store ss1f

				al+=4,s1r+=4,s2r+=4,s1g+=4,s2g+=4,s1b+=4,s2b+=4;
			}
			for(int i=0;i<sizeRem;i++)
			{
				*s1r= *al * *s1r +(1.f-*al) * *s2r;
				*s1g= *al * *s1g +(1.f-*al) * *s2g;
				*s1b= *al * *s1b +(1.f-*al) * *s2b;

				al++,s1r++,s2r++,s1g++,s2g++,s1b++,s2b++;
			}
			for(int c=0;c<3;c++)
			{
				ss1f[c].convertTo(ss1[c],CV_8U);
			}
			merge(ss1,dest);
		}
	}
	else if(dest.channels()==1)
	{
		Mat ss1f,ss2f;
		s1.convertTo(ss1f,CV_32F);
		s2.convertTo(ss2f,CV_32F);
		{
			float* s1r = ss1f.ptr<float>(0);
			float* s2r = ss2f.ptr<float>(0);
			float* al = a.ptr<float>(0);
			const int size = src1.size().area()/4;
			const int nn = src1.size().area() - size*4;
			const __m128 ones = _mm_set1_ps(1.0f);
			for(int i=size;i--;)
			{
				const __m128 msa = _mm_load_ps(al);
				const __m128 imsa = _mm_sub_ps(ones,msa);
				__m128 ms1 = _mm_load_ps(s1r);
				__m128 ms2 = _mm_load_ps(s2r);
				ms1 = _mm_mul_ps(ms1,msa);
				ms2 = _mm_mul_ps(ms2,imsa);
				ms1 = _mm_add_ps(ms1,ms2);
				_mm_store_ps(s1r,ms1);//store ss1f

				al+=4,s1r+=4,s2r+=4;
			}
			for(int i=nn;i--;)
			{
				*s1r = *al * *s1r + (1.0f-*al)* *s2r;
				al++,s1r++,s2r++;
			}
			if(src1.depth()==CV_32F)
				ss1f.copyTo(dest);
			else
				ss1f.convertTo(dest,src1.depth());
		}
	}
}

void alphaBlendSSE_8u(const Mat& src1, const Mat& src2, const Mat& alpha, Mat& dest)
{
	if(dest.empty())dest.create(src1.size(),CV_8U);

	const int imsize = (src1.size().area()/16);
	uchar* s1 = (uchar*)src1.data;
	uchar* s2 = (uchar*)src2.data;
	uchar* a = (uchar*)alpha.data;
	uchar* d = dest.data;

	const __m128i zero = _mm_setzero_si128();
	const __m128i amax = _mm_set1_epi8(char(255));
	int i=0;
	if(s1==d)
	{
		for(;i<imsize;++i)
		{
			__m128i ms1h = _mm_load_si128((__m128i*)(s1));
			__m128i ms2h = _mm_load_si128((__m128i*)(s2));
			__m128i mah = _mm_load_si128((__m128i*)(a));
			__m128i imah = _mm_sub_epi8(amax,mah);

			__m128i ms1l = _mm_unpacklo_epi8(ms1h, zero);
			ms1h = _mm_unpackhi_epi8(ms1h, zero);

			__m128i ms2l = _mm_unpacklo_epi8(ms2h, zero);
			ms2h = _mm_unpackhi_epi8(ms2h, zero);

			__m128i mal = _mm_unpacklo_epi8(mah, zero);
			mah = _mm_unpackhi_epi8(mah, zero);

			__m128i imal = _mm_unpacklo_epi8(imah, zero);
			imah = _mm_unpackhi_epi8(imah, zero);

			ms1l = _mm_mullo_epi16(ms1l,mal);
			ms2l = _mm_mullo_epi16(ms2l,imal);
			ms1l = _mm_add_epi16(ms1l,ms2l);
			//ms1l = _mm_srli_epi16(ms1l,8);
			ms1l = _mm_srai_epi16(ms1l,8);

			ms1h = _mm_mullo_epi16(ms1h,mah);
			ms2h = _mm_mullo_epi16(ms2h,imah);
			ms1h = _mm_add_epi16(ms1h,ms2h);
			//ms1h = _mm_srli_epi16(ms1h,8);
			ms1h = _mm_srai_epi16(ms1h,8);

			_mm_stream_si128((__m128i*)s1,_mm_packs_epi16(ms1l,ms1h));

			s1+=16;
			s2+=16;
			a+=16;
		}
	}
	else
	{
		for(;i<imsize;++i)
		{
			__m128i ms1h = _mm_load_si128((__m128i*)(s1));
			__m128i ms2h = _mm_load_si128((__m128i*)(s2));
			__m128i mah = _mm_load_si128((__m128i*)(a));
			__m128i imah = _mm_sub_epi8(amax,mah);

			__m128i ms1l = _mm_unpacklo_epi8(ms1h, zero);
			ms1h = _mm_unpackhi_epi8(ms1h, zero);

			__m128i ms2l = _mm_unpacklo_epi8(ms2h, zero);
			ms2h = _mm_unpackhi_epi8(ms2h, zero);

			__m128i mal = _mm_unpacklo_epi8(mah, zero);
			mah = _mm_unpackhi_epi8(mah, zero);

			__m128i imal = _mm_unpacklo_epi8(imah, zero);
			imah = _mm_unpackhi_epi8(imah, zero);

			ms1l = _mm_mullo_epi16(ms1l,mal);
			ms2l = _mm_mullo_epi16(ms2l,imal);
			ms1l = _mm_add_epi16(ms1l,ms2l);
			//ms1l = _mm_srli_epi16(ms1l,8);
			ms1l = _mm_srai_epi16(ms1l,8);

			ms1h = _mm_mullo_epi16(ms1h,mah);
			ms2h = _mm_mullo_epi16(ms2h,imah);
			ms1h = _mm_add_epi16(ms1h,ms2h);
			//ms1h = _mm_srli_epi16(ms1h,8);
			ms1h = _mm_srai_epi16(ms1h,8);

			_mm_store_si128((__m128i*)d,_mm_packs_epi16(ms1l,ms1h));

			s1+=16;
			s2+=16;
			a+=16;
			d+=16;
		}
	}

	{
		uchar* s1 = (uchar*)src1.data;
		uchar* s2 = (uchar*)src2.data;
		uchar* a = (uchar*)alpha.data;
		uchar* d = dest.data;
		for(int n=i*16;n<src1.size().area();n++)
		{
			d[n] = (a[n]*s1[n] + (255-a[n])*s2[n])>>8;
		}
	}
}
static void alphablend1(Mat& src1, Mat& src2,Mat& alpha, Mat& dest)
{
	if(dest.empty())dest.create(src1.size(),CV_8U);
	const int imsize = (src1.size().area());
	uchar* s1 = src1.data;
	uchar* s2 = src2.data;
	uchar* a = alpha.data;
	uchar* d = dest.data;
	const double div = 1.0/255;
	for(int i=0;i<imsize;i++)
	{
		d[i]=(uchar)((a[i]*s1[i]+(255-a[i])*s2[i])*div + 0.5);
	}
}
static void alphablend2(Mat& src1, Mat& src2,Mat& alpha, Mat& dest)
{
	if(dest.empty())dest.create(src1.size(),CV_8U);
	const int imsize = (src1.size().area());
	uchar* s1 = src1.data;
	uchar* s2 = src2.data;
	uchar* a = alpha.data;
	uchar* d = dest.data;
//	const double div = 1.0/255;
	for(int i=0;i<imsize;i++)
	{
		d[i]=(a[i]*s1[i]+(255-a[i])*s2[i])>>8;
	}
}

void guiAlphaBlend(InputArray src1_, InputArray src2_)
{
	Mat& src1 = src1_.getMat();
	Mat& src2 = src2_.getMat();
	Mat s1,s2;

	if(src1.depth()==CV_8U || src1.depth()==CV_32F)
	{
		if(src1.channels()==1)cvtColor(src1,s1,CV_GRAY2BGR);
		else s1 = src1;
		if(src2.channels()==1)cvtColor(src2,s2,CV_GRAY2BGR);
		else s2 = src2;
	}
	else
	{
		Mat ss1,ss2;
		src1.convertTo(ss1,CV_32F);
		src2.convertTo(ss2,CV_32F);

		if(src1.channels()==1)cvtColor(ss1,s1,CV_GRAY2BGR);
		else s1 = ss1.clone();
		if(src2.channels()==1)cvtColor(ss2,s2,CV_GRAY2BGR);
		else s2 = ss2.clone();
	}
	namedWindow("alphaBlend");
	int a = 0;
	createTrackbar("a","alphaBlend",&a,100);
	int key = 0;
	Mat show;
	while(key!='q')
	{	
		addWeighted(s1,1.0-a/100.0,s2,a/100.0,0.0,show);

		if(show.depth()==CV_8U)
			imshow("alphaBlend",show);
		else
		{
			double minv,maxv;
			minMaxLoc(show, &minv, &maxv);

			Mat s;
			if(maxv<=255)
				show.convertTo(s,CV_8U);
			else
				show.convertTo(s,CV_8U,255/maxv);

			imshow("alphaBlend",s);
		}
		key = waitKey(1);
		if(key=='f')
		{
			a = (a > 0) ? 0 : 100;
			setTrackbarPos("a","alphaBlend",a);
		}
		if(key=='i')
		{
			showMatInfo(src1,"========src1========");
			cout<<endl;
			showMatInfo(src2,"========src2========");
		}
	}
	destroyWindow("alphaBlend");
}


void warpShift_(Mat& src, Mat& dst, int shiftx, int shifty)
{
	Mat dest =Mat::zeros(src.size(),src.type());
	
	int width = src.cols;
	int height = src.rows;
	if(shiftx>=0 &&shifty>=0)
	{
		Mat d = dest(Rect( shiftx, shifty, width-shiftx, height-shifty ));
		Mat(src(Rect(0, 0, width-shiftx, height-shifty))).copyTo(d);
	}
	else if(shiftx>=0 &&shifty<0)
	{
		Mat d = dest(Rect( shiftx, 0, width-shiftx, height+shifty));
		Mat(src(Rect( 0, -shifty, width-shiftx, height+shifty))).copyTo(d);
	}
	else if(shiftx<0 &&shifty<0)
	{
		Mat d = dest(Rect( 0, 0, width+shiftx, height+shifty));
		Mat(src(Rect( -shiftx, -shifty, width+shiftx, height+shifty))).copyTo(d);
	}
	else if(shiftx<0 &&shifty>=0)
	{
		Mat d = dest(Rect( 0, shifty, width+shiftx, height-shifty));
		Mat(src(Rect( -shiftx, 0, width+shiftx, height-shifty))).copyTo(d);
	}
	dest.copyTo(dst);
}

void warpShift_(Mat& src, Mat& dest, int shiftx, int shifty, int borderType)
{
	if(dest.empty())dest=Mat::zeros(src.size(),src.type());
	

	int width = src.cols;
	int height = src.rows;
	if(shiftx>=0 &&shifty>=0)
	{
		Mat im; copyMakeBorder(src,im,shifty,0,shiftx,0, borderType);
		Mat(im(Rect(0, 0, width, height))).copyTo(dest);
	}
	else if(shiftx>=0 &&shifty<0)
	{
		Mat im; copyMakeBorder(src,im,0,-shifty,shiftx,0, borderType);
		Mat(im(Rect(0, -shifty, width, height))).copyTo(dest);
	}
	else if(shiftx<0 &&shifty<0)
	{
		Mat im; copyMakeBorder(src,im, 0, -shifty, 0, -shiftx, borderType);
		Mat(im(Rect(-shiftx, -shifty, width, height))).copyTo(dest);
	}
	else if(shiftx<0 &&shifty>=0)
	{
		Mat im; copyMakeBorder(src,im,shifty, 0, 0, -shiftx, borderType);
		Mat(im(Rect(-shiftx, 0, width, height))).copyTo(dest);
	}
}

void warpShift(InputArray src_, OutputArray dest_, int shiftx, int shifty, int borderType)
{
	Mat src = src_.getMat();
	if(dest_.empty() ||dest_.size()!=src_.size() || dest_.type() != src_.type()) dest_.create( src.size(), src.type() );
	Mat dest = dest_.getMat();

	if(borderType<0)
		warpShift_(src,dest,shiftx,shifty);
	else
		warpShift_(src,dest,shiftx,shifty,borderType);
}

void CalcTime::start()
{
	pre = getTickCount();
}

void CalcTime::restart()
{
	start();
}

void CalcTime::lap(string message)
{
	string v = message + format(" %f",getTime());
	switch(timeMode)
	{
	case TIME_NSEC:
		v += " NSEC";
		break;
	case TIME_SEC:
		v += " SEC";
		break;
	case TIME_MIN:
		v += " MIN";
		break;
	case TIME_HOUR:
		v += " HOUR";
		break;

	case TIME_MSEC:
	default:
		v += " msec";
		break;
	}


	lap_mes.push_back(v);
	restart();
}
void CalcTime:: show()
{
	getTime();

	int mode = timeMode;
	if(timeMode==TIME_AUTO)
	{
		mode = autoMode;
	}

	switch(mode)
	{
	case TIME_NSEC:
		cout<< mes<< ": "<<cTime<<" nsec"<<endl;
		break;
	case TIME_SEC:
		cout<< mes<< ": "<<cTime<<" sec"<<endl;
		break;
	case TIME_MIN:
		cout<< mes<< ": "<<cTime<<" minute"<<endl;
		break;
	case TIME_HOUR:
		cout<< mes<< ": "<<cTime<<" hour"<<endl;
		break;

	case TIME_MSEC:
	default:
		cout<<mes<< ": "<<cTime<<" msec"<<endl;
		break;
	}
}

void CalcTime:: show(string mes)
{
	getTime();

	int mode = timeMode;
	if(timeMode==TIME_AUTO)
	{
		mode = autoMode;
	}

	switch(mode)
	{
	case TIME_NSEC:
		cout<< mes<< ": "<<cTime<<" nsec"<<endl;
		break;
	case TIME_SEC:
		cout<< mes<< ": "<<cTime<<" sec"<<endl;
		break;
	case TIME_MIN:
		cout<< mes<< ": "<<cTime<<" minute"<<endl;
		break;
	case TIME_HOUR:
		cout<< mes<< ": "<<cTime<<" hour"<<endl;
		break;
	case TIME_DAY:
		cout<< mes<< ": "<<cTime<<" day"<<endl;
	case TIME_MSEC:
		cout<<mes<< ": "<<cTime<<" msec"<<endl;
		break;
	default:
		cout<<mes<< ": error"<<endl;
		break;
	}
}

int CalcTime::autoTimeMode()
{
	if(cTime>60.0*60.0*24.0)
	{
		return TIME_DAY;
	}
	else if(cTime>60.0*60.0)
	{
		return TIME_HOUR;
	}
	else if(cTime>60.0)
	{
		return TIME_MIN;
	}
	else if(cTime>1.0)
	{
		return TIME_SEC;
	}
	else if(cTime>1.0/1000.0)
	{
		return TIME_MSEC;
	}
	else
	{

		return TIME_NSEC;
	}
}
double CalcTime:: getTime()
{
	cTime = (getTickCount()-pre)/(getTickFrequency());

	int mode=timeMode;
	if(mode==TIME_AUTO)
	{
		mode = autoTimeMode();
		autoMode=mode;
	}

	switch(mode)
	{
	case TIME_NSEC:
		cTime*=1000000.0;
		break;
	case TIME_SEC:
		cTime*=1.0;
		break;
	case TIME_MIN:
		cTime /=(60.0);
		break;
	case TIME_HOUR:
		cTime /=(60*60);
		break;
	case TIME_DAY:
		cTime /=(60*60*24);
		break;
	case TIME_MSEC:
	default:
		cTime *=1000.0;
		break;
	}
	return cTime;
}
void CalcTime::setMessage(string src)
{
	mes=src;
}
void CalcTime:: setMode(int mode)
{
	timeMode = mode;
}

void CalcTime::init(string message, int mode, bool isShow)
{
	_isShow = isShow;
	timeMode = mode;

	setMessage(message);
	start();
}


CalcTime::CalcTime()
{
	init("time ", TIME_AUTO, true);
}

CalcTime::CalcTime(string message,int mode,bool isShow)
{
	init(message,mode,isShow);
}
CalcTime::~CalcTime()
{
	getTime();
	if(_isShow)	show();
	if(lap_mes.size()!=0)
	{
		for(int i=0;i<lap_mes.size();i++)
		{
			cout<<lap_mes[i]<<endl;
		}
	}
}


