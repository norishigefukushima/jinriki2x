#include <opencv2/opencv.hpp>
using namespace cv;

#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_contrib"CV_VERSION_NUMBER"d.lib")
#else
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_contrib"CV_VERSION_NUMBER".lib")
#endif

void weightedMedianFilter(Mat& src, Mat& guide, Mat& dst, int r, int truncate, double sig_s, double sig_c, int metric, int sig_bin);
void weightedModeFilter(Mat& src, Mat& guide, Mat& dst, int r, double sig_s, double sig_c, int metric, int sig_bin);

void hqx(InputArray src, OutputArray dest, const int scaleBy);//scaleBy:2-4

void coherenceEnhancingShockFilter(InputArray src, OutputArray dest,int sigma, int str_sigma, double blend, int iter);
void guiCoherenceEnhancingShockFilter(InputArray src, OutputArray dest);
enum
{
	TIME_AUTO=0,
	TIME_NSEC,
	TIME_MSEC,
	TIME_SEC,
	TIME_MIN,
	TIME_HOUR,
	TIME_DAY
};

class CalcTime
{
	int64 pre;
	string mes;

	int timeMode;

	double cTime;
	bool _isShow;

	int autoMode;
	int autoTimeMode();
	vector<string> lap_mes;
public:
	
	void start();
	void setMode(int mode);
	void setMessage(string src);
	void restart();
	double getTime();
	void show();
	void show(string message);
	void lap(string message);
	void init(string message, int mode, bool isShow);

	CalcTime(string message, int mode=TIME_AUTO, bool isShow=true);
	CalcTime();

	~CalcTime();
};

void showMatInfo(InputArray src_, string name="");
void guiAlphaBlend(InputArray src1_, InputArray src2_);
void alphaBlend(InputArray src1_, InputArray src2_, double alpha, OutputArray dest);
void warpShift(InputArray src_, OutputArray dest_, int shiftx, int shifty, int borderType);
