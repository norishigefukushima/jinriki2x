#include "util.h"
using namespace std;


void guiBilateralUpsample(InputArray srcimage, OutputArray dest, int resizeFactor)
{
	string windowName = "bilateral";
	namedWindow(windowName);
	Mat src = srcimage.getMat();

	int alpha = 0; createTrackbar("a",windowName, &alpha, 100);

	int r = 3; createTrackbar("r",windowName, &r, 30);
	int sc = 30; createTrackbar("sigma_color",windowName, &sc, 255);
	int ss = 30; createTrackbar("sigma_space",windowName, &ss, 255);
	int iter = 3; createTrackbar("iteration",windowName, &iter, 10);

	int key = 0;
	while(key!='q')
	{
		Mat srctemp;
		src.copyTo(srctemp);
		for(int i=0;i<iter;i++)
		{
			Mat tmp;
			bilateralFilter(srctemp, tmp, 2*r+1, sc, ss, BORDER_REPLICATE);
			tmp.copyTo(srctemp);
		}

		alphaBlend(srcimage, srctemp, alpha/100.0, srctemp);
		

		resize(srctemp, dest, Size(src.cols*resizeFactor, src.rows*resizeFactor), 0,0, INTER_CUBIC);

		imshow(windowName, dest);
		key = waitKey(30);
		if(key=='f')
		{
			alpha = (alpha != 0) ? 100:0;
			setTrackbarPos("a", windowName, alpha);
		}
	}
	destroyWindow(windowName);
}

void guiWeightedModeUpsample(InputArray srcimage, OutputArray dest, int resizeFactor)
{
	string windowName = "weighted mode";
	namedWindow(windowName);
	Mat src = srcimage.getMat();

	int alpha = 0; createTrackbar("a",windowName, &alpha, 100);
	int sw = 0; createTrackbar("sw",windowName, &sw, 1);
	int sw2 = 0; createTrackbar("sw2",windowName, &sw2, 1);

	int r = 3; createTrackbar("r",windowName, &r, 30);
	int sc = 40; createTrackbar("sigma_color",windowName, &sc, 255);
	int ss = 30; createTrackbar("sigma_space",windowName, &ss, 255);
	int sb = 10; createTrackbar("sigma_bin",windowName, &sb, 255);
	int iter = 2; createTrackbar("iteration",windowName, &iter, 10);

	int key = 0;
	
	while(key!='q')
	{
		Mat srctemp;
		{
			Mat med;
			medianBlur(srcimage, med,1);
			
			src.copyTo(srctemp);
			CalcTime t;
			for(int i=0;i<iter;i++)
			{
				Mat tmp = srctemp.clone();
				weightedModeFilter(srctemp, med, tmp, r, ss, sc, 2, sb);
				tmp.copyTo(srctemp);
			}

			alphaBlend(srcimage, srctemp, alpha/100.0, srctemp);

			if(sw==0) hqx(srctemp, dest, resizeFactor);
			else resize(srctemp, dest, Size(src.cols*resizeFactor, src.rows*resizeFactor), 0,0, INTER_CUBIC);

			
			
			//Mat dest2 = dest.getMat().clone();
			//guiCoherenceEnhancingShockFilter(dest2,dest);
		}
		if(sw2!=0)
		{
			Mat a = dest.getMat();
			blurRemoveMinMax(a,a,2);
			a.copyTo(dest);
		}
		
		imshow(windowName, dest);
		key = waitKey(30);
		if(key=='f')
		{
			alpha = (alpha != 0) ? 0:100;
			setTrackbarPos("a", windowName, alpha);
		}
	}
	destroyWindow(windowName);
}

int main(int argc, char** argv)
{
	Mat src = imread("images/miku_small_noisy.jpg");
	Mat refNN = imread("images/miku_small_waifu2x.png");
	Mat cubic;
	Mat bilateral;
	Mat weightedmode;
	
	resize(src, cubic, Size(src.cols*2, src.rows*2), 0,0, INTER_CUBIC);

	//guiBilateralUpsample(src, bilateral, 2);
	guiWeightedModeUpsample(src, weightedmode, 2);

	//guiAlphaBlend(cubic, refNN);
	//guiAlphaBlend(bilateral, refNN);
	guiAlphaBlend(weightedmode, refNN);
	
	
	return 0;
}