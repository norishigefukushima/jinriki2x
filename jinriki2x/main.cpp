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

//X. Zhang, X. Wu, gImage interpolation by adaptive 2D autoregressive modeling and soft-decision estimationh, IEEE Trans. on Image Processing, vol. 17, no. 6, pp. 887-896, June 2008.
void SAI(InputArray src, OutputArray dest)
{
	Mat temp;
	cvtColor(src, temp, CV_BGR2YCrCb);
	vector<Mat> v;split(temp, v);

	imwrite("temp.pgm",v[0]);

	char cmd[64];
	sprintf(cmd,"ARInterpolation.exe temp.pgm tout.pgm");
	//cout<<cmd<<endl;
	FILE* sai = _popen(cmd,"w");
	fflush(sai);
	_pclose(sai);

	Mat y = imread("tout.pgm",0);
	Mat cr,cb;
	jointBilateralUpsample(v[1],y,cr, 30,0.6);
	jointBilateralUpsample(v[2],y,cb, 30,0.6);
	vector<Mat> d(3);
	d[0]=y;
	d[1]=cr;
	d[2]=cb;
	merge(d,dest);
	cvtColor(dest, dest, CV_YCrCb2BGR);
}


void guiWeightedModeUpsample(InputArray srcimage, OutputArray dest, int resizeFactor, InputArray ref)
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

	int iter2 = 2; createTrackbar("iteration2",windowName, &iter2, 10);

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

			//upsampling function
			if(sw==0) hqx(srctemp, dest, resizeFactor);
			else SAI(srctemp, dest);
			//else resize(srctemp, dest, Size(src.cols*resizeFactor, src.rows*resizeFactor), 0,0, INTER_LANCZOS4);
			
		}
		//shock filter
		if(sw2!=0)
		{
			Mat a = dest.getMat();
			//blurRemoveMinMax(a,a,2);

			iterativeBackProjectionDeblurGaussian(a, a, Size(9,9), 3, 0.2, iter2);
			a.copyTo(dest);
			
		}
		//blending referece image for debug
		alphaBlend(ref, dest, alpha/100.0, dest);
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

void guiNLMUpsample(InputArray srcimage, OutputArray dest, int resizeFactor, InputArray ref)
{
	string windowName = "weighted mode";
	namedWindow(windowName);
	Mat src = srcimage.getMat();

	int alpha = 0; createTrackbar("a",windowName, &alpha, 100);
	int sw = 0; createTrackbar("sw",windowName, &sw, 1);
	int sw2 = 0; createTrackbar("sw2",windowName, &sw2, 1);

	int tr = 0; createTrackbar("tr",windowName, &tr, 10);
	int sr = 3; createTrackbar("sr",windowName, &sr, 30);
	int h = 100; createTrackbar("h/10",windowName, &h, 255);
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
				if(srcimage.channels()==1) fastNlMeansDenoising(srctemp, tmp, h/10.f, 2*tr+1, 2*sr+1);
				else fastNlMeansDenoisingColored(srctemp, tmp, (float)h/10.f,(float)h/10.f, 2*tr+1, 2*sr+1);

				tmp.copyTo(srctemp);
			}

			//upsampling function
			if(sw==0) hqx(srctemp, dest, resizeFactor);
			else resize(srctemp, dest, Size(src.cols*resizeFactor, src.rows*resizeFactor), 0,0, INTER_LANCZOS4);
			
		}
		//shock filter
		if(sw2!=0)
		{
			Mat a = dest.getMat();
			blurRemoveMinMax(a,a,2);
			a.copyTo(dest);
		}
		//blending referece image for debug
		alphaBlend(ref, dest, alpha/100.0, dest);
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


void CUBIC(InputArray src, OutputArray dest)
{
	int odd_w =0;
	int odd_h =0;
	if(src.size().width%2!=0)odd_w=1;
	if(src.size().height%2!=0)odd_h=1;
	Mat im;
	copyMakeBorder(src, im,4,4+odd_h,4,4+odd_w,BORDER_REPLICATE);
	Mat dst;
	//resize(im, dst,Size(im.cols*2,im.rows*2),0,0,INTER_CUBIC);
	resize(im, dst,Size(im.cols*2,im.rows*2),0,0,INTER_LINEAR);

	warpShift(dst,dst,-0,-0,BORDER_REPLICATE);
	Mat(dst(Rect(4,4,src.size().width*2,src.size().height*2))).copyTo(dest);
}

int main(int argc, char** argv)
{
	//grayscale
	Mat src = imread("images/miku_small_noisy.jpg");
	//Mat src = imread("images/miku_small.png",0);
	Mat refNN = imread("images/miku_small_noisy_waifu2x.png");
	//color
	//Mat src = imread("images/miku_small_noisy.jpg");
	//Mat refNN = imread("images/miku_small_noisy_waifu2x.png");

	Mat dst;
	Mat cubic;
	
	
	Mat bilateral;
	Mat weightedmode;
	
	resize(src, cubic, Size(src.cols*2, src.rows*2), 0,0, INTER_CUBIC);

	//guiBilateralUpsample(src, bilateral, 2);
	guiWeightedModeUpsample(src, weightedmode, 2, refNN);
	//guiNLMUpsample(src, weightedmode, 2, refNN);

	//guiAlphaBlend(cubic, refNN);
	//guiAlphaBlend(bilateral, refNN);
	//guiAlphaBlend(weightedmode, refNN);
	
	
	return 0;
}