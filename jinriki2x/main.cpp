#include "util.h"
using namespace std;


void guiBilateralUpsample(InputArray srcimage, OutputArray dest, int resizeFactor)
{
	string windowName = "bilateral";
	namedWindow(windowName);
	Mat src = srcimage.getMat();

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

		resize(srctemp, dest, Size(src.cols*resizeFactor, src.rows*resizeFactor), 0,0, INTER_CUBIC);
		imshow(windowName, dest);
		key = waitKey(30);
	}
	destroyWindow(windowName);
}

int main(int argc, char** argv)
{
	Mat src = imread("images/miku_small_noisy.jpg");
	Mat refNN = imread("images/miku_small_waifu2x.png");
	Mat cubic;
	Mat bilateral;
	
	resize(src, cubic, Size(src.cols*2, src.rows*2), 0,0, INTER_CUBIC);

	guiBilateralUpsample(src, bilateral, 2);

	
	
	guiAlphaBlend(cubic, refNN);
	guiAlphaBlend(bilateral, refNN);

	return 0;
}