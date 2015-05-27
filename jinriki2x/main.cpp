#include "util.h"
using namespace std;

int main(int argc, char** argv)
{
	Mat src = imread("images/miku_small_noisy.jpg");
	Mat refNN = imread("images/miku_small_waifu2x.png");
	Mat cubic;
	
	resize(src, cubic, Size(src.cols*2, src.rows*2), 0,0, INTER_CUBIC);
	guiAlphaBlend(cubic, refNN);

	return 0;
}