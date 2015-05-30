#include "util.h"

void iterativeBackProjectionDeblurGaussian(InputArray src, OutputArray dest, const Size ksize, const double sigma, const double lambda, const int iteration)
{
	Mat srcf;
	Mat destf;
	Mat subf;
	Mat srcm = src.getMat();
	srcm.convertTo(srcf,CV_32FC3);
	srcm.convertTo(destf,CV_32FC3);
	Mat bdest;

	double maxe = DBL_MAX;

	int i;
	for(i=0;i<iteration;i++)
	{
		GaussianBlur(destf,bdest,ksize,sigma);
		subtract(srcf,bdest,subf);
		double e = norm(subf);

		GaussianBlur(subf,subf,ksize,sigma);

		destf += lambda*subf;

		//printf("%f\n",e);
		if(i!=0)
		{
			if(maxe>e)
			{
				maxe=e;
			}
			else break;
		}
		//if(isWrite)
		//{
		//	destf.convertTo(dest,CV_8UC3);
		//	imwrite(format("B%03d.png",i),dest);
		//}
	}
	printf("%d\n",i);
	destf.convertTo(dest,CV_8UC3);
}


void guiIterativeBackProjectionDeblurGaussian(InputArray src, OutputArray dest)
{	
	string wname = "iterative back projection: Gaussian";
	namedWindow(wname);

	Mat bl;
	
	int p1=1;
	int p2=1;
	
	int alpha=50;

	
	
	int r = 2; createTrackbar("input blur r",wname,&r,20);
	int sigma = 50; createTrackbar("input blur sigma",wname,&sigma,255);
	int lambda=10; createTrackbar("lambda/10.0",wname,&lambda,100);
	int iter=1; createTrackbar("iter",wname,&iter,50);

	

	//createTrackbar("alpha",wname,&alpha,100);
	int key=0;
	
	Mat src2; 
	src.getMat().copyTo(src2);
	Mat show;
	while(key!='q')
	{
		if(r==0)src.getMat().copyTo(bl);
		else GaussianBlur(src2,bl,Size(2*r+1,2*r+1),sigma);

		{
//			CalcTime t(wname);
			iterativeBackProjectionDeblurGaussian(src, show, Size(2*r+1,2*r+1), sigma/10.0, lambda/10.0, iter);
			coherenceEnhancingShockFilter(bl,show,2*p1+1,2*p2+1,alpha/100.0,iter);
		}
		show.convertTo(dest,CV_8U);
		imshow(wname,dest);
		key=waitKey(1);
	}
}


/*
void iterativeBackProjectionDeblurBilateral(const Mat& src, Mat& dest, const Size ksize, const double sigma_color, const double sigma_space, const double lambda, const int iteration)
{
	Mat srcf;
	Mat destf;
	Mat subf;
	src.convertTo(srcf,CV_32FC3);
	src.convertTo(destf,CV_32FC3);
	Mat bdest;

	double maxe = DBL_MAX;

	int i;
	for(i=0;i<iteration;i++)
	{
		GaussianBlur(destf,bdest,ksize,sigma_space);

		subtract(srcf,bdest,subf);
		
		//normarize from 0 to 255 for joint birateral filter (range 0 to 255)
		double minv, maxv;
		minMaxLoc(subf,&minv,&maxv);
		subtract(subf,Scalar(minv,minv,minv),subf);
		multiply(subf,Scalar(2,2,2),subf);
		
		jointBilateralFilter(subf,destf,subf,ksize,sigma_color,sigma_space);

		multiply(subf,Scalar(0.5,0.5,0.5),subf);
		add(subf,Scalar(minv,minv,minv),subf);
		
		double e = norm(subf);
		
		//imshow("a",subf);
		multiply(subf,Scalar(lambda,lambda,lambda),subf);
		
		add(destf,subf,destf);
		//destf += ((float)lambda)*subf;

		//printf("%f\n",e);
	//	if(i!=0)
	//	{
	//		if(maxe>e)
	//		{
	//			maxe=e;
	//		}
	//		else break;
	//	}
		//if(isWrite)
		//{
		//	destf.convertTo(dest,CV_8UC3);
		//	imwrite(format("B%03d.png",i),dest);
		//}
	}
	destf.convertTo(dest,CV_8UC3);
}
*/