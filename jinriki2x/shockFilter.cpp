#include "util.h"


void coherenceEnhancingShockFilter(InputArray src, OutputArray dest,int sigma, int str_sigma, double blend, int iter)
{
	str_sigma = min(31,str_sigma);
	src.getMat().copyTo(dest);
	
	
	for(int i=0;i<iter;i++)
	{
		Mat gray;
		if(src.channels()==3)cvtColor(dest,gray,CV_BGR2GRAY);
		else gray = dest.getMat();

		Mat eigen;
		if(gray.type()==CV_8U ||gray.type()==CV_32F || gray.type()==CV_64F)
			cornerEigenValsAndVecs(gray,eigen,str_sigma,3);
		else
		{
			Mat grayf;gray.convertTo(grayf,CV_32F);
			cornerEigenValsAndVecs(grayf,eigen,str_sigma,3);
		}
			
		vector<Mat> esplit(6);
		split(eigen,esplit);
		Mat x=esplit[2];
		Mat y=esplit[3];
		Mat gxx;
		Mat gyy;
		Mat gxy;
		Sobel(gray,gxx,CV_32F,2,0,sigma);
		Sobel(gray,gyy,CV_32F,0,2,sigma);
		Sobel(gray,gxy,CV_32F,1,1,sigma);

		Mat gvv = x.mul(x).mul(gxx)  + 2*x.mul(y).mul(gxy) + y.mul(y).mul(gyy);
		
		Mat mask;
		compare(gvv,0,mask,cv::CMP_LT);
		
		Mat di,ero;
		erode(dest,ero,Mat());
		dilate(dest,di,Mat());
		di.copyTo(ero,mask);
		addWeighted(dest,blend,ero,1.0-blend,0.0,dest);
	}
}


void guiCoherenceEnhancingShockFilter(InputArray src, OutputArray dest)
{	
	Mat bl;
	
	int p1=1;
	int p2=1;
	int iter=1;
	int alpha=50;

	string wname = "shock filter";
	namedWindow(wname);
	int sigma = 5;
	int r = 2;
	createTrackbar("input blur sigma",wname,&sigma,20);
	createTrackbar("input blur r",wname,&r,20);

	createTrackbar("sobel_sigma",wname,&p1,20);
	createTrackbar("eigen_sigma",wname,&p2,20);
	createTrackbar("iter",wname,&iter,50);
	createTrackbar("alpha",wname,&alpha,100);
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
			coherenceEnhancingShockFilter(bl,show,2*p1+1,2*p2+1,alpha/100.0,iter);
		}
		show.convertTo(dest,CV_8U);
		imshow(wname,dest);
		key=waitKey(1);
	}
}
