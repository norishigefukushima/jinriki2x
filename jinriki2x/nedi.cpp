#include "util.h"

 void NEDI_1st_x2(Mat& src, Mat& dest)
 { 
	 CV_Assert(src.cols%2==0 && src.rows%2==0);
	 dest = Mat::zeros(Size(src.cols*2,src.rows*2), CV_64F);
	 const int th = 8;

	 //upsampling: 0-th copy
	 for(int j=0;j<src.rows;j++)
	 {
		 double* s=src.ptr<double>(j);
		 double* d=dest.ptr<double>(2*j);
		 for(int i=0;i<src.cols;i++)
		 {
			 d[2*i]=s[i];
		 }
	 }

	 Matx44d C;
	 Matx41d y;
	 for(int j=3 ;j<dest.rows-3;j+=2)
	 {
		 for(int i=3; i<dest.cols-3;i+=2)
		 {
			 /*
			 C(0,0)=dest.at<double>(j-3,i-3);
			 C(0,1)=dest.at<double>(j-3,i+1);
			 C(0,2)=dest.at<double>(j+1,i-3);
			 C(0,3)=dest.at<double>(j+1,i+1);

			 C(1,0)=dest.at<double>(j-3,i+1);
			 C(1,1)=dest.at<double>(j-3,i+3);
			 C(1,2)=dest.at<double>(j+1,i+1);
			 C(1,3)=dest.at<double>(j+1,i+3);

			 C(2,0)=dest.at<double>(j-1,i-3);
			 C(2,1)=dest.at<double>(j-1,i+1);
			 C(2,2)=dest.at<double>(j+3,i-3);
			 C(2,3)=dest.at<double>(j+3,i+1);

			 C(3,0)=dest.at<double>(j-1,i-3);
			 C(3,1)=dest.at<double>(j-1,i+1);
			 C(3,2)=dest.at<double>(j+3,i-3);
			 C(3,3)=dest.at<double>(j+3,i+1);
			 */
			 /*
			 C(0,0)=dest.at<double>(j-3,i-3);
			 C(0,1)=dest.at<double>(j-3,i+1);
			 C(0,2)=dest.at<double>(j+1,i-3);
			 C(0,3)=dest.at<double>(j+1,i+1);

			 C(1,0)=dest.at<double>(j-3,i+1);
			 C(1,1)=dest.at<double>(j-3,i+3);
			 C(1,2)=dest.at<double>(j+1,i+1);
			 C(1,3)=dest.at<double>(j+1,i+3);

			 C(2,0)=dest.at<double>(j-1,i-3);
			 C(2,1)=dest.at<double>(j-1,i+1);
			 C(2,2)=dest.at<double>(j+3,i-3);
			 C(2,3)=dest.at<double>(j+3,i+1);

			 C(3,0)=dest.at<double>(j-1,i-3);
			 C(3,1)=dest.at<double>(j-1,i+1);
			 C(3,2)=dest.at<double>(j+3,i-3);
			 C(3,3)=dest.at<double>(j+3,i+1);
			 */

			 y(0)=dest.at<double>(j-1,i-1);
			 y(1)=dest.at<double>(j-1,i+1);
			 y(2)=dest.at<double>(j+1,i-1);
			 y(3)=dest.at<double>(j+1,i+1);
			

			 const double ave = (y(0)+y(1)+y(2)+y(3))*0.25;
			 const double var = (
				 +(y(0)-ave)*(y(0)-ave)
				 +(y(1)-ave)*(y(1)-ave)
				 +(y(2)-ave)*(y(2)-ave)
				 +(y(3)-ave)*(y(3)-ave)
				 )*0.25;
				 
			 Mat a = Mat::ones(4,1,CV_64F)*0.25;

			 Matx44d CtC=Mat(C.t()*C);
			 if(determinant(CtC)!=0 && var>th) Mat(CtC.inv()*(C.t()*y)).copyTo(a);
			 dest.at<double>(j,i)=a.dot(y);
		 }
	 }
	 for(int j=3;j<dest.rows-3;j++)
	 {
		 int offset = (j%2==0) ? 1:0;
		 for(int i=4;i<dest.cols-3;i+=2)
		 {
			 int I=i+offset; 
			 dest.at<double>(j,I) = (dest.at<double>(j-1,I)+dest.at<double>(j,I-1)+dest.at<double>(j,I+1)+dest.at<double>(j+1,I))*0.25;
			 /*
#ifdef TEST
			 C(0,0)=(double)dest.at<double>(j-2,I-1);
			 C(1,0)=(double)dest.at<double>(j-2,I+1);
			 C(2,0)=(double)dest.at<double>(j,I-1);
			 C(3,0)=(double)dest.at<double>(j,I+1);

			 C(0,1)=(double)dest.at<double>(j-1,I-2);
			 C(1,1)=(double)dest.at<double>(j-1,I+2);
			 C(2,1)=(double)dest.at<double>(j+1,I-2);
			 C(3,1)=(double)dest.at<double>(j+1,I+2);

			 C(0,2)=(double)dest.at<double>(j-1,I);
			 C(1,2)=(double)dest.at<double>(j-1,I+2);
			 C(2,2)=(double)dest.at<double>(j+1,I);
			 C(3,2)=(double)dest.at<double>(j+1,I+2);

			 C(0,3)=(double)dest.at<double>(j,I-1);
			 C(1,3)=(double)dest.at<double>(j,I+1);
			 C(2,3)=(double)dest.at<double>(j+2,I-1);
			 C(3,3)=(double)dest.at<double>(j+2,I+1);
			 #endif
			 C(0,0)=dest.at<double>(j-3,I+0);
			 C(0,1)=dest.at<double>(j-1,I-2);
			 C(0,2)=dest.at<double>(j-1,I+2);
			 C(0,3)=dest.at<double>(j+1,I+0);

			 C(1,0)=dest.at<double>(j-2,I-1);
			 C(1,1)=dest.at<double>(j+0,I-3);
			 C(1,2)=dest.at<double>(j+0,I+1);
			 C(1,3)=dest.at<double>(j+2,I-1);

			 C(2,0)=dest.at<double>(j-2,I+1);
			 C(2,1)=dest.at<double>(j+0,I-1);
			 C(2,2)=dest.at<double>(j+0,I+3);
			 C(2,3)=dest.at<double>(j+2,I+1);

			 C(3,0)=dest.at<double>(j-1,I+0);
			 C(3,1)=dest.at<double>(j+1,I-2);
			 C(3,2)=dest.at<double>(j+1+2,I+2);
			 C(3,3)=dest.at<double>(j+3,I+0);

			// C=C.t();

			 y(0)=dest.at<double>(j-1,I+0);
			 y(1)=dest.at<double>(j+0,I-1);
			 y(2)=dest.at<double>(j+0,I+1);
			 y(3)=dest.at<double>(j+1,I+0);

			 const double ave = (y(0)+y(1)+y(2)+y(3))*0.25;
			 const double var = (
				 +(y(0)-ave)*(y(0)-ave)
				 +(y(1)-ave)*(y(1)-ave)
				 +(y(2)-ave)*(y(2)-ave)
				 +(y(3)-ave)*(y(3)-ave)
				 )*0.25;
				 
			 Mat a = Mat::ones(4,1,CV_64F)*0.25;

			 Matx44d CtC=Mat(C.t()*C);
			 if(determinant(CtC)!=0 && var>th) Mat(CtC.inv()*(C.t()*y)).copyTo(a);
			 dest.at<double>(j,I)=a.dot(y);
			 */
		 }
	 }
 }
void NEDI(InputArray src, OutputArray dest, int rate)
{
	int odd_w =0;
	int odd_h =0;
	if(src.size().width%2!=0)odd_w=1;
	if(src.size().height%2!=0)odd_h=1;
	Mat im;
	copyMakeBorder(src, im,4,4+odd_h,4,4+odd_w,BORDER_REPLICATE);
	Mat imf;im.convertTo(imf, CV_64F, 1.0/255.0);
	Mat dst;
	NEDI_1st_x2(imf, dst);

	Mat(dst(Rect(4,4,src.size().width*2,src.size().height*2))).convertTo(dest, CV_8U, 255);
}