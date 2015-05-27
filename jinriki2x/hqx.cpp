/*
 * Copyright (C) 2010 Cameron Zemek ( grom@zeminvaders.net)
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#include "util.h"
//#include <unistd.h>
#include <stdint.h>
#include "hqx.h"

uint32_t   RGBtoYUV[16777216];
uint32_t   YUV1, YUV2;

void HQX_CALLCONV hqxInit(void)
{
    /* Initalize RGB to YUV lookup table */
    uint32_t c, r, g, b, y, u, v;
    for (c = 0; c < 16777215; c++) {
        r = (c & 0xFF0000) >> 16;
        g = (c & 0x00FF00) >> 8;
        b = c & 0x0000FF;
        y = (uint32_t)(0.299*r + 0.587*g + 0.114*b);
        u = (uint32_t)(-0.169*r - 0.331*g + 0.5*b) + 128;
        v = (uint32_t)(0.5*r - 0.419*g - 0.081*b) + 128;
        RGBtoYUV[c] = (y << 16) + (u << 8) + v;
    }
}


static inline uint32_t swapByteOrder(uint32_t ui)
{
    return (ui >> 24) | ((ui << 8) & 0x00FF0000) | ((ui >> 8) & 0x0000FF00) | (ui << 24);
}

void hqx_(Mat& src, Mat& dest, int scaleBy)
{
	Mat srca;
	cvtColor(src,srca,CV_BGR2BGRA);
	Mat dst = Mat::zeros(Size(src.cols*scaleBy,src.rows*scaleBy),CV_8UC4);
	

	int width = src.cols;
	int height = src.rows;

    // Allocate memory for image data
    size_t srcSize = width * height * sizeof(uint32_t);
    uint8_t *srcData = (uint8_t *) malloc(srcSize);
    size_t destSize = width * scaleBy * height * scaleBy * sizeof(uint32_t);
    uint8_t *destData = (uint8_t *) malloc(destSize);

	uint32_t *sp = (uint32_t *) srca.data;
	uint32_t *dp = (uint32_t *) dst.data;

    hqxInit();
    switch (scaleBy) {
    case 2:
        hq2x_32(sp, dp, width, height);
        break;
    case 3:
        hq3x_32(sp, dp, width, height);
        break;
    case 4:
        hq4x_32(sp, dp, width, height);
        break;
	default:
		std::cout<<"upscale "<<scaleBy<<" is not supported"<<std::endl;
		break;
    }

	cvtColor(dst,dest,CV_BGRA2BGR);
}

void hqx(InputArray src, OutputArray dest, const int scaleBy)
{
	if(src.channels()==1)
	{
		Mat srcc,destc;
		cvtColor(src,srcc,CV_GRAY2BGR);
		hqx_(srcc,destc,scaleBy);
		cvtColor(destc,dest,CV_BGR2GRAY);
	}
	else
	{
		Mat a,b;
		a = src.getMat();
		b = dest.getMat();
		hqx_(a, b, scaleBy);
		b.copyTo(dest);
	}
}
