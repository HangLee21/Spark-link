/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

#pragma once

#include "av_processor.h"

extern "C" {
#include "libswscale/swscale.h"
#include<GL/gl.h>
}
#include <iostream>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace gtoe_emulation {

class Splicer: public AVProcessor {
public:
    Splicer();
    ~Splicer();
    std::vector<AVFrame *> Process(
        const std::vector<AVFrame *>& frames) override;

private:
    AVFrame* Splice(const std::vector<AVFrame *>& frames);
    AVFrame* SpliceOneRow(const std::vector<AVFrame *>& frames);
    AVFrame* SpliceTwoRows(const std::vector<AVFrame *>& frames);
    cv::Mat avframe_to_cvmat(AVFrame* frame);
    AVFrame* cvmat_to_avframe(const cv::Mat image);

    void hard_encode_mat();
    int width;
    int height; // the height of the first row for the dst AVFrame
    int heightX; // the height of the second row for the dst AVFrame
    uint8_t* prgb24;

    std::vector<int> widthVec;

    std::vector<SwsContext* > yuv2rgbSwsCtxes;
    SwsContext* rgb2yuvSwsCtx;

    AVPixelFormat format;
};

} // namespace gtoe_emulation

static const  char* camera_names[4] = {
    "front", "left", "back", "right"
};

static const  char* camera_flip_mir[4] = {
    "n", "r-", "m", "r+"
};
//单个格子10cm
//--------------------------------------------------------------------
//(shift_width, shift_height): how far away the birdview looks outside
//of the calibration pattern in horizontal and vertical directions
static const  int shift_w = 230;
static const  int shift_h = 230;

static const  int cali_map_w  = 800;
static const  int cali_map_h  = 800;
//size of the gap between the calibration pattern and the car
//in horizontal and vertical directions
static const  int inn_shift_w = 400;
static const  int inn_shift_h = 400;

//total width/height of the stitched image
static const  int total_w = cali_map_w + 2 * shift_w;
static const  int total_h = cali_map_h + 2 * shift_h;

//four corners of the rectangular region occupied by the car
//top-left (x_left, y_top), bottom-right (x_right, y_bottom)
static const  int xl = shift_w +  inn_shift_w;
static const  int xr = total_w - xl;
static const  int yt = shift_h + inn_shift_h;
static const  int yb = total_h - yt;
//--------------------------------------------------------------------

static std::map<std::string, cv::Size> project_shapes = {
    {"front",  cv::Size(total_w, yt)},
    {"back",   cv::Size(total_w, yt)},
    {"left",   cv::Size(total_h, xl)},
    {"right",  cv::Size(total_h, xl)},
};

//pixel locations of the four points to be chosen.
//you must click these pixels in the same order when running
//the get_projection_map.py script
static std::map<std::string, std::vector<cv::Point2f>> project_keypoints = {
    {"front", {cv::Point2f(shift_w + 120, shift_h),
              cv::Point2f(shift_w + 480, shift_h),
              cv::Point2f(shift_w + 120, shift_h + 160),
              cv::Point2f(shift_w + 480, shift_h + 160)}},

    {"back", {cv::Point2f(shift_w + 120, shift_h),
              cv::Point2f(shift_w + 480, shift_h),
              cv::Point2f(shift_w + 120, shift_h + 160),
              cv::Point2f(shift_w + 480, shift_h + 160)}},

    {"left", {cv::Point2f(shift_h + 280, shift_w),
              cv::Point2f(shift_h + 840, shift_w),
              cv::Point2f(shift_h + 280, shift_w + 160),
              cv::Point2f(shift_h + 840, shift_w + 160)}},

    {"right", {cv::Point2f(shift_h + 160, shift_w),
              cv::Point2f(shift_h + 720, shift_w),
              cv::Point2f(shift_h + 160, shift_w + 160),
              cv::Point2f(shift_h + 720, shift_w + 160)}}
};

struct CameraPrms
{
    std::string name;
    cv::Mat dist_coff;
    cv::Mat camera_matrix;
    cv::Mat project_matrix;
    cv::Mat trans_matrix;
    cv::Size size;

    cv::Mat scale_xy;
    cv::Mat shift_xy;
};

struct BgrSts {
    int b;
    int g;
    int r;

    BgrSts() {
        b = g = r = 0;
    }
};

template<typename _T>
static inline _T clip(float data, int max)
{
    if (data > max)
        return max;
    return (_T)data;
}

void display_mat(cv::Mat& img, std::string name);
bool read_prms(const std::string& path, CameraPrms& prms);
bool save_prms(const std::string& path, CameraPrms& prms);
void undist_by_remap(const cv::Mat& src, cv::Mat& dst, const CameraPrms& prms);

void merge_image(const cv::Mat& src1, const cv::Mat& src2, const cv::Mat& w, const cv::Mat& out);
void awb_and_lum_banlance(std::vector<cv::Mat*> srcs);