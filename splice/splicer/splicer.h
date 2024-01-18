/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

#pragma once

#include "av_processor.h"
#include "srcs/common.h"
extern "C" {
#include "libswscale/swscale.h"
}

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
    AVFrame* MergeFrame(const std::vector<cv::Mat>& mats);
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