/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */

#include "splicer.h"
#include <stdexcept>
#include <bits/stdc++.h>


const file_path = "../json/";
std::vector<std::string> json_name = {"0.json", "1.json", "2.json", "3.json"};
std::vector<cv::Mat> camaraMatrixes = {};
std::vector<cv::Mat> distVectors = {};
namespace gtoe_emulation {

Splicer::Splicer(): rgb2yuvSwsCtx(NULL), width(-1),
    height(-1), heightX(-1), prgb24(NULL)
{
}

Splicer::~Splicer()
{
    if (rgb2yuvSwsCtx) {
        sws_freeContext(rgb2yuvSwsCtx);
    }

    for (auto swsCtx: yuv2rgbSwsCtxes) {
        sws_freeContext(swsCtx);
    }

    if (prgb24) {
        free(prgb24);
    }
}

std::vector<AVFrame *> Splicer::Process(const std::vector<AVFrame *>& frames)
{
    std::vector<AVFrame *> resultFrames;
    std::vector<cv::Mat> mergeFrames;
    AVFrame* frame = Splice(frames);
    // read json
    for(int i = 0 ; i < 4; i++){
        // 读取原始图像
        cv::Mat distortedImage = frame[0];
    
        cv::Mat cameraMatrix = camaraMatrixes[i];  // 替换为实际的相机矩阵
        cv::Mat distortionCoefficients = distVectors[i];  // 替换为实际的畸变系数

        // 创建输出图像
        cv::Mat undistortedImage;

        // 进行去畸变
        cv::undistort(distortedImage, undistortedImage, cameraMatrix, distortionCoefficients);
        AVFrame* frame = mergeFrames.push_back(undistortedImage);
    }

    // end merge
    if (frame) {
        resultFrames.push_back(frame);
    }

    return resultFrames;
}

AVFrame* Splicer::MergeFrame(const std::vector<cv::Mat>& mats){
    AVFrame* frame = nullptr;
    // TODO 
    return frame;
}

cv::Mat Splicer::cvmat_to_avframe(const cv::Mats image){
    // TODO 
}

cv::Mat Splicer::avframe_to_cvmat(AVFrame* frame){
    // TODO
}

AVFrame* Splicer::Splice(const std::vector<AVFrame *>& frames)
{
    AVFrame* resultFrame = NULL;

    switch (frames.size()) {
        case 0:
            break;
        case 1:
            resultFrame = frames[0];
            break;
        case 2:
        case 3:
            resultFrame = SpliceOneRow(frames);
            break;
        case 4:
        case 5:
        case 6:
            resultFrame = SpliceTwoRows(frames);
            break;
        default:
            throw std::runtime_error("Don't support " +
                std::to_string(frames.size()) + " channels of inputs");
    }

    // release memory
    for (auto frame: frames) {
        if (frame && frame != resultFrame) {
            av_frame_free(&frame);
        }
    }

    return resultFrame;
}

static AVPixelFormat GetFormat(int format)
{
    // replace deprecated format
    switch (format) {
        case AV_PIX_FMT_YUVJ420P:
            format = AV_PIX_FMT_YUV420P;
            break;
        case AV_PIX_FMT_YUVJ422P:
            format = AV_PIX_FMT_YUV422P;
            break;
        case AV_PIX_FMT_YUVJ444P:
            format = AV_PIX_FMT_YUV444P;
            break;
        case AV_PIX_FMT_YUVJ440P:
            format = AV_PIX_FMT_YUV440P;
            break;
        default:
            break;
    }
    return (AVPixelFormat)format;
}

AVFrame* Splicer::SpliceOneRow(const std::vector<AVFrame *>& frames)
{
    if (width == -1) {
        width = 0;
        height = 0;

        for (auto f: frames) {
            if (!f) {
                throw std::runtime_error(
                    "AVFrame should not be null at the very start");
            }

            width += f->width;
            height = FFMAX(height, f->height);
        }

        format = (AVPixelFormat) frames[0]->format;
    }

    if (yuv2rgbSwsCtxes.empty()) {
        for (int i = 0; i < frames.size(); i++) {
            SwsContext *swsCtx =
                sws_getContext(frames[i]->width,
                               frames[i]->height,
                               GetFormat(frames[i]->format),
                               frames[i]->width,
                               height,
                               AV_PIX_FMT_RGB24,
                               SWS_BILINEAR,
                               NULL,
                               NULL,
                               NULL);

            if (!swsCtx) {
                throw std::runtime_error("Failed to call sws_getContext");
            }

            yuv2rgbSwsCtxes.push_back(swsCtx);
            widthVec.push_back(frames[i]->width);
        }

        rgb2yuvSwsCtx =
            sws_getContext(width,
                           height,
                           AV_PIX_FMT_RGB24,
                           width,
                           height,
                           GetFormat(frames[0]->format),
                           SWS_BILINEAR,
                           NULL,
                           NULL,
                           NULL);
        if (!rgb2yuvSwsCtx) {
            throw std::runtime_error("Failed to call sws_getContext");
        }

        prgb24 = (uint8_t *) malloc(3 * width * height);
        if (!prgb24) {
            throw std::runtime_error("Cannot allocate memory for RGB buffer");
        }
    }

    memset(prgb24, 0, 3 * width * height);

    // Convert from YUV to RGB and splice
    uint8_t *rgb24[1] = { prgb24 };
    int rgb24Stride[1] = { 3 * width };
    for (int i = 0; i < frames.size(); i++) {
        if (i > 0) {
            if (frames[i - 1]) {
                rgb24[0] += 3 * frames[i - 1]->width;
            } else {
                rgb24[0] += 3 * widthVec[i - 1];
            }
        }

        if (!frames[i]) {
            continue;
        }

        sws_scale(yuv2rgbSwsCtxes[i],
                  frames[i]->data,
                  frames[i]->linesize,
                  0,
                  height,
                  rgb24,
                  rgb24Stride);
    }

    AVFrame* splicedFrame = av_frame_alloc();
    if (!splicedFrame) {
        throw std::runtime_error("Failed to allocate AVFrame");
    }

    splicedFrame->format = format;
    splicedFrame->width = width;
    splicedFrame->height = height;
    AVFrame *framePtr = nullptr;
    for (int i = 0; i < frames.size(); ++i) {
        if (frames[i]) {
            framePtr = frames[i];
            break;
        }
    }
    if (framePtr) {
        splicedFrame->pts = framePtr->pts;
        splicedFrame->pkt_dts = framePtr->pkt_dts;
        splicedFrame->duration = framePtr->duration;
        splicedFrame->pict_type = framePtr->pict_type;
        splicedFrame->color_primaries = framePtr->color_primaries;
        splicedFrame->color_range = framePtr->color_range;
        splicedFrame->color_trc = framePtr->color_trc;
        splicedFrame->colorspace = framePtr->colorspace;
        splicedFrame->chroma_location = framePtr->chroma_location;
        splicedFrame->best_effort_timestamp = framePtr->best_effort_timestamp;
    }

    int ret = av_frame_get_buffer(splicedFrame, 0);
    if (ret < 0) {
        throw std::runtime_error("Cannot allocate buffer for AVFrame");
    }

    // Convert from RGB to YUV
    rgb24[0] = prgb24;
    sws_scale(rgb2yuvSwsCtx,
              rgb24,
              rgb24Stride,
              0,
              height,
              splicedFrame->data,
              splicedFrame->linesize);

    return splicedFrame;
}

AVFrame* Splicer::SpliceTwoRows(const std::vector<AVFrame *>& frames)
{

    //return frames[0];
    bool shrinkFlag = false;
    const int n = frames.size();
    const int m = (n + 1) / 2;

    if (width == -1) {
        width = 0;
        height = 0;
        heightX = 0;

        for (int i = 0; i < m; i++) {
            if (!frames[i]) {
                throw std::runtime_error(
                    "AVFrame should not be null at the very start");
            }

            width += frames[i]->width;
            height = FFMAX(height, frames[i]->height);
        }

        int widthX = 0;
        for (int i = m; i < n; i++) {
            if (!frames[i]) {
                throw std::runtime_error(
                    "AVFrame should not be null at the very start");
            }

            widthX += frames[i]->width;
            heightX = FFMAX(heightX, frames[i]->height);
        }
        width = FFMAX(width, widthX);

        if (n > 4) {
            shrinkFlag = true;
            width /= 2;
            height /= 2;
            heightX /= 2;
        }

        format = (AVPixelFormat) frames[0]->format;
    }

    if (yuv2rgbSwsCtxes.empty()) {
        // first row
        for (int i = 0; i < m; i++) {
            SwsContext *swsCtx =
                sws_getContext(
                    frames[i]->width,
                    frames[i]->height,
                    GetFormat(frames[i]->format),
                    shrinkFlag ? frames[i]->width / 2 : frames[i]->width,
                    height,
                    AV_PIX_FMT_RGB24,
                    SWS_BILINEAR,
                    NULL,
                    NULL,
                    NULL);

            if (!swsCtx) {
                throw std::runtime_error("Failed to call sws_getContext");
            }

            yuv2rgbSwsCtxes.push_back(swsCtx);
            widthVec.push_back(frames[i]->width);
        }

        // second row
        for (int i = m; i < n; i++) {
            SwsContext *swsCtx =
                sws_getContext(
                    frames[i]->width,
                    frames[i]->height,
                    GetFormat(frames[i]->format),
                    shrinkFlag ? frames[i]->width / 2 : frames[i]->width,
                    heightX,
                    AV_PIX_FMT_RGB24,
                    SWS_BILINEAR,
                    NULL,
                    NULL,
                    NULL);

            if (!swsCtx) {
                throw std::runtime_error("Failed to call sws_getContext");
            }

            yuv2rgbSwsCtxes.push_back(swsCtx);
            widthVec.push_back(frames[i]->width);
        }

        rgb2yuvSwsCtx =
            sws_getContext(width,
                           height + heightX,
                           AV_PIX_FMT_RGB24,
                           width,
                           height + heightX,
                           GetFormat(frames[0]->format),
                           SWS_BILINEAR,
                           NULL,
                           NULL,
                           NULL);
        if (!rgb2yuvSwsCtx) {
            throw std::runtime_error("Failed to call sws_getContext");
        }

        prgb24 = (uint8_t *) malloc(3 * width * (height + heightX));
        if (!prgb24) {
            throw std::runtime_error("Cannot allocate memory for RGB buffer");
        }
    }

    memset(prgb24, 0, 3 * width * (height + heightX));

    // Convert from YUV to RGB and splice

    uint8_t *rgb24[1] = { prgb24 };
    int rgb24Stride[1] = { 3 * width };

    // first row
    for (int i = 0; i < m; i++) {
        if (i > 0) {
            if (shrinkFlag) {
                if (frames[i - 1]) {
                    rgb24[0] += 3 * frames[i - 1]->width / 2;
                } else {
                    rgb24[0] += 3 * widthVec[i - 1] / 2;
                }
            } else {
                if (frames[i - 1]) {
                    rgb24[0] += 3 * frames[i - 1]->width;
                } else {
                    rgb24[0] += 3 * widthVec[i - 1];
                }
            }
        }

        if (!frames[i]) {
            continue;
        }

        sws_scale(yuv2rgbSwsCtxes[i],
                  frames[i]->data,
                  frames[i]->linesize,
                  0,
                  height,
                  rgb24,
                  rgb24Stride);
    }

    // second row
    rgb24[0] = { prgb24 + 3 * width * height };
    for (int i = m; i < n; i++) {
        if (i != m) {
            if (shrinkFlag) {
                if (frames[i - 1]) {
                    rgb24[0] += 3 * frames[i - 1]->width / 2;
                } else {
                    rgb24[0] += 3 * widthVec[i - 1] / 2;
                }
            } else {
                if (frames[i - 1]) {
                    rgb24[0] += 3 * frames[i - 1]->width;
                } else {
                    rgb24[0] += 3 * widthVec[i - 1];
                }
            }
        }

        if (!frames[i]) {
            continue;
        }

        sws_scale(yuv2rgbSwsCtxes[i],
                  frames[i]->data,
                  frames[i]->linesize,
                  0,
                  heightX,
                  rgb24,
                  rgb24Stride);
    }

    AVFrame* splicedFrame = av_frame_alloc();
    if (!splicedFrame) {
        throw std::runtime_error("Failed to allocate AVFrame");
    }

    splicedFrame->format = format;
    splicedFrame->width = width;
    splicedFrame->height = height + heightX;
    AVFrame *framePtr = nullptr;
    for (int i = 0; i < frames.size(); ++i) {
        if (frames[i]) {
            framePtr = frames[i];
            break;
        }
    }
    if (framePtr) {
        splicedFrame->pts = framePtr->pts;
        splicedFrame->pkt_dts = framePtr->pkt_dts;
        splicedFrame->duration = framePtr->duration;
        splicedFrame->pict_type = framePtr->pict_type;
        splicedFrame->color_primaries = framePtr->color_primaries;
        splicedFrame->color_range = framePtr->color_range;
        splicedFrame->color_trc = framePtr->color_trc;
        splicedFrame->colorspace = framePtr->colorspace;
        splicedFrame->chroma_location = framePtr->chroma_location;
        splicedFrame->best_effort_timestamp = framePtr->best_effort_timestamp;
    }

    int ret = av_frame_get_buffer(splicedFrame, 0);
    if (ret < 0) {
        throw std::runtime_error("Cannot allocate buffer for AVFrame");
    }

    // Convert from RGB to YUV
    rgb24[0] = prgb24;
    sws_scale(rgb2yuvSwsCtx,
              rgb24,
              rgb24Stride,
              0,
              height + heightX,
              splicedFrame->data,
              splicedFrame->linesize);

    return splicedFrame;
}

} // namespace gtoe_emulation

extern "C" gtoe_emulation::AVProcessor* CreateSplicer()
{
    return new gtoe_emulation::Splicer();
}
