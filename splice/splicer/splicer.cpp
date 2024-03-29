/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */

#include "splicer.h"
#include <stdexcept>
#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "jsoncpp/json/json.h"
#include <fstream>
const std::string file_path = "../json/";
std::vector<std::string> json_name = {"0.json", "1.json", "2.json", "3.json"};
std::vector<cv::Mat> cameraMatrixes = {};
std::vector<cv::Mat> distVectors = {};
std::vector<cv::Mat> shadowMatrixes = {};
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


void Splicer::hard_encode_mat(){
    cv::Mat camara_mat_0;
    camara_mat_0.create(3, 3, CV_64F);  // 创建一个3x3的单通道8位无符号整数矩阵
    camara_mat_0.at<double>(0, 0) = 365.71645588482716;  // 赋值第一个像素为255
    camara_mat_0.at<double>(0, 1) = 0.0;  // 赋值第一个像素为255
    camara_mat_0.at<double>(0, 2) = 690.3554811474158;  // 赋值第一个像素为255
    camara_mat_0.at<double>(1, 0) = 0.0;  // 赋值第一个像素为255
    camara_mat_0.at<double>(1, 1) = 274.77928168180495;  // 赋值第一个像素为255
    camara_mat_0.at<double>(1, 2) = 342.8503226467983;  // 赋值第一个像素为255
    camara_mat_0.at<double>(2, 0) = 0.0;  // 赋值第一个像素为255
    camara_mat_0.at<double>(2, 1) = 0.0;  // 赋值第一个像素为255
    camara_mat_0.at<double>(2, 2) = 1.0;  // 赋值第一个像素为255
    cameraMatrixes.push_back(camara_mat_0);

    cv::Mat dist_mat_0;
    dist_mat_0.create(5, 1, CV_64F);  // 创建一个3x3的单通道8位无符号整数矩阵
    dist_mat_0.at<double>(0, 0) = -0.23416148966938685;  // 赋值第一个像素为255
    dist_mat_0.at<double>(1, 0) = 0.0484646853675212;  // 赋值第一个像素为255
    dist_mat_0.at<double>(2, 0) = 0.0026369026846284472;  // 赋值第一个像素为255
    dist_mat_0.at<double>(3, 0) = 0.0013246980666203948;  // 赋值第一个像素为255
    dist_mat_0.at<double>(4, 0) = -0.00399352748387932;  // 赋值第一个像素为255
    distVectors.push_back(dist_mat_0);

    cv::Mat camara_mat_1;
    camara_mat_1.create(3, 3, CV_64F);  // 创建一个3x3的单通道8位无符号整数矩阵
    camara_mat_1.at<double>(0, 0) = 373.9884786517187;  // 赋值第一个像素为255
    camara_mat_1.at<double>(0, 1) = 0.0;  // 赋值第一个像素为255
    camara_mat_1.at<double>(0, 2) = 664.4622453968176;  // 赋值第一个像素为255
    camara_mat_1.at<double>(1, 0) = 0.0;  // 赋值第一个像素为255
    camara_mat_1.at<double>(1, 1) = 131.06730496430424;  // 赋值第一个像素为255
    camara_mat_1.at<double>(1, 2) = 214.69112984409753;  // 赋值第一个像素为255
    camara_mat_1.at<double>(2, 0) = 0.0;  // 赋值第一个像素为255
    camara_mat_1.at<double>(2, 1) = 0.0;  // 赋值第一个像素为255
    camara_mat_1.at<double>(2, 2) = 1.0;  // 赋值第一个像素为255
    cameraMatrixes.push_back(camara_mat_1);

    cv::Mat dist_mat_1;
    dist_mat_1.create(5, 1, CV_64F);  // 创建一个3x3的单通道8位无符号整数矩阵
    dist_mat_1.at<double>(0, 0) = -0.21844755906734947;  // 赋值第一个像素为255
    dist_mat_1.at<double>(1, 0) = -0.016874401855457532;  // 赋值第一个像素为255
    dist_mat_1.at<double>(2, 0) = 0.2536237647137276;  // 赋值第一个像素为255
    dist_mat_1.at<double>(3, 0) = 0.0035275474453427685;  // 赋值第一个像素为255
    dist_mat_1.at<double>(4, 0) = 0.0038249029210460075;  // 赋值第一个像素为255
    distVectors.push_back(dist_mat_1);

    cv::Mat camara_mat_2;
    camara_mat_2.create(3, 3, CV_64F);  // 创建一个3x3的单通道8位无符号整数矩阵
    camara_mat_2.at<double>(0, 0) = 354.4921697196074;  // 赋值第一个像素为255
    camara_mat_2.at<double>(0, 1) = 0.0;  // 赋值第一个像素为255
    camara_mat_2.at<double>(0, 2) = 631.359415673472;  // 赋值第一个像素为255
    camara_mat_2.at<double>(1, 0) = 0.0;  // 赋值第一个像素为255
    camara_mat_2.at<double>(1, 1) = 263.2563867498032;  // 赋值第一个像素为255
    camara_mat_2.at<double>(1, 2) = 312.0320234604565;  // 赋值第一个像素为255
    camara_mat_2.at<double>(2, 0) = 0.0;  // 赋值第一个像素为255
    camara_mat_2.at<double>(2, 1) = 0.0;  // 赋值第一个像素为255
    camara_mat_2.at<double>(2, 2) = 1.0;  // 赋值第一个像素为255
    cameraMatrixes.push_back(camara_mat_2);

    cv::Mat dist_mat_2;
    dist_mat_2.create(5, 1, CV_64F);  // 创建一个3x3的单通道8位无符号整数矩阵
    dist_mat_2.at<double>(0, 0) = -0.2525602792580275;  // 赋值第一个像素为255
    dist_mat_2.at<double>(1, 0) = 0.06777938800619913;  // 赋值第一个像素为255
    dist_mat_2.at<double>(2, 0) = 0.001199961749514866;  // 赋值第一个像素为255
    dist_mat_2.at<double>(3, 0) = -0.0004376986145580113;  // 赋值第一个像素为255
    dist_mat_2.at<double>(4, 0) = -0.008158497446121587;  // 赋值第一个像素为255
    distVectors.push_back(dist_mat_2);

    cv::Mat camara_mat_3;
    camara_mat_3.create(3, 3, CV_64F);  // 创建一个3x3的单通道8位无符号整数矩阵
    camara_mat_3.at<double>(0, 0) = 298.16809905743787;  // 赋值第一个像素为255
    camara_mat_3.at<double>(0, 1) = 0.0;  // 赋值第一个像素为255
    camara_mat_3.at<double>(0, 2) = 668.6623659431037;  // 赋值第一个像素为255
    camara_mat_3.at<double>(1, 0) = 0.0;  // 赋值第一个像素为255
    camara_mat_3.at<double>(1, 1) = 194.92743436219445;  // 赋值第一个像素为255
    camara_mat_3.at<double>(1, 2) = 285.66734013539826;  // 赋值第一个像素为255
    camara_mat_3.at<double>(2, 0) = 0.0;  // 赋值第一个像素为255
    camara_mat_3.at<double>(2, 1) = 0.0;  // 赋值第一个像素为255
    camara_mat_3.at<double>(2, 2) = 1.0;  // 赋值第一个像素为255
    cameraMatrixes.push_back(camara_mat_3);

    cv::Mat dist_mat_3;
    dist_mat_3.create(5, 1, CV_64F);  // 创建一个3x3的单通道8位无符号整数矩阵
    dist_mat_3.at<double>(0, 0) = -0.15203735255507325;  // 赋值第一个像素为255
    dist_mat_3.at<double>(1, 0) = 0.022393019553115786;  // 赋值第一个像素为255
    dist_mat_3.at<double>(2, 0) = 0.0035329643078907164;  // 赋值第一个像素为255
    dist_mat_3.at<double>(3, 0) = 0.004626616567290379;  // 赋值第一个像素为255
    dist_mat_3.at<double>(4, 0) = -0.0014035543102246485;  // 赋值第一个像素为255
    distVectors.push_back(dist_mat_3);
}

std::vector<AVFrame *> Splicer::Process(const std::vector<AVFrame *>& frames)
{

    std::vector<AVFrame *> resultFrames;

    //AVFrame* frame = Splice(frames);
    
    cv::Mat origin_dir_img[4];
    cv::Mat undist_dir_img[4];
    cv::Mat merge_weights_img[4];
    cv::Mat out_put_img;
    CameraPrms prms[4];


    out_put_img = cv::Mat(cv::Size(total_w, total_h), CV_8UC3, cv::Scalar(0, 0, 0));

    float *w_ptr[4];

    out_put_img = cv::Mat(cv::Size(total_w, total_h), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat weights = cv::imread("/home/sparklink/Downloads/data/weights.png", -1);

    if (weights.channels() != 4) {
        std::cerr << "imread weights failed " << weights.channels() << "\r\n";
        return std::vector<AVFrame *>();
    }

    for (int i = 0; i < 4; ++i) {
        merge_weights_img[i] = cv::Mat(weights.size(), CV_32FC1, cv::Scalar(0, 0, 0));
        w_ptr[i] = (float *)merge_weights_img[i].data;
    }
    //read weights of corner
    int pixel_index = 0;
    for (int h = 0; h < weights.rows; ++h) {
        uchar* uc_pixel = weights.data + h * weights.step;
        for (int w = 0; w < weights.cols; ++w) {
            w_ptr[0][pixel_index] = uc_pixel[0] / 255.0f;
            w_ptr[1][pixel_index] = uc_pixel[1] / 255.0f;
            w_ptr[2][pixel_index] = uc_pixel[2] / 255.0f;
            w_ptr[3][pixel_index] = uc_pixel[3] / 255.0f;
            uc_pixel += 4;
            ++pixel_index;
        }
    }
    
    //1. read calibration prms
    for (int i = 0; i < 4; ++i) {
        auto& prm = prms[i];
        prm.name = camera_names[i];
        int zeroCount = cv::countNonZero(prm.camera_matrix);
        if(zeroCount == 0){
            auto ok = read_prms("/home/sparklink/Downloads/data/eye_" + prm.name + ".yaml", prm);
            if (!ok) {
                return std::vector<AVFrame *>();
            }
        }
    }
    
    // 2.avframe to cvmat
    for(int i = 0 ; i < 4; i++){
        cv::Mat distortedImage = avframe_to_cvmat(frames[i]);
        origin_dir_img[i] = distortedImage;
    }

    //3. undistort image
    for (int i = 0; i < 4; ++i) {
        auto& prm = prms[i];
        cv::Mat& src = origin_dir_img[i];

        undist_by_remap(src, src, prm);
        cv::warpPerspective(src, src, prm.project_matrix, project_shapes[prm.name]);

        if (camera_flip_mir[i] == "r+") {
            cv::rotate(src, src, cv::ROTATE_90_CLOCKWISE);
        } else if (camera_flip_mir[i] == "r-") {
            cv::rotate(src, src, cv::ROTATE_90_COUNTERCLOCKWISE);
        } else if (camera_flip_mir[i] == "m") {
            cv::rotate(src, src, cv::ROTATE_180);
        }
        cv::rotate(src, src, cv::ROTATE_180);
        undist_dir_img[i] = src.clone();
    }
    
    
    //4.start combine
    //4.1 out_put_img center copy
    for (int i = 0; i < 4; ++i) {
        cv::Rect roi;
        bool is_cali_roi = false;
        if (std::string(camera_names[i]) == "front") {
            roi = cv::Rect(xl, 0, xr - xl, yt);
            //std::cout << "\nfront" << roi;
            undist_dir_img[i](roi).copyTo(out_put_img(roi));
        } else if (std::string(camera_names[i]) == "left") {
            roi = cv::Rect(0, yt, xl, yb - yt);
            //std::cout << "\nleft" << roi << out_put_img.size();
            undist_dir_img[i](roi).copyTo(out_put_img(roi));
        } else if (std::string(camera_names[i]) == "right") {
            roi = cv::Rect(0, yt, xl, yb - yt);
            //std::cout << "\nright" << roi << out_put_img.size();
            undist_dir_img[i](roi).copyTo(out_put_img(cv::Rect(xr, yt, total_w - xr, yb - yt)));
        } else if (std::string(camera_names[i]) == "back") {
            roi = cv::Rect(xl, 0, xr - xl, yt);
            //std::cout << "\nright" << roi << out_put_img.size();
            undist_dir_img[i](roi).copyTo(out_put_img(cv::Rect(xl, yb, xr - xl, yt)));
        }
    }
    
    //4.2the four corner merge
    //w: 0 左下 1 右上 2 左上 3 左下
    //image: "front", "left", "back", "right"
    cv::Rect roi;
    //左上
    roi = cv::Rect(0, 0, xl, yt);
    merge_image(undist_dir_img[0](roi), undist_dir_img[1](roi), merge_weights_img[2], out_put_img(roi));
    //右上
    roi = cv::Rect(xr, 0, xl, yt);
    merge_image(undist_dir_img[0](roi), undist_dir_img[3](cv::Rect(0, 0, xl, yt)), merge_weights_img[1], out_put_img(cv::Rect(xr, 0, xl, yt)));
    //左下
    roi = cv::Rect(0, yb, xl, yt);
    merge_image(undist_dir_img[2](cv::Rect(0, 0, xl, yt)), undist_dir_img[1](roi), merge_weights_img[0], out_put_img(roi));
    //右下
    roi = cv::Rect(xr, 0, xl, yt);
    merge_image(undist_dir_img[2](roi), undist_dir_img[3](cv::Rect(0, yb, xl, yt)), merge_weights_img[3], out_put_img(cv::Rect(xr, yb, xl, yt)));

    cv::flip(out_put_img, out_put_img, 1);
    AVFrame* frame = cvmat_to_avframe(out_put_img);

    // end merge
    if (frame) {
        resultFrames.push_back(frame);
    }
    return resultFrames;
}

AVFrame* Splicer::cvmat_to_avframe(const cv::Mat image){
    //创建AVFrame结构体
    AVFrame* frame = av_frame_alloc();
    if (!frame){
        throw std::runtime_error("Failed to allocate AVFrame");
    }
    int width = image.cols; // 图像宽度
    int height = image.rows; // 图像高度
    frame->format = AV_PIX_FMT_YUV420P; // 设置格式为YUV420P
    frame->width = width;
    frame->height = height;

    //分配数据内存空间
    int ret = av_frame_get_buffer(frame, 0);
    if (ret < 0){
        throw std::runtime_error("Failed to allocate AVFrame data");
    }

    //创建sws context以进行颜色空间转换
    SwsContext* sws_ctx = sws_getContext(width, height, AV_PIX_FMT_RGB24, width, height, AV_PIX_FMT_YUV420P, 0, nullptr, nullptr, nullptr);
    if (!sws_ctx){
        throw std::runtime_error("Failed to create SwsContext");
    }

    //创建临时存储转换后数据的缓冲区
    uint8_t* converted_data[3];
    int converted_linesize[3];
    converted_data[0] = frame->data[0];
    converted_data[1] = frame->data[1];
    converted_data[2]  = frame->data[2];
    converted_linesize[0] = frame->linesize[0];
    converted_linesize[1] = frame->linesize[1];
    converted_linesize[2] = frame->linesize[2];

    //执行颜色空间转换
    cv::Mat bgr_image;
    cv::cvtColor(image, bgr_image, cv::COLOR_RGB2BGR);
    const uint8_t* src_data[1]  = { bgr_image.data}; 
    int src_linesize[1] = {static_cast<int>(bgr_image.step)};
    sws_scale(sws_ctx, src_data, src_linesize, 0, height, converted_data, converted_linesize);

    // 释放
    sws_freeContext(sws_ctx);
    return frame;
}

cv::Mat Splicer::avframe_to_cvmat(AVFrame* frame){
    // 创建一个新的 SwsContext 用于YUV到BGR的转换
    SwsContext* conversion = sws_getContext(frame->width, frame->height, (AVPixelFormat)frame->format, frame->width, frame->height, AV_PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    // 创建一个新的 AVFrame 用于存储转换后的数据
    AVFrame* converted_frame = av_frame_alloc();
    converted_frame->format = AV_PIX_FMT_BGR24;
    converted_frame->width = frame->width;
    converted_frame->height = frame->height;
    av_frame_get_buffer(converted_frame, 0);
    //将YUV数据转换为BGR
    sws_scale(conversion, frame->data, frame->linesize,0,frame->height,converted_frame->data,converted_frame->linesize);

    //创建一个新的 cv::Mat 并将转换后的数据复制到其中
    cv::Mat img = cv::Mat(converted_frame->height, converted_frame->width, CV_8UC3, converted_frame->data[0], converted_frame->linesize[0]).clone();

    //释放已分配的资源
    sws_freeContext(conversion);
    av_frame_free(&converted_frame);
    return img;
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

void display_mat(cv::Mat& img, std::string name)
{
    cv::imshow(name, img);
    cv::waitKey();
}

//read cali prms
bool read_prms(const std::string& path, CameraPrms& prms)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);

    if (!fs.isOpened()) {
        throw std::string("error open file");
        return false;
    }
    
    prms.camera_matrix  = fs["camera_matrix"].mat();
    prms.dist_coff      = fs["dist_coeffs"].mat();
    prms.project_matrix = fs["project_matrix"].mat();
    prms.shift_xy       = fs["shift_xy"].mat();
    prms.scale_xy       = fs["scale_xy"].mat();
    auto size_          = fs["resolution"].mat();
    prms.size           = cv::Size(size_.at<int>(0), size_.at<int>(1));

    fs.release();
    
    return true;
}
//save cali prms
bool save_prms(const std::string& path, CameraPrms& prms)
{
    cv::FileStorage fs(path, cv::FileStorage::WRITE);

    if (!fs.isOpened()) {
        throw std::string("error open file");
        return false;
    }
    
    if (!prms.project_matrix.empty())
        fs << "project_matrix" << prms.project_matrix;

    fs.release();
    
    return true;
}

//undist image by remap
void undist_by_remap(const cv::Mat& src, cv::Mat& dst, const CameraPrms& prms)
{
    //get new camera matrix
    cv::Mat new_camera_matrix = prms.camera_matrix.clone();
    double* matrix_data = (double *)new_camera_matrix.data;

    const auto scale = (const float *)(prms.scale_xy.data);
    const auto shift = (const float * )(prms.shift_xy.data);

    if (!matrix_data || !scale || !shift) {
        return;
    }

    matrix_data[0]         *=  (double)scale[0];
    matrix_data[3 * 1 + 1] *=  (double)scale[1];
    matrix_data[2]         +=  (double)shift[0];
    matrix_data[1 * 3 + 2] +=  (double)shift[1];
    //std::cout << new_camera_matrix;
    //undistort
    cv::Mat map1, map2;
    cv::fisheye::initUndistortRectifyMap(prms.camera_matrix, prms.dist_coff, cv::Mat(), new_camera_matrix, prms.size , CV_16SC2, map1, map2);
    
    cv::remap(src, dst, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);   
}

//merge image by weights
void merge_image(const cv::Mat& src1, const cv::Mat& src2, const cv::Mat& w, const cv::Mat& out)
{
    if (src1.size() != src2.size()) {
        return;
    }

    int p_index = 0;
    float* weights = (float *)(w.data);
    for (int h = 0; h < src1.rows; ++h) {
        uchar* p1 = src1.data + h * src1.step;
        uchar* p2 = src2.data + h * src2.step;
        uchar*  o = out.data + h * out.step;
        for (int w = 0; w < src1.cols; ++w) {
            o[0] = clip<uint8_t>(p1[0] * weights[p_index] + p2[0] * (1 - weights[p_index]), 255);
            o[1] = clip<uint8_t>(p1[1] * weights[p_index] + p2[1] * (1 - weights[p_index]), 255);
            o[2] = clip<uint8_t>(p1[2] * weights[p_index] + p2[2] * (1 - weights[p_index]), 255);
            // o[0] = clip<uint8_t>(p1[0] * 1 + p2[0] * 1, 255);
            // o[1] = clip<uint8_t>(p1[1] * 1 + p2[1] * 1, 255);
            // o[2] = clip<uint8_t>(p1[2] * 1 + p2[2] * 1, 255);
            p1 += 3;
            p2 += 3;
            o  += 3;
            ++p_index;
        }
    }
}

//r g b channel statics
void rgb_info_statics(cv::Mat& src, BgrSts& sts)
{
    int nums = src.rows * src.cols;

    sts.b = sts.r = sts.g = 0;

    for (int h = 0; h < src.rows; ++h) {
        uchar* uc_pixel = src.data + h * src.step;
        for (int w = 0; w < src.cols; ++w) {
            sts.b += uc_pixel[0];
            sts.g += uc_pixel[1];
            sts.r += uc_pixel[2];
            uc_pixel += 3;
        }
    }

    sts.b /= nums;
    sts.r /= nums;
    sts.g /= nums;
}

//r g b digtial gain
void rgb_dgain(cv::Mat& src, float r_gain, float g_gain, float b_gain)
{
    if (src.empty()) {
        return;
    }
    for (int h = 0; h < src.rows; ++h) {
        uchar* uc_pixel = src.data + h * src.step;
        for (int w = 0; w < src.cols; ++w) {
            uc_pixel[0]  = clip<uint8_t>(uc_pixel[0] * b_gain, 255);
            uc_pixel[1]  = clip<uint8_t>(uc_pixel[1] * g_gain, 255);
            uc_pixel[2]  = clip<uint8_t>(uc_pixel[2] * r_gain, 255);
            uc_pixel += 3;
        }
    }
}

//gray world awb amd lum banlance for four channeal images
void awb_and_lum_banlance(std::vector<cv::Mat*> srcs)
{
    BgrSts sts[4];
    int    gray[4] = {0, 0, 0, 0};
    float    gray_ave = 0;

    if (srcs.size() != 4) {
        return;
    }

    for (int i = 0; i < 4; ++i) {
        if (srcs[i] == nullptr) {
            return;
        }
        rgb_info_statics(*srcs[i], sts[i]);
        gray[i] = sts[i].r * 20 + sts[i].g * 60 + sts[i].b;
        gray_ave += gray[i];
    }
    
    gray_ave /= 4;

    for (int i = 0; i < 4; ++i) {
        float lum_gain = gray_ave / gray[i];
        float r_gain = sts[i].g * lum_gain / sts[i].r;
        float g_gain = lum_gain;
        float b_gain = sts[i].g * lum_gain / sts[i].b;
        //std::cout << "gains : " << r_gain << " | " << g_gain << " | " << b_gain << "\r\n";
        rgb_dgain(*srcs[i], r_gain, g_gain, b_gain);
    }
}

