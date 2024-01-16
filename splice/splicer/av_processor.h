/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */

#ifndef GTOE_EMULATION_AV_PROCESSOR_H
#define GTOE_EMULATION_AV_PROCESSOR_H

#include <vector>

extern "C" {
#include "libavutil/frame.h"
}

namespace gtoe_emulation {

class AVProcessor {
public:
    /*
     * An AVProcessor object is constructed and destructed in main thread while
     * Process() is called in packets-handler-thread. This is necessary
     * especially when you derive your own player as the window that handles
     * events has to be in main thread.
     */
    virtual ~AVProcessor() {};

    /*
     * Called in packets-handler-thread after encoded data packets from
     * different channels are gathered and decoded.
     *
     * If decoded frame from channel i is missing due to packet loss or bit
     * error, frames[i] is set to null or last non-null frame. You can
     * select a strategy in GTDE SIL vscode extension.
     */
    virtual std::vector<AVFrame *> Process(
        const std::vector<AVFrame *>& frames) = 0;

    /*
     * Do some cleanup in packets-handler-thread right before it ends. For a
     * player, it may tell main thread (or UI thread) to stop.
     */
    virtual void Finish() {};
};

} // namespace gtoe_emulation

#endif // GTOE_EMULATION_AV_PROCESSOR_H
