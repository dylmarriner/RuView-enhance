/**
 * @file csi_gain_lock.h
 * @brief Optional receiver gain lock for CSI (CONFIG_CSI_GAIN_LOCK, default off).
 *
 * The receiver's AGC and FFT-scale gains change from packet to packet, which
 * scales every CSI amplitude. After observing CONFIG_CSI_GAIN_LOCK_SAMPLES
 * frames this module takes the median AGC/FFT gain and forces the receiver to
 * hold it, so later amplitudes are directly comparable. The sensing server
 * already cancels gain steps by normalising each frame, so this is an
 * experiment for links where quantisation of weak frames matters more than
 * the risk below.
 *
 * Risk: a forced gain is one value for every transmitter. Frames much weaker
 * or stronger than the median can saturate or fall under the noise floor and
 * be lost. The lock therefore releases itself and re-collects when the frame
 * stream goes quiet for CSI_GAIN_LOCK_GAP_US, and it re-collects after every
 * channel change.
 *
 * Gain control uses Espressif's esp_csi_gain_ctrl component (Apache-2.0,
 * prebuilt for ESP32-S3 / ESP-IDF 5.4); only csi_gain_lock_hw.c touches it.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "sdkconfig.h"
#include "esp_wifi_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Frames without a callback for this long release a held lock. */
#define CSI_GAIN_LOCK_GAP_US (5 * 1000 * 1000)

/** Maximum samples the state machine can buffer. */
#define CSI_GAIN_LOCK_MAX_SAMPLES 256

#ifdef CONFIG_CSI_GAIN_LOCK

/** Feed one received frame. `now_us` is a monotonic microsecond clock. */
void csi_gain_lock_on_frame(const wifi_pkt_rx_ctrl_t *rx, int64_t now_us);

/** Forget samples and release a held lock (e.g. after a channel change). */
void csi_gain_lock_reset(void);

/** True while the receiver gain is being forced. */
bool csi_gain_lock_is_locked(void);

/* Hardware boundary (csi_gain_lock_hw.c, or a fake in host tests). */
void csi_gain_lock_hw_read(const wifi_pkt_rx_ctrl_t *rx, uint8_t *agc, int8_t *fft);
bool csi_gain_lock_hw_force(uint8_t agc, int8_t fft);   /* false on failure */
void csi_gain_lock_hw_release(void);

#else

static inline void csi_gain_lock_on_frame(const wifi_pkt_rx_ctrl_t *rx, int64_t now_us)
{
    (void)rx;
    (void)now_us;
}
static inline void csi_gain_lock_reset(void) {}
static inline bool csi_gain_lock_is_locked(void) { return false; }

#endif

#ifdef __cplusplus
}
#endif
