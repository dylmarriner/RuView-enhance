/**
 * @file csi_gain_lock.c
 * @brief Gain-lock state machine: collect -> lock -> (release on gap/reset).
 *
 * Clean-room design from the behaviour described in Espressif's esp-csi
 * documentation (gain baseline = median of observed gains, then force it).
 * Hardware access is isolated in csi_gain_lock_hw.c so this logic is host
 * testable.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "csi_gain_lock.h"

#ifdef CONFIG_CSI_GAIN_LOCK

#include <stdlib.h>
#include <string.h>

#ifndef CONFIG_CSI_GAIN_LOCK_SAMPLES
#define CONFIG_CSI_GAIN_LOCK_SAMPLES 100
#endif

#if CONFIG_CSI_GAIN_LOCK_SAMPLES > CSI_GAIN_LOCK_MAX_SAMPLES
#error "CONFIG_CSI_GAIN_LOCK_SAMPLES exceeds CSI_GAIN_LOCK_MAX_SAMPLES"
#endif

static uint8_t s_agc[CONFIG_CSI_GAIN_LOCK_SAMPLES];
static int8_t  s_fft[CONFIG_CSI_GAIN_LOCK_SAMPLES];
static unsigned s_count;
static bool     s_locked;
static int64_t  s_last_frame_us;

static int cmp_u8(const void *a, const void *b)
{
    return (int)*(const uint8_t *)a - (int)*(const uint8_t *)b;
}

static int cmp_i8(const void *a, const void *b)
{
    return (int)*(const int8_t *)a - (int)*(const int8_t *)b;
}

static void release_lock(void)
{
    if (s_locked) {
        csi_gain_lock_hw_release();
        s_locked = false;
    }
    s_count = 0;
}

void csi_gain_lock_reset(void)
{
    release_lock();
}

bool csi_gain_lock_is_locked(void)
{
    return s_locked;
}

void csi_gain_lock_on_frame(const wifi_pkt_rx_ctrl_t *rx, int64_t now_us)
{
    if (rx == NULL) {
        return;
    }
    /* A long silence while forced is the signature of a gain that lost the
     * link: let the AGC run again and re-collect rather than stay deaf. */
    if (s_locked && s_last_frame_us != 0 && (now_us - s_last_frame_us) > CSI_GAIN_LOCK_GAP_US) {
        release_lock();
    }
    s_last_frame_us = now_us;
    if (s_locked) {
        return;
    }

    uint8_t agc = 0;
    int8_t fft = 0;
    csi_gain_lock_hw_read(rx, &agc, &fft);
    s_agc[s_count] = agc;
    s_fft[s_count] = fft;
    if (++s_count < CONFIG_CSI_GAIN_LOCK_SAMPLES) {
        return;
    }

    qsort(s_agc, s_count, sizeof(s_agc[0]), cmp_u8);
    qsort(s_fft, s_count, sizeof(s_fft[0]), cmp_i8);
    uint8_t agc_med = s_agc[s_count / 2];
    int8_t fft_med = s_fft[s_count / 2];
    s_count = 0;
    /* (0, 0) means "release" to the hardware layer, so it cannot be a target. */
    if (agc_med == 0 && fft_med == 0) {
        return;
    }
    s_locked = csi_gain_lock_hw_force(agc_med, fft_med);
}

#endif /* CONFIG_CSI_GAIN_LOCK */
