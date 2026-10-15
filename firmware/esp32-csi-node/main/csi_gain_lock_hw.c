/**
 * @file csi_gain_lock_hw.c
 * @brief ESP32-S3 binding of the gain-lock hardware boundary.
 *
 * Wraps Espressif's esp_csi_gain_ctrl component (SPDX-FileCopyrightText: 2025
 * Espressif Systems (Shanghai) CO LTD, Apache-2.0). Only the calls below are
 * used; the component's baseline/compensation helpers are not, because the
 * sensing server normalises gain per frame instead.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "csi_gain_lock.h"

#ifdef CONFIG_CSI_GAIN_LOCK

#include "esp_csi_gain_ctrl.h"
#include "esp_log.h"

static const char *TAG = "csi_gain";

void csi_gain_lock_hw_read(const wifi_pkt_rx_ctrl_t *rx, uint8_t *agc, int8_t *fft)
{
    esp_csi_gain_ctrl_get_rx_gain(rx, agc, fft);
}

bool csi_gain_lock_hw_force(uint8_t agc, int8_t fft)
{
    esp_err_t err = esp_csi_gain_ctrl_set_rx_force_gain(agc, fft);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "force gain failed: %s", esp_err_to_name(err));
        return false;
    }
    ESP_LOGI(TAG, "RX gain locked: agc=%u fft=%d", (unsigned)agc, (int)fft);
    return true;
}

void csi_gain_lock_hw_release(void)
{
    esp_csi_gain_ctrl_set_rx_force_gain(0, 0);
    ESP_LOGI(TAG, "RX gain lock released (AGC active)");
}

#endif /* CONFIG_CSI_GAIN_LOCK */
