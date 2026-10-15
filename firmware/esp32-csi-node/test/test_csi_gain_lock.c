/** Deterministic host test for the CSI gain-lock state machine (fake hardware). */

#include "esp_stubs.h"
#include "csi_gain_lock.h"

#include <assert.h>
#include <stdio.h>

/* ---- fake hardware boundary ---- */
static uint8_t s_next_agc;
static int8_t  s_next_fft;
static int     s_force_calls, s_release_calls;
static uint8_t s_forced_agc;
static int8_t  s_forced_fft;
static bool    s_force_ok = true;

void csi_gain_lock_hw_read(const wifi_pkt_rx_ctrl_t *rx, uint8_t *agc, int8_t *fft)
{
    (void)rx;
    *agc = s_next_agc;
    *fft = s_next_fft;
}

bool csi_gain_lock_hw_force(uint8_t agc, int8_t fft)
{
    s_force_calls++;
    s_forced_agc = agc;
    s_forced_fft = fft;
    return s_force_ok;
}

void csi_gain_lock_hw_release(void)
{
    s_release_calls++;
}

static void feed(unsigned n, uint8_t agc, int8_t fft, int64_t *t_us)
{
    wifi_pkt_rx_ctrl_t rx;
    memset(&rx, 0, sizeof(rx));
    for (unsigned i = 0; i < n; i++) {
        s_next_agc = agc;
        s_next_fft = fft;
        *t_us += 20000;
        csi_gain_lock_on_frame(&rx, *t_us);
    }
}

int main(void)
{
    int64_t t = 1000000;
    const unsigned n = CONFIG_CSI_GAIN_LOCK_SAMPLES;

    /* Nothing is forced before enough frames have been seen. */
    feed(n - 1, 40, -2, &t);
    assert(!csi_gain_lock_is_locked() && s_force_calls == 0);

    /* The n-th frame locks onto the median, ignoring a few outliers. */
    feed(1, 40, -2, &t);
    assert(csi_gain_lock_is_locked());
    assert(s_force_calls == 1 && s_forced_agc == 40 && s_forced_fft == -2);

    /* Median, not mean: outliers below the majority do not move the target. */
    csi_gain_lock_reset();
    assert(!csi_gain_lock_is_locked() && s_release_calls == 1);
    feed(n / 4, 90, 5, &t);
    feed(n - n / 4, 30, -1, &t);
    assert(csi_gain_lock_is_locked());
    assert(s_forced_agc == 30 && s_forced_fft == -1);

    /* A long silence releases the lock; the next frames collect again. */
    int releases = s_release_calls;
    t += CSI_GAIN_LOCK_GAP_US + 1;
    feed(1, 31, -1, &t);
    assert(!csi_gain_lock_is_locked());
    assert(s_release_calls == releases + 1);

    /* A rejected force leaves the lock off and the module retrying. */
    csi_gain_lock_reset();
    s_force_ok = false;
    feed(n, 50, 0, &t);
    assert(!csi_gain_lock_is_locked());
    s_force_ok = true;
    feed(n, 50, 0, &t);
    assert(csi_gain_lock_is_locked());

    /* (0, 0) would mean "release" to the hardware: never forced. */
    csi_gain_lock_reset();
    int calls = s_force_calls;
    feed(n, 0, 0, &t);
    assert(!csi_gain_lock_is_locked() && s_force_calls == calls);

    puts("CSI gain lock state machine: PASS");
    return 0;
}
