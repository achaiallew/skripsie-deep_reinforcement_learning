#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "pico/stdlib.h"
#include "model_weights.h"

#define STATE_SIZE (4 * 56 * 56)

static uint8_t state[STATE_SIZE];
static float   in_f[STATE_SIZE];        // the picture as decimals 0..1
static float   c1[16 * 26 * 26];        // conv1 output
static float   c2[32 * 11 * 11];        // conv2 output
static float   c3[512];                 // conv3 output (32*4*4)
static float   h[128];                  // head output
static float   q_out[3];                // final scores

// Conv with uint8 weights: real weight = delta * (q - z). Then ReLU.
static void conv_wo(const float *in, int inC, int inH, int inW,
                    const uint8_t *q, const float *delta, const uint8_t *z, const float *b,
                    float *out, int outC, int k, int stride) {
    int outH = (inH - k) / stride + 1;
    int outW = (inW - k) / stride + 1;
    for (int oc = 0; oc < outC; oc++) {
        int zp = z[oc];
        float d = delta[oc];
        for (int oy = 0; oy < outH; oy++)
            for (int ox = 0; ox < outW; ox++) {
                float acc = 0.0f;
                for (int ic = 0; ic < inC; ic++)
                    for (int ky = 0; ky < k; ky++)
                        for (int kx = 0; kx < k; kx++)
                            acc += in[(ic * inH + oy * stride + ky) * inW + ox * stride + kx]
                                 * (float)((int)q[((oc * inC + ic) * k + ky) * k + kx] - zp);
                float v = acc * d + b[oc];
                out[(oc * outH + oy) * outW + ox] = v > 0.0f ? v : 0.0f;
            }
    }
}

// Linear layer with uint8 weights.
static void linear_wo(const float *x, int n_in, int n_out,
                      const uint8_t *q, const float *delta, const uint8_t *z, const float *b,
                      float *out, int relu) {
    for (int o = 0; o < n_out; o++) {
        int zp = z[o];
        float acc = 0.0f;
        for (int j = 0; j < n_in; j++)
            acc += x[j] * (float)((int)q[o * n_in + j] - zp);
        float v = acc * delta[o] + b[o];
        out[o] = (relu && v < 0.0f) ? 0.0f : v;
    }
}

static int run_network(const uint8_t *pixels) {
    for (int i = 0; i < STATE_SIZE; i++) in_f[i] = pixels[i] / 255.0f;

    conv_wo(in_f, 4, 56, 56, conv1_q, conv1_delta, conv1_z, conv1_b, c1, 16, 5, 2);  // -> 16x26x26
    conv_wo(c1, 16, 26, 26, conv2_q, conv2_delta, conv2_z, conv2_b, c2, 32, 5, 2);   // -> 32x11x11
    conv_wo(c2, 32, 11, 11, conv3_q, conv3_delta, conv3_z, conv3_b, c3, 32, 5, 2);   // -> 32x4x4

    linear_wo(c3, 512, 128, head_q, head_delta, head_z, head_b, h, 1);
    linear_wo(h, 128, 3, fc1_q, fc1_delta, fc1_z, fc1_b, q_out, 0);

    int best = 0;
    for (int i = 1; i < 3; i++) if (q_out[i] > q_out[best]) best = i;
    return best;
}

int main() {
    stdio_init_all();
    while (true) {
        for (int i = 0; i < STATE_SIZE; i++) state[i] = (uint8_t)getchar();
        printf("%d\n", run_network(state));
    }
}