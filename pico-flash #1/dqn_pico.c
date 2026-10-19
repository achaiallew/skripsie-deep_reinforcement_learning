#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include "pico/stdlib.h"
#include "model_weights.h"

#define state_size (4 * 56 * 56)

static uint8_t state[state_size];
static float   in_f[state_size];        // the picture as decimals 0..1
static float   c1[16 * 26 * 26];        // conv1 output
static float   c2[32 * 11 * 11];        // conv2 output
static float   c3[512];                 // conv3 output (32*4*4)
static float   h[128];                  // head output
static float   q_out[3];                // final scores
static int8_t  xq[512];                 // int8 version of an input

// Slide a k x k window over the picture, multiply, add bias, apply ReLU.
static void conv_relu(const float *in, int inC, int inH, int inW,
                      const float *w, const float *b,
                      float *out, int outC, int k, int stride) {
    int outH = (inH - k) / stride + 1;
    int outW = (inW - k) / stride + 1;
    for (int oc = 0; oc < outC; oc++)
        for (int oy = 0; oy < outH; oy++)
            for (int ox = 0; ox < outW; ox++) {
                float sum = b[oc];
                for (int ic = 0; ic < inC; ic++)
                    for (int ky = 0; ky < k; ky++)
                        for (int kx = 0; kx < k; kx++)
                            sum += in[(ic * inH + oy * stride + ky) * inW + ox * stride + kx]
                                 * w[((oc * inC + ic) * k + ky) * k + kx];
                out[(oc * outH + oy) * outW + ox] = sum > 0.0f ? sum : 0.0f;
            }
}

// "Dynamic" step: look at the numbers, pick a scale, turn them into int8.
static void quantize_input(const float *x, int n, float *scale, int *zp) {
    float mn = 0.0f, mx = 0.0f;                 // range always includes 0
    for (int i = 0; i < n; i++) {
        if (x[i] < mn) mn = x[i];
        if (x[i] > mx) mx = x[i];
    }
    float s = (mx - mn) / 255.0f;
    if (s < FLT_EPSILON) s = FLT_EPSILON;
    int z = -128 - (int)nearbyintf(mn / s);
    if (z < -128) z = -128;
    if (z > 127)  z = 127;
    for (int i = 0; i < n; i++) {
        int q = (int)nearbyintf(x[i] / s) + z;
        if (q < -128) q = -128;
        if (q > 127)  q = 127;
        xq[i] = (int8_t)q;
    }
    *scale = s;
    *zp = z;
}

// int8 layer: out = (weights * input) * scales + bias
static void linear_int8(const float *x, int n_in, int n_out,
                        const int8_t *W, const float *w_scale, const float *bias,
                        float *out, int relu) {
    float s; int z;
    quantize_input(x, n_in, &s, &z);
    for (int o = 0; o < n_out; o++) {
        int32_t acc = 0;
        for (int j = 0; j < n_in; j++)
            acc += (int32_t)W[o * n_in + j] * ((int32_t)xq[j] - z);
        float v = (float)acc * w_scale[o] * s + bias[o];
        out[o] = (relu && v < 0.0f) ? 0.0f : v;
    }
}

static int run_network(const uint8_t *pixels) {
    for (int i = 0; i < state_size; i++) in_f[i] = pixels[i] / 255.0f;

    conv_relu(in_f, 4, 56, 56, conv1_w, conv1_b, c1, 16, 5, 2);   // -> 16x26x26
    conv_relu(c1, 16, 26, 26, conv2_w, conv2_b, c2, 32, 5, 2);    // -> 32x11x11
    conv_relu(c2, 32, 11, 11, conv3_w, conv3_b, c3, 32, 5, 2);    // -> 32x4x4

    linear_int8(c3, 512, 128, head_q, head_scale, head_bias, h, 1);
    linear_int8(h, 128, 3, fc1_q, fc1_scale, fc1_bias, q_out, 0);

    int best = 0;
    for (int i = 1; i < 3; i++) if (q_out[i] > q_out[best]) best = i;
    return best;
}

int main() {
    // Turn on USB Talking
    stdio_init_all();
    while (true) {
        // Wait for Full Picture
        for (int i = 0; i < state_size; i++)
            state[i] = (uint8_t)getchar();   // for all 12,544 bytes

        // Select Action
        int action = run_network(state);

        // Send Answer Back to PC
        printf("%d\n", action);
    }
}