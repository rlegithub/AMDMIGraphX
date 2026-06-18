/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/gpu/gptoss_moe.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/device/moe.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_gptoss_moe::compute_shape(std::vector<shape> inputs) const
{
    // After lowering, an output buffer is appended (7 inputs); before that there
    // are the 6 op inputs. Use the first 6 either way and report the op's output
    // shape (= hidden_states shape).
    if(inputs.size() < 6)
        MIGRAPHX_THROW("gpu::gptoss_moe: expected >=6 inputs, got " +
                       std::to_string(inputs.size()));
    return inputs.front();
}

argument
hip_gptoss_moe::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    const auto& hidden  = args[0];
    const auto& router  = args[1];
    const auto& fc1_w   = args[2];
    const auto& fc1_s   = args[3];
    const auto& fc2_w   = args[4];
    const auto& fc2_s   = args[5];
    const auto& output  = args[6];

    const int S      = static_cast<int>(hidden.get_shape().lens()[0]);
    const int E      = op.num_experts;
    const int top_k  = op.top_k;
    // Worst case: every token routes to top_k distinct experts.
    const int maxtok = S * top_k;

    // Scratch (device). For first-light these are allocated per call via
    // allocate_gpu (bypasses memory_coloring; revisit with the workspace-via
    // compile() idiom once correctness is proven).
    auto topk_w   = allocate_gpu(shape{shape::float_type, {static_cast<std::size_t>(S), static_cast<std::size_t>(top_k)}});
    auto topk_e   = allocate_gpu(shape{shape::int32_type, {static_cast<std::size_t>(S), static_cast<std::size_t>(top_k)}});
    auto etok_ids = allocate_gpu(shape{shape::int32_type, {static_cast<std::size_t>(E), static_cast<std::size_t>(maxtok)}});
    auto etok_cnt = allocate_gpu(shape{shape::int32_type, {static_cast<std::size_t>(E)}});

    device::moe_params p;
    p.num_tokens            = S;
    p.hidden_size           = op.hidden_size;
    p.intermediate_size     = op.intermediate_size;
    p.num_experts           = E;
    p.top_k                 = top_k;
    p.max_tokens_per_expert = maxtok;
    p.swiglu_alpha          = op.swiglu_alpha;
    p.swiglu_beta           = op.swiglu_beta;
    p.swiglu_limit          = op.swiglu_limit;

    device::gptoss_moe(ctx.get_stream().get(),
                       output,
                       hidden,
                       router,
                       fc1_w,
                       fc1_s,
                       fc2_w,
                       fc2_s,
                       topk_w,
                       topk_e,
                       etok_ids,
                       etok_cnt,
                       p);
    return output;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
