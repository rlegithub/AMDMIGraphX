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
#include <migraphx/errors.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/half.hpp>
#include <fstream>
#include <cstdlib>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_gptoss_moe::compute_shape(std::vector<shape> inputs) const
{
    // After lowering, an output buffer is appended (9 inputs); before that there
    // are the 8 op inputs. Either way report the op's output shape (= hidden shape).
    if(inputs.size() < 8)
        MIGRAPHX_THROW("gpu::gptoss_moe: expected >=8 inputs, got " +
                       std::to_string(inputs.size()));
    return inputs.front();
}

argument
hip_gptoss_moe::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    // First-light: throw (propagates across the DLL boundary to the runner's
    // catch + stderr; fprintf from migraphx_device.dll does not).
    if(args.size() != 9)
        MIGRAPHX_THROW("gpu::gptoss_moe::compute: expected 9 args (8 inputs + output), got " +
                       std::to_string(args.size()));

    const auto& hidden  = args[0];
    const auto& router  = args[1];
    const auto& fc1_w   = args[2];
    const auto& fc1_s   = args[3];
    const auto& fc2_w   = args[4];
    const auto& fc2_s   = args[5];
    const auto& fc1_b   = args[6];
    const auto& fc2_b   = args[7];
    const auto& output  = args[8];

    // Surface the shapes/types we actually received so a mismatch is diagnosable.
    auto sstr = [](const shape& s) { return s.type_string() + to_string_range(s.lens()); };
    if(hidden.get_shape().type() != shape::float_type)
        MIGRAPHX_THROW("gpu::gptoss_moe: hidden must be float (got " + sstr(hidden.get_shape()) +
                       ")");
    if(fc1_w.get_shape().type() != shape::uint32_type or
       fc2_w.get_shape().type() != shape::uint32_type)
        MIGRAPHX_THROW("gpu::gptoss_moe: fc weights must be uint32 (got fc1=" +
                       sstr(fc1_w.get_shape()) + " fc2=" + sstr(fc2_w.get_shape()) + ")");

    const int S      = static_cast<int>(hidden.get_shape().lens()[0]);
    const int E      = op.num_experts;
    const int top_k  = op.top_k;
    // Worst case: every token routes to top_k distinct experts.
    const int maxtok = S * top_k;

    // Gated input-binding probe: dump shapes + first elements of each arg the EP
    // hands us, so we can compare against the oracle's known-good bins. Enable with
    // MIGRAPHX_MOE_DUMP=1. Writes once (first call) to C:/Users/atgsc/moe_ep_inputs.txt.
    static int moe_call = 0;
    const int this_call = moe_call++;
    if(std::getenv("MIGRAPHX_MOE_DUMP") != nullptr and this_call < 2)
    {
        {
            ctx.finish();
            const std::string tag = "_call" + std::to_string(this_call);
            std::ofstream f("C:/Users/atgsc/moe_ep_inputs" + tag + ".txt");
            const char* names[9] = {"hidden","router","fc1_w","fc1_s","fc2_w","fc2_s","fc1_b","fc2_b","output"};
            for(int i = 0; i < 9; i++)
            {
                const auto& a   = args[i];
                auto host       = from_gpu(a);
                f << names[i] << " " << a.get_shape().type_string()
                  << to_string_range(a.get_shape().lens())
                  << " strides" << to_string_range(a.get_shape().strides()) << " first=";
                auto sh = a.get_shape();
                std::size_t n = std::min<std::size_t>(8, sh.elements());
                if(sh.type() == shape::float_type) { auto* p = reinterpret_cast<float*>(host.data()); for(std::size_t k=0;k<n;k++) f << p[k] << " "; }
                else if(sh.type() == shape::uint32_type) { auto* p = reinterpret_cast<std::uint32_t*>(host.data()); for(std::size_t k=0;k<n;k++) f << p[k] << " "; }
                else if(sh.type() == shape::half_type) { auto* p = reinterpret_cast<half*>(host.data()); for(std::size_t k=0;k<n;k++) f << static_cast<float>(p[k]) << " "; }
                f << "\n";
                // also dump full raw bytes for offline numpy comparison
                std::string bp = std::string("C:/Users/atgsc/moe_ep_") + names[i] + tag + ".bin";
                std::ofstream bf(bp, std::ios::binary);
                bf.write(host.data(), a.get_shape().bytes());
                bf.close();
            }
            f.close();
        }
    }

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
                       fc1_b,
                       fc2_b,
                       topk_w,
                       topk_e,
                       etok_ids,
                       etok_cnt,
                       p);

    if(std::getenv("MIGRAPHX_MOE_DUMP") != nullptr and this_call < 2)
    {
        ctx.finish();
        auto host = from_gpu(output);
        std::ofstream bf("C:/Users/atgsc/moe_ep_output_call" + std::to_string(this_call) + ".bin",
                         std::ios::binary);
        bf.write(host.data(), output.get_shape().bytes());
        bf.close();
    }
    return output;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
