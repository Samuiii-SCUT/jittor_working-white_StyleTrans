from typing import List, Optional, Tuple, Union
import jittor as jt


def randn_tensor(
    shape: Union[Tuple, List],
    seed: Optional[Union[List[int], int]] = None,
    dtype: Optional[str] = None,
):
    batch_size = shape[0]

    # if isinstance(seed, list) and len(seed) == 1:
    #     seed = seed[0]

    if isinstance(seed, list):
        shape = [1,] + shape
        latents = [
            seed_randn(shape, dtype, seed[i])
            for i in range(len(seed))
        ]
        latents = jt.concat(latents, dim=0)
    else:
        latents = seed_randn(shape, dtype, seed)

    return latents


def seed_randn(shape, dtype, seed: int):
    if seed is not None:
        jt.set_global_seed(seed)
    var = jt.randn(shape, dtype=dtype)
    return var


def register_attention_control_efficient(model_unet, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            #if not is_cross and self.injection_schedule is not None and (
            if self.injection_schedule is not None and (    
                    self.t in self.injection_schedule or self.t == 1000):
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)

                source_batch_size = int(q.shape[0] // 3)
                # inject unconditional
                q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                # inject conditional
                q[2 * source_batch_size:] = q[:source_batch_size]
                k[2 * source_batch_size:] = k[:source_batch_size]

                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)

            v = self.to_v(encoder_hidden_states)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model_unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)
            module = model_unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)

def register_conv_control_efficient(model_unet, injection_schedule, up_block_indices=[1]):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    for up_block_index in up_block_indices:
        if up_block_index not in [1, 2, 3]:
            raise ValueError(f"up_block_index: {up_block_indices} can not be found in unet")
        conv_module = model_unet.up_blocks[up_block_index].resnets[1]
        conv_module.forward = conv_forward(conv_module)
        setattr(conv_module, 'injection_schedule', injection_schedule)
