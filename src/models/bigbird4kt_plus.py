from matplotlib.style import context
import torch
import torch.nn as nn

import math
import numpy as np

# BigbirdBlockSparseAttention
class BigBirdBlockSparseAttention(nn.Module):
    def __init__(self, hidden_size, n_splits, max_seq_len, config, seed=42):
        #config.hidden_size % config.num_attention_heads == 0
        super().__init__()

        self.max_seqlen = max_seq_len # 100
        self.seed = seed # 42
        self.num_attention_heads = n_splits # 16
        self.num_random_blocks = config.num_random_blocks # 3
        self.block_size = config.block_size # 5

        self.attention_head_size = int(hidden_size / n_splits) # 512/16 = 32
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 16 * 32 = 512

        # 512 -> 512, 확인 필요
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=True)

    def forward(
        self,
        Q,
        K,
        V,
        mask, #원래 없는데 추가함
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
        output_attentions=None,
    ):
        # Currently this `class` can't be used in decoder.

        # |Q| = |K| = |V| = (bs, sq, hs) = (bs, 100, 512)
        hidden_states = Q
        # |hidden_states| = (bs, sq, hs)
        batch_size, seqlen, _ = hidden_states.size()
        # batch_size = bs
        # seqlen = sq = 100
        to_seq_length = from_seq_length = seqlen
        # to_seq_length = from_seq_length = seqlen = sq = 100
        from_block_size = to_block_size = self.block_size
        # from_block_size = to_block_size = self.block_size = 5

        # from_seq_length % from_block_size == 0
        # to_seq_length % to_block_size == 0

        # |self.query(Q)| = |self.key(K)| = |self.value(V)| = (bs, sq, hs)
        query_layer = self.transpose_for_scores(self.query(Q))
        key_layer = self.transpose_for_scores(self.key(K))
        value_layer = self.transpose_for_scores(self.value(V))
        # |query_layer| = (bs, self.num_attention_heads, sq, self.attention_head_size) 
        # = (bs, 16, 100, 32)
        # = |key_layer| = |value_layer|

        context_layer, attention_probs = self.bigbird_block_sparse_attention(
            query_layer, #(bs, self.num_attention_heads, sq, self.attention_head_size) = (bs, 16, 100, 32)
            key_layer, #(bs, self.num_attention_heads, sq, self.attention_head_size) = (bs, 16, 100, 32)
            value_layer, #(bs, self.num_attention_heads, sq, self.attention_head_size) = (bs, 16, 100, 32)
            band_mask, # None
            from_mask, # None
            to_mask, # None
            from_blocked_mask, # None
            to_blocked_mask, # None
            self.num_attention_heads, # 16
            self.num_random_blocks, # 3
            self.attention_head_size, # 32
            from_block_size, # 5
            to_block_size, # 5
            batch_size, # bs
            from_seq_length, # 100
            to_seq_length, # 100
            seed=self.seed, # 42
            plan_from_length=None, # None
            plan_num_rand_blocks=None, # None
            output_attentions=output_attentions, # None
        )

        context_layer = context_layer.contiguous().view(batch_size, from_seq_length, -1)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    # bigbird attention
    def bigbird_block_sparse_attention(
        self,
        query_layer, #(bs, self.num_attention_heads, sq, self.attention_head_size) = (bs, 16, 100, 32)
        key_layer, #(bs, self.num_attention_heads, sq, self.attention_head_size) = (bs, 16, 100, 32)
        value_layer, #(bs, self.num_attention_heads, sq, self.attention_head_size) = (bs, 16, 100, 32)
        band_mask, # None
        from_mask, # None
        to_mask, # None
        from_blocked_mask, # None
        to_blocked_mask, # None
        n_heads, # 16
        n_rand_blocks, # 3
        attention_head_size, # 32
        from_block_size, # 5
        to_block_size, # 5
        batch_size, # bs
        from_seq_len, # 100
        to_seq_len, # 100
        seed, # 42
        plan_from_length, # None
        plan_num_rand_blocks, # None
        output_attentions, # None
    ):

        # BigBird block-sparse attention as suggested in paper

        # ITC:
        #     global tokens: 2 x block_size
        #     window tokens: 3 x block_size
        #     random tokens: num_rand_tokens x block_size

        # Note:
        #     1) Currently, ETC is not supported.
        #     2) Window size is fixed to 3 blocks & it can be changed only by
        #     changing `block_size`.
        #     3) Number of global blocks are fixed (2 blocks here) & global tokens can be
        #     controlled only by `block_size`.

        # attention is calculated separately for q[0], q[1], q[2:-2], q[-2], q[-1] in order to use special trick of shifting tokens (for calculating sliding attention)
        # hence following code can be divided into 5 parts.

        rsqrt_d = 1 / math.sqrt(attention_head_size)
        # rsqrt_d = 0.17677669529663687
        bsz = batch_size
        # bsz = bs
        attn_mask_penalty = -10000.0

        # generate random attention and corresponding masks
        np.random.seed(seed)
        # seed = 42

        # Gives the plan of where to put random attention.
        plan_from_length, plan_num_rand_blocks = self._get_rand_attn_plan(
            from_seq_len, # 100
            from_block_size, # 10
            n_rand_blocks # 3
        )
        # plan_from_length = [80, 100]
        # plan_num_rand_blocks = [1, 2]
        # 아래 메쏘드(_bigbird_block_rand_mask_with_head)에 사용됨

        # Create adjacency list of random attention.
        rand_attn = self._bigbird_block_rand_mask_with_head(
            from_seq_length=from_seq_len, # 100
            to_seq_length=to_seq_len, # 100
            from_block_size=from_block_size, # 10
            to_block_size=to_block_size, # 10
            num_heads=n_heads, # 16
            plan_from_length=plan_from_length, # [80, 100]
            plan_num_rand_blocks=plan_num_rand_blocks, # [1, 2]
        )
        # len(rand_attn) = 16
        # rand_attn[0].shape = (18, 3) -> [[5 8 4], [9 6 7], ... [4 2 5]]

        rand_attn = np.stack(rand_attn, axis=0)
        # |rand_attn| = (16, 18, 3), numpy array
        rand_attn = torch.tensor(rand_attn, device=query_layer.device, dtype=torch.long)
        # |rand_attn| = (16, 18, 3), torch.tensor
        rand_attn.unsqueeze_(0)
        # |rand_attn| = (1, 16, 18, 3), torch.tensor
        rand_attn = torch.cat([rand_attn for _ in range(batch_size)], dim=0)
        # |rand_attn| = (bs, 16, 18, 3), torch.tensor

        rand_mask = self._create_rand_mask_from_inputs(
            from_blocked_mask, # None
            to_blocked_mask, # None
            rand_attn, # |rand_attn| = (bs, 16, 18, 3)
            n_heads, # 16
            n_rand_blocks, # 3
            bsz, # bs
            from_seq_len, # 100
            from_block_size # 5
        )
        # 여기서 문제 발생
        # to_blocked_mask에 어떤 값이 들어가야하는지?? 이것이 중요

        blocked_query_matrix = query_layer.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, -1)
        blocked_key_matrix = key_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
        blocked_value_matrix = value_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)

        # preparing block for randn attn
        gathered_key = self.torch_gather_b2(blocked_key_matrix, rand_attn)
        gathered_key = gathered_key.view(
            bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1
        )  # [bsz, n_heads, to_seq_len//to_block_size-2, n_rand_blocks, to_block_size, -1]
        gathered_value = self.torch_gather_b2(blocked_value_matrix, rand_attn)
        gathered_value = gathered_value.view(
            bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1
        )  # [bsz, n_heads, to_seq_len//to_block_size-2, n_rand_blocks, to_block_size, -1]

        # 1st PART
        # 1st block (global block) attention scores
        # q[0] x (k[0], k[1], k[2], k[3], k[4] .... )

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, to_seq_len]
        first_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 0], key_layer, ndim=4)

        first_product = first_product * rsqrt_d
        first_product += (1.0 - to_mask) * attn_mask_penalty
        first_attn_weights = nn.functional.softmax(
            first_product, dim=-1
        )  # [bsz, n_heads, from_block_size, to_seq_len]

        # [bsz, n_heads, from_block_size, to_seq_len] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, -1]
        first_context_layer = self.torch_bmm_nd(first_attn_weights, value_layer, ndim=4)
        first_context_layer.unsqueeze_(2)

        # 2nd PART
        # 2nd block attention scores
        # q[1] x (sliding_keys, random_keys, global_keys)
        # sliding key blocks -> 2nd, 3rd blocks
        # global key blocks -> 1st block

        second_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, 1],
                blocked_key_matrix[:, :, 2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, 0],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1]
        second_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, 1],
                blocked_value_matrix[:, :, 2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, 0],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1]

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]
        second_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, 1], second_key_mat, ndim=4)
        second_seq_pad = torch.cat(
            [
                to_mask[:, :, :, : 3 * to_block_size],
                to_mask[:, :, :, -to_block_size:],
                to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size]),
            ],
            dim=3,
        )
        second_rand_pad = torch.cat(
            [
                rand_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]),
                rand_mask[:, :, 0],
            ],
            dim=3,
        )
        second_product = second_product * rsqrt_d
        second_product += (1.0 - torch.minimum(second_seq_pad, second_rand_pad)) * attn_mask_penalty
        second_attn_weights = nn.functional.softmax(
            second_product, dim=-1
        )  # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]

        # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, -1]
        second_context_layer = self.torch_bmm_nd(second_attn_weights, second_value_mat, ndim=4)

        second_context_layer.unsqueeze_(2)

        # 3rd PART
        # Middle blocks attention scores
        # q[-2:2] x (sliding_keys, random_keys, global_keys)
        # sliding attn is calculated using special trick of shifting tokens as discussed in paper
        # random keys are generated by taking random indices as per `rand_attn`
        # global keys -> 1st & last block

        exp_blocked_key_matrix = torch.cat(
            [blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2], blocked_key_matrix[:, :, 3:-1]], dim=3
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        exp_blocked_value_matrix = torch.cat(
            [blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2], blocked_value_matrix[:, :, 3:-1]],
            dim=3,
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2]

        # sliding attention scores for q[-2:2]
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [b, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        inner_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, exp_blocked_key_matrix, ndim=5)
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, 3*to_block_size]
        inner_band_product = inner_band_product * rsqrt_d

        # randn attention scores for q[-2:2]
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, from_seq_len//from_block_size-4, n_rand_blocks*to_block_size, -1]
        rand_band_product = self.torch_bmm_nd_transpose(middle_query_matrix, gathered_key[:, :, 1:-1], ndim=5)
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, n_rand_blocks*to_block_size]
        rand_band_product = rand_band_product * rsqrt_d

        # Including 1st block (since it's global)
        first_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, 0]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size]
        first_band_product = first_band_product * rsqrt_d

        # Including last block (since it's global)
        last_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, -1]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size]
        last_band_product = last_band_product * rsqrt_d

        # masking padded tokens
        inner_band_product += (1.0 - band_mask) * attn_mask_penalty
        first_band_product += (1.0 - to_mask[:, :, :, :to_block_size].unsqueeze(3)) * attn_mask_penalty
        last_band_product += (1.0 - to_mask[:, :, :, -to_block_size:].unsqueeze(3)) * attn_mask_penalty
        rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * attn_mask_penalty

        # completing attention scores matrix for all q[-2:2]
        band_product = torch.cat(
            [first_band_product, inner_band_product, rand_band_product, last_band_product], dim=-1
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, (5+n_rand_blocks)*to_block_size]

        # safely doing softmax since attention matrix is completed
        attn_weights = nn.functional.softmax(
            band_product, dim=-1
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, (5+n_rand_blocks)*to_block_size]

        # contribution of sliding keys
        # [bsz, n_heads, m//from_block_size-4, from_block_size, 3*to_block_size] x [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        context_layer = self.torch_bmm_nd(
            attn_weights[:, :, :, :, to_block_size : 4 * to_block_size], exp_blocked_value_matrix, ndim=5
        )
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]

        # adding contribution of random keys
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, n_rand_blocks*to_block_size] x [bsz, n_heads, from_seq_len//from_block_size-4, n_rand_blocks*to_block_size, -1]
        context_layer += self.torch_bmm_nd(
            attn_weights[:, :, :, :, 4 * to_block_size : -to_block_size], gathered_value[:, :, 1:-1], ndim=5
        )
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]

        # adding contribution of global keys
        context_layer += torch.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, :to_block_size], blocked_value_matrix[:, :, 0]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]
        context_layer += torch.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, -to_block_size:], blocked_value_matrix[:, :, -1]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]

        # 4th PART
        # last 2nd token attention scores
        # q[-2] x (sliding_keys, random_keys, global_keys)
        # sliding key blocks -> last 3 blocks
        # global key block -> 1st block
        # random key block -> based on indices stored in `randn_attn`

        second_last_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, -3],
                blocked_key_matrix[:, :, -2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, -1],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+n_random_blocks)*to_block_size, -1]
        second_last_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, -3],
                blocked_value_matrix[:, :, -2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, -1],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+r)*to_block_size, -1]

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]
        second_last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -2], second_last_key_mat, ndim=4)
        second_last_seq_pad = torch.cat(
            [
                to_mask[:, :, :, :to_block_size],
                to_mask[:, :, :, -3 * to_block_size :],
                to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size]),
            ],
            dim=3,
        )
        second_last_rand_pad = torch.cat(
            [
                rand_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]),
                rand_mask[:, :, -1],
            ],
            dim=3,
        )
        second_last_product = second_last_product * rsqrt_d
        second_last_product += (1.0 - torch.minimum(second_last_seq_pad, second_last_rand_pad)) * attn_mask_penalty
        second_last_attn_weights = nn.functional.softmax(
            second_last_product, dim=-1
        )  # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]

        # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, -1]
        second_last_context_layer = self.torch_bmm_nd(second_last_attn_weights, second_last_value_mat, ndim=4)
        second_last_context_layer.unsqueeze_(2)

        # 5th PART
        # last block (global) attention scores
        # q[-1] x (k[0], k[1], k[2], k[3], .... )

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, to_seq_len]
        last_product = self.torch_bmm_nd_transpose(blocked_query_matrix[:, :, -1], key_layer, ndim=4)
        last_product = last_product * rsqrt_d
        last_product += (1.0 - to_mask) * attn_mask_penalty
        last_attn_weights = nn.functional.softmax(last_product, dim=-1)  # [bsz, n_heads, from_block_size, n]

        # [bsz, n_heads, from_block_size, to_seq_len] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, -1]
        last_context_layer = self.torch_bmm_nd(last_attn_weights, value_layer, ndim=4)
        last_context_layer.unsqueeze_(2)

        # combining representations of all tokens
        context_layer = torch.cat(
            [first_context_layer, second_context_layer, context_layer, second_last_context_layer, last_context_layer],
            dim=2,
        )
        context_layer = context_layer.view((bsz, n_heads, from_seq_len, -1)) * from_mask
        context_layer = torch.transpose(context_layer, 1, 2)

        # this is just for visualizing; forward pass doesn't depend on following code
        if output_attentions:
            # TODO(PVP): need to verify if below code is correct
            attention_probs = torch.zeros(
                bsz, n_heads, from_seq_len, to_seq_len, dtype=torch.float, device=context_layer.device
            )

            # 1st query block
            # corresponding to `first_context_layer`
            attention_probs[:, :, :from_block_size, :] = first_attn_weights  # all keys global

            # 2nd query block
            # corresponding to `second_context_layer`
            attention_probs[:, :, from_block_size : 2 * from_block_size, : 3 * to_block_size] = second_attn_weights[
                :, :, :, : 3 * to_block_size
            ]  # 1st three key blocks (global + sliding)
            attention_probs[:, :, from_block_size : 2 * from_block_size, -to_block_size:] = second_attn_weights[
                :, :, :, 3 * to_block_size : 4 * to_block_size
            ]  # last key block (global)
            # random keys
            for p1, i1, w1 in zip(range(bsz), rand_attn, second_attn_weights):
                # p1, i1, w1 corresponds to batch_dim i.e. following operation is done for each sequence in batch
                for p2, i2, w2 in zip(range(n_heads), i1, w1):
                    # p2, i2, w2 corresponds to head_dim i.e. following operation is done for each heads
                    attn_probs_view = attention_probs.view(
                        bsz,
                        n_heads,
                        from_seq_len // from_block_size,
                        from_block_size,
                        to_seq_len // to_block_size,
                        to_block_size,
                    )
                    right_slice = w2[:, 4 * to_block_size :]
                    attn_probs_view[p1, p2, 1, :, i2[0]] = right_slice.view(
                        from_block_size, n_rand_blocks, to_block_size
                    )

            # Middle query blocks
            # corresponding to `context_layer`
            # sliding keys
            for q_idx in range(from_seq_len // from_block_size - 4):
                attn_probs_view = attention_probs.view(
                    bsz,
                    n_heads,
                    from_seq_len // from_block_size,
                    from_block_size,
                    to_seq_len // to_block_size,
                    to_block_size,
                )[:, :, 2:-2, :, 1:-1, :]
                right_slice = attn_weights[:, :, q_idx, :, to_block_size : 4 * to_block_size]
                attn_probs_view[:, :, q_idx, :, q_idx : q_idx + 3, :] = right_slice.view(
                    bsz, n_heads, from_block_size, 3, to_block_size
                )  # inner_band_product
            # global keys (corresponding to 1st key block)
            attention_probs[:, :, 2 * from_block_size : -2 * from_block_size, :to_block_size] = attn_weights[
                :, :, :, :, :to_block_size
            ].view(
                bsz, n_heads, -1, to_block_size
            )  # first_band_product
            # global keys (corresponding to last key block)
            attention_probs[:, :, 2 * from_block_size : -2 * from_block_size, -to_block_size:] = attn_weights[
                :, :, :, :, -to_block_size:
            ].view(
                bsz, n_heads, -1, to_block_size
            )  # last_band_product
            # random keys
            for p1, i1, w1 in zip(range(bsz), rand_attn, attn_weights):
                # p1, i1, w1 corresponds to batch_dim i.e. following operation is done for each sequence in batch
                for p2, i2, w2 in zip(range(n_heads), i1, w1):
                    # p2, i2, w2 corresponds to head_dim i.e. following operation is done for each heads
                    for q_idx in range(1, len(i2) - 1):
                        attn_probs_view = attention_probs.view(
                            bsz,
                            n_heads,
                            from_seq_len // from_block_size,
                            from_block_size,
                            to_seq_len // to_block_size,
                            to_block_size,
                        )
                        right_slice = w2[q_idx - 1, :, 4 * to_block_size : -to_block_size]
                        attn_probs_view[p1, p2, q_idx + 1, :, i2[q_idx]] = right_slice.view(
                            from_block_size, n_rand_blocks, to_block_size
                        )

            # Second-last query block
            # corresponding to `second_last_context_layer`
            attention_probs[:, :, -2 * from_block_size : -from_block_size, :to_block_size] = second_last_attn_weights[
                :, :, :, :to_block_size
            ]  # 1st key block (global)
            attention_probs[
                :, :, -2 * from_block_size : -from_block_size, -3 * to_block_size :
            ] = second_last_attn_weights[
                :, :, :, to_block_size : 4 * to_block_size
            ]  # last three blocks (global + sliding)
            # random keys
            for p1, i1, w1 in zip(range(bsz), rand_attn, second_last_attn_weights):
                # p1, i1, w1 corresponds to batch_dim i.e. following operation is done for each sequence in batch
                for p2, i2, w2 in zip(range(n_heads), i1, w1):
                    # p2, i2, w2 corresponds to head_dim i.e. following operation is done for each heads
                    attn_probs_view = attention_probs.view(
                        bsz,
                        n_heads,
                        from_seq_len // from_block_size,
                        from_block_size,
                        to_seq_len // to_block_size,
                        to_block_size,
                    )
                    right_slice = w2[:, 4 * to_block_size :]
                    attn_probs_view[p1, p2, -2, :, i2[-1]] = right_slice.view(
                        from_block_size, n_rand_blocks, to_block_size
                    )

            # last query block
            # corresponding to `last_context_layer`
            attention_probs[:, :, -from_block_size:, :] = last_attn_weights  # all keys global

        else:
            attention_probs = None

        return context_layer, attention_probs

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def torch_bmm_nd(inp_1, inp_2, ndim=None):
        """Fast nd matrix multiplication"""
        # faster replacement of torch.einsum ("bhqk,bhkd->bhqd")
        return torch.bmm(inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:])).view(
            inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 1])
        )

    @staticmethod
    def torch_bmm_nd_transpose(inp_1, inp_2, ndim=None):
        """Fast nd matrix multiplication with transpose"""
        # faster replacement of torch.einsum (bhqd,bhkd->bhqk)
        return torch.bmm(
            inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:]).transpose(1, 2)
        ).view(inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 2]))

    @staticmethod
    def torch_gather_b2(params, indices):
        # this operation is equivalent to tf.gather when batch_dims=2

        if params.shape[:2] != indices.shape[:2]:
            raise ValueError(
                "Make sure that the first two dimensions of params and indices are identical,                 but"
                f" they are params: {params.shape[:2]} vs. indices: {indices.shape[:2]}"
            )
        num_indices_to_gather = indices.shape[-2] * indices.shape[-1]
        num_indices_to_pick_from = params.shape[2]

        indices_shift = (
            torch.arange(indices.shape[0] * indices.shape[1] * num_indices_to_gather, device=indices.device)
            // num_indices_to_gather
            * num_indices_to_pick_from
        )

        flattened_indices = indices.view(-1) + indices_shift
        flattened_params = params.reshape(-1, params.shape[-2], params.shape[-1])

        out_flattened = flattened_params.index_select(0, flattened_indices)

        out = out_flattened.reshape(params.shape[:2] + (num_indices_to_gather,) + params.shape[3:])
        return out

    @staticmethod
    def _create_rand_mask_from_inputs(
        from_blocked_mask, # None
        to_blocked_mask, # None
        rand_attn, # |rand_attn| = (bs, 16, 18, 3)
        num_attention_heads, # 16
        num_rand_blocks, # 3
        batch_size, # bs
        from_seq_length, # 100
        from_block_size, # 5
    ):
        """
        Create 3D attention mask from a 2D tensor mask.
        Args:
            from_blocked_mask: 2D Tensor of shape [batch_size, from_seq_length//from_block_size, from_block_size].
            to_blocked_mask: int32 Tensor of shape [batch_size, to_seq_length//to_block_size, to_block_size].
            rand_attn: [batch_size, num_attention_heads, from_seq_length//from_block_size-2, num_rand_blocks]
            num_attention_heads: int. Number of attention heads.
            num_rand_blocks: int. Number of random chunks per row.
            batch_size: int. Batch size for computation.
            from_seq_length: int. length of from sequence.
            from_block_size: int. size of block in from sequence.
        Returns:
            float Tensor of shape [batch_size, num_attention_heads, from_seq_length//from_block_size-2,
            from_block_size, num_rand_blocks*to_block_size].
        """
        num_windows = from_seq_length // from_block_size - 2 # 100 // 5 - 2 = 18
        # num_windows = 18
        
        # |to_blocked_mask| = (bs, to_seq_length//to_block_size, to_block_size) = (bs, 100//5, 5) = (bs, 20, 5)
        # |rand_attn| = (bs, 16, 18, 3)
        rand_mask = torch.stack(
            [p1[i1.flatten()] for p1, i1 in zip(to_blocked_mask, rand_attn)]
            )
        # |rand_mask| = ()
        print("rand_mask", rand_mask)
        rand_mask = rand_mask.view(batch_size, num_attention_heads, num_windows, num_rand_blocks * from_block_size)
        rand_mask = torch.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1], rand_mask)
        return rand_mask

    @staticmethod
    def _get_rand_attn_plan(
        from_seq_length, # 100
        from_block_size, # 10
        num_rand_blocks # 3
        ):
        """
        Gives the plan of where to put random attention.
        Args:
            from_seq_length: int. length of from sequence.
            from_block_size: int. size of block in from sequence.
            num_rand_blocks: int. Number of random chunks per row.
        Returns:
            plan_from_length: ending location of from block plan_num_rand_blocks: number of random ending location for
            each block
        """

        plan_from_length = []
        # [80, 100]
        plan_num_rand_blocks = []
        # [1, 2]

        if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(0)
        # 8 < 10
        elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((num_rand_blocks + 5) * from_block_size)) # apppend 80
            plan_num_rand_blocks.append(num_rand_blocks // 2) # append 3//2 = 1
            plan_from_length.append(from_seq_length) # append 100
            plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2)) # append 3 - 1 = 2
        else:
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks)

        return plan_from_length, plan_num_rand_blocks

    @staticmethod
    def _bigbird_block_rand_mask(
        from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1
    ):
        """
        Create adjacency list of random attention.
        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_rand_blocks: int. Number of random chunks per row.
            last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
            if positive then num_rand_blocks blocks chosen only up to last_idx.
        Returns:
            adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
        """
        # using this method when from_seq_length in [1024, 3072, 4096]

        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError("Error the number of blocks needs to be same!")

        rand_attn = np.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        last = to_seq_length // to_block_size - 1
        if last_idx > (2 * to_block_size):
            last = (last_idx // to_block_size) - 1

        r = num_rand_blocks  # shorthand
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
            elif i == 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
            elif i == from_seq_length // from_block_size - 3:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -3: should have been sliced till last-3
            elif i == from_seq_length // from_block_size - 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -4: should have been sliced till last-4
            else:
                if start > last:
                    start = last
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                elif (end + 1) == last:
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                else:
                    rand_attn[i - 1, :] = np.random.permutation(
                        np.concatenate((middle_seq[:start], middle_seq[end + 1 : last]))
                    )[:r]
        return rand_attn

    def _bigbird_block_rand_mask_with_head(
        self,
        from_seq_length, # 100
        to_seq_length, # 100
        from_block_size, # 10
        to_block_size, # 10
        num_heads, # 16
        plan_from_length, # [80, 100]
        plan_num_rand_blocks, # [1, 2]
        window_block_left=1,
        window_block_right=1,
        global_block_top=1,
        global_block_bottom=1,
        global_block_left=1,
        global_block_right=1,
    ):
        """
        Create adjacency list of random attention.
        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_heads: int. total number of heads.
            plan_from_length: list. plan from length where num_random_blocks are chosen from.
            plan_num_rand_blocks: list. number of rand blocks within the plan.
            window_block_left: int. number of blocks of window to left of a block.
            window_block_right: int. number of blocks of window to right of a block.
            global_block_top: int. number of blocks at the top.
            global_block_bottom: int. number of blocks at the bottom.
            global_block_left: int. Number of blocks globally used to the left.
            global_block_right: int. Number of blocks globally used to the right.
        Returns:
            adjacency list of size num_head where each element is of size from_seq_length//from_block_size-2 by
            num_rand_blocks
        """

        # Total number of blocks in the mmask
        num_blocks = from_seq_length // from_block_size
        # num_blocks = 100 // 10 = 10

        # Number of blocks per plan
        plan_block_length = np.array(plan_from_length) // from_block_size
        # plan_block_length = [8, 10]

        # till when to follow plan
        max_plan_idx = plan_from_length.index(from_seq_length)
        # max_plan_idx = 1

        # Random Attention adjacency list
        rand_attn = [
            np.zeros(
                (
                    num_blocks, # 10
                    np.sum(
                        # plan_num_rand_blocks = [1, 2]
                        plan_num_rand_blocks[: max_plan_idx + 1] # max_plan_idx = 1
                        )
                    # np.sum() => 3
                ), dtype=np.int32)
            # np.zeros(10, 3)
            for i in range(num_heads) # num_heads = 16
        ]
        # np.zeros(10, 3)을 16개 생성해서 리스트에 담음
        # len(rand_attn) = 16        

        # We will go iteratively over the plan blocks and pick random number of
        # Attention blocks from the legally allowed blocks
        for plan_idx in range(max_plan_idx + 1): # range(2)
            rnd_r_cnt = 0

            # 두번째 for iter부터 적용
            if plan_idx > 0:
                # set the row for all from_blocks starting from 0 to
                # plan_block_length[plan_idx-1]
                # column indx start fromm plan_block_length[plan_idx-1] and ends at
                # plan_block_length[plan_idx]
                if plan_num_rand_blocks[plan_idx] > 0:
                    rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                    curr_r_cnt = int(np.sum(plan_num_rand_blocks[: plan_idx + 1]))
                    for blk_rw_idx in range(global_block_top, plan_block_length[plan_idx - 1]):
                        for h in range(num_heads):
                            rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(
                                block_id=blk_rw_idx,
                                to_start_block_id=plan_block_length[plan_idx - 1],
                                to_end_block_id=plan_block_length[plan_idx],
                                num_rand_blocks=plan_num_rand_blocks[plan_idx],
                                window_block_left=window_block_left,
                                window_block_right=window_block_right,
                                global_block_left=global_block_left,
                                global_block_right=global_block_right,
                            )

                for pl_id in range(plan_idx):
                    if plan_num_rand_blocks[pl_id] == 0:
                        continue
                    for blk_rw_idx in range(plan_block_length[plan_idx - 1], plan_block_length[plan_idx]):
                        rnd_r_cnt = 0
                        to_start_block_id = 0
                        if pl_id > 0:
                            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
                            to_start_block_id = plan_block_length[pl_id - 1]
                        curr_r_cnt = int(np.sum(plan_num_rand_blocks[: pl_id + 1]))
                        for h in range(num_heads):
                            rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(
                                block_id=blk_rw_idx,
                                to_start_block_id=to_start_block_id,
                                to_end_block_id=plan_block_length[pl_id],
                                num_rand_blocks=plan_num_rand_blocks[pl_id],
                                window_block_left=window_block_left,
                                window_block_right=window_block_right,
                                global_block_left=global_block_left,
                                global_block_right=global_block_right,
                            )

            if plan_num_rand_blocks[plan_idx] == 0:
                continue
            # 가장 처음 idx는 0이므로, 첫번째는 여기부터 시작
            curr_r_cnt = int(np.sum(
                plan_num_rand_blocks[: plan_idx + 1])
                # 첫번째 iter: plan_idx는 0이므로, plan_num_rand_blocks[:1] = 1
                )
            # 첫번째 iter: curr_r_cnt = 1

            from_start_block_id = global_block_top
            # from_start_block_id = 1

            to_start_block_id = 0

            # 두번째부터 적용
            if plan_idx > 0:
                rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                from_start_block_id = plan_block_length[plan_idx - 1]
                to_start_block_id = plan_block_length[plan_idx - 1]

            # plan_block_length = [8, 10]
            # 첫번째 iter: range(1, 8), plan_block_length[plan_idx] = 8, 1부터 7까지 반복
            for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
                # np.zeros(10, 3)을 16개 생성해서 리스트에 담음, len(rand_attn) = 16
                for h in range(num_heads): #range(16)
                    # 첫 iter: rand_attn[0][1, 0:1]
                    #For a single row block get random row attention.
                    rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention( 
                        block_id=blk_rw_idx, # 첫번째 iter: 1
                        to_start_block_id=to_start_block_id, # 첫번째 iter: 0
                        to_end_block_id=plan_block_length[plan_idx], # 첫번째 iter: plan_block_length[0] = 8
                        num_rand_blocks=plan_num_rand_blocks[plan_idx], # 첫번째 iter: plan_num_rand_blocks[0] = 1
                        window_block_left=window_block_left, # 첫번째 iter: 1
                        window_block_right=window_block_right, # 첫번째 iter: 1
                        global_block_left=global_block_left, # 첫번째 iter: 1
                        global_block_right=global_block_right, # 첫번째 iter: 1
                    )

        for nh in range(num_heads):
            rand_attn[nh] = rand_attn[nh][global_block_top : num_blocks - global_block_bottom, :]

        return rand_attn

    @staticmethod
    def _get_single_block_row_attention(
        block_id,
        to_start_block_id,
        to_end_block_id,
        num_rand_blocks,
        window_block_left=1,
        window_block_right=1,
        global_block_left=1,
        global_block_right=1,
    ):
        """
        For a single row block get random row attention.
        Args:
            block_id: int. block id of row.
            to_start_block_id: int. random attention column start id.
            to_end_block_id: int. random attention column end id.
            num_rand_blocks: int. number of random blocks to be selected.
            window_block_left: int. number of blocks of window to left of a block.
            window_block_right: int. number of blocks of window to right of a block.
            global_block_left: int. Number of blocks globally used to the left.
            global_block_right: int. Number of blocks globally used to the right.
        Returns:
            row containing the random attention vector of size num_rand_blocks.
        """
        # list of to_blocks from which to choose random attention
        to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
        # permute the blocks
        perm_block = np.random.permutation(to_block_list)

        # illegal blocks for the current block id, using window
        illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))

        # Add blocks at the start and at the end
        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))

        # The second from_block cannot choose random attention on second last to_block
        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)

        # The second last from_block cannot choose random attention on second to_block
        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)

        selected_random_blokcs = []

        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blokcs.append(perm_block[i])
            if len(selected_random_blokcs) == num_rand_blocks:
                break
        return np.array(selected_random_blokcs, dtype=np.int32)

class EncoderBlock(nn.Module):

    def __init__(
        self,
        hidden_size, #512
        n_splits,
        use_leakyrelu,
        max_seq_len,
        config,
        dropout_p=.1,
    ):
        super().__init__()

        self.use_leakyrelu = use_leakyrelu

        self.attn = BigBirdBlockSparseAttention(
            hidden_size, 
            n_splits, 
            max_seq_len, 
            config
            )
        self.attn_norm = nn.LayerNorm(hidden_size) #attention을 위한 layerNorm
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if self.use_leakyrelu else nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        # |x| = (bs, n, emb_size), torch.float32
        # |mask| = (bs, n, n)

        # Pre-LN:
        z = self.attn_norm(x)
        # |z| = (bs, n, emb_size)

        # x+ means redisual connection
        z = x + self.attn_dropout(self.attn(Q=z,
                                            K=z,
                                            V=z, 
                                            mask=mask))
        # |z| = (bs, n, hs)

        z = z + self.fc_dropout(self.fc(self.fc_norm(z)))
        # |z| = (bs, n, hs)

        return z, mask


class MySequential(nn.Sequential):
    # 원래 sequential은 x 하나만 받을 수 있어서 상속받아 새로 정의
    # input을 *x로 받아서 튜플도 받을 수 있게 처리
    def forward(self, *x):
        # nn.Sequential class does not provide multiple input arguments and returns.
        # Thus, we need to define new class to solve this issue.
        # Note that each block has same function interface.

        for module in self._modules.values():
            x = module(*x)

        return x


class Bigbird4ktPlus(nn.Module):

    def __init__(
        self,
        num_q,
        num_r,
        num_pid,
        hidden_size,
        output_size,
        num_head,
        num_encoder,
        max_seq_len,
        device,
        use_leakyrelu,
        config,
        dropout_p=.1,
    ):
        self.num_q = num_q
        self.num_r = num_r + 2 # <PAD>와 <MASK>를 추가한만큼의 Emb값이 필요, 여기에 추가로 1을 더 더해줌
        self.num_pid = num_pid

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_head = num_head
        self.num_encoder = num_encoder
        self.max_seq_len = max_seq_len
        self.device = device
        self.use_leakyrelu = use_leakyrelu
        self.dropout_p = dropout_p

        super().__init__()

        # question embedding
        self.emb_q = nn.Embedding(self.num_q, self.hidden_size).to(self.device)
        # response embedding
        self.emb_r = nn.Embedding(self.num_r, self.hidden_size).to(self.device)
        # positional embedding
        self.emb_pid = nn.Embedding(self.num_pid, self.hidden_size).to(self.device)
        self.emb_p = nn.Embedding(self.max_seq_len, self.hidden_size).to(self.device)
        self.emb_dropout = nn.Dropout(self.dropout_p)

        # MySequential을 활용해 필요한만큼 encoder block을 만듦
        self.encoder = MySequential(
            *[EncoderBlock(
                hidden_size,
                num_head,
                self.use_leakyrelu,
                self.max_seq_len,
                config,
                dropout_p
              ) for _ in range(num_encoder)],
        )

        self.generator = nn.Sequential(
            nn.LayerNorm(hidden_size), # Only for Pre-LN Transformer.
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid() # binary
        )

    # positional embedding
    @torch.no_grad()
    def _positional_embedding(self, q):
        # |q| = (bs, n)
        # |r| = (bs, n)
        seq_len = q.size(1)
        # seq_len = (n,)
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand_as(q).to(self.device)
        # |pos| = (bs, n)
        
        pos_emb = self.emb_p(pos)
        # |emb| = (bs, n, hs)

        return pos_emb

    def forward(self, q, r, pid, mask):
        # |q| = (bs, n)
        # |r| = (bs, n)
        # |mask| = (bs, n)

        # Mask to prevent having attention weight on padding position.
        # with torch.no_grad():
        #     mask_enc = mask.unsqueeze(-1).expand(mask.size(0), mask.size(1), mask.size(1)).bool()
        #      # |mask_enc| = (bs, n, n), (bs, n_attn_head, n, attn_head_size)

        emb = self.emb_q(q) + self.emb_r(r) + self.emb_pid(pid) + self._positional_embedding(q)
        # |emb| = (bs, n, emb_size)

        z = self.emb_dropout(emb)
        # |z| = (bs, n, emb_size)

        # |mask_enc| = (bs, n, n)
        # |z| = (bs, n, emb_size)
        z, _ = self.encoder(z, mask)
        # |z| = (bs, n, hs)

        y_hat = self.generator(z)
        #|y_hat| = (bs, n, output_size=1)

        return y_hat