import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

# SeparableConv1D
class SeparableConv1D(nn.Module):
    def __init__(self, input_filters, output_filters, kernel_size):
        super().__init__()

        # input_filters = 512 <- hs
        # output_filters = 256 <- all_attn_h_size

        self.depthwise = nn.Conv1d(input_filters, input_filters, kernel_size=kernel_size, groups=input_filters, padding=kernel_size //2, bias = False)
        self.pointwise = nn.Conv1d(input_filters, output_filters, kernel_size=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_filters, 1))

        self.depthwise.weight.data.normal_(mean=0.0, std=0.02)
        self.pointwise.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, hidden_states):
        # |hidden_states| = (bs, hs, n)

        x = self.depthwise(hidden_states)
        # |x| = (bs, hs, n)

        x = self.pointwise(x)
        # |x| = (bs, hs/2(all_attn_h_size), n)

        x += self.bias
        # |x| = (bs, hs/2(all_attn_h_size), n)
        return x

# huggingface conv bert
class ForgettingMonotonicConvBertSelfAttention(nn.Module):
    # hidden % n_splits == 0
    def __init__(self, hidden_size, n_splits, dropout_p, head_ratio=2, conv_kernel_size=9):
        super().__init__()

        #n_splits = 16, head_ratio = 2
        new_num_attention_heads = n_splits // head_ratio
        self.num_attention_heads = new_num_attention_heads
        # self.new_num_attention_heads = 8

        self.head_ratio = head_ratio
        # self.head_ratio = 2

        self.conv_kernel_size = conv_kernel_size
        # self.conv_kernel_size = 9

        self.attention_head_size = hidden_size // n_splits
        # self.attention_head_size = 512//16 = 32

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # self.all_head_size = 32 * 8 = 256

        # q, k, v layers
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=False) # 512 -> 256
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=False) # 512 -> 256
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=False) # 512 -> 256

        # conv layers
        self.key_conv_attn_layer = SeparableConv1D(
            hidden_size, self.all_head_size, self.conv_kernel_size
        )
        self.conv_kernel_layer = nn.Linear(self.all_head_size, 
                                        self.num_attention_heads * self.conv_kernel_size # 8 * 9 = 72
                                        )
        self.conv_out_layer = nn.Linear(hidden_size, self.all_head_size)

        self.unfold = nn.Unfold(
            kernel_size=[self.conv_kernel_size, 1], padding=[int((self.conv_kernel_size - 1) / 2), 0]
        )
        
        self.gammas = nn.Parameter(torch.zeros(self.num_attention_heads, 1, 1))

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, Q, K, V, td, mask=None):
        # |Q| = |K| = |V| = (bs, n, hs)
        # |mask| = (bs, n)
        # |td| = (bs, n)

        batch_size = Q.size(0)

        mixed_query_layer = self.query(Q)
        mixed_key_layer = self.key(K)
        mixed_value_layer = self.value(V)
        # |mixed_query_layer| = |mixed_key_layer| = |mixed_value_layer| = (bs, n, hs/2(all_attn_h_size))

        mixed_key_conv_attn_layer = self.key_conv_attn_layer(
            K.transpose(1, 2) # |hidden_states.transpose(1, 2)| = (bs, hs, n)
        )
        # |mixed_key_conv_attn_layer| = (bs, hs/2(all_attn_h_size), n)
        mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        # |mixed_key_conv_attn_layer| = (bs, n, hs/2(all_attn_h_size))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        # |query_layer| = (bs, n_attn_head, n, attn_head_size) = (64, 8, 100, 32)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        # |query_layer| = (bs, n_attn_head, n, attn_head_size) = (64, 8, 100, 32)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # |query_layer| = (bs, n_attn_head, n, attn_head_size) = (64, 8, 100, 32)

        ##############
        # conv layer #
        ##############
        # conv를 거친 key와 linear를 거친 query의 element-wise multiply
        conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        # |conv_attn_layer| = (bs, n, hs/2(all_attn_h_size))
        conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        # |conv_kernel_layer| = (bs, n, (n_attn_h * conv_kernel_size) = (64, 100, 8 * 9) = (64, 100, 72)
        conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        # |conv_kernel_layer| = (51200, 9, 1)
        conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        # |conv_kernel_layer| = (51200, 9, 1), 각 head별 확률값들을 도출하는 듯

        # Q X K와 V가 결합되는 부분
        conv_out_layer = self.conv_out_layer(V)
        # |conv_out_layer| = (bs, n, hs/2(all_attn_h_size))
        conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
        # |conv_out_layer| = (bs, n, hs/2(all_attn_h_size))
        conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        # |conv_out_layer| = (bs, hs/2(all_attn_h_size), n, 1)
        # unfold 참고 -> #https://www.facebook.com/groups/PyTorchKR/posts/1685133764959631/
        conv_out_layer = nn.functional.unfold( 
            conv_out_layer,
            kernel_size=[self.conv_kernel_size, 1],
            dilation=1,
            padding=[(self.conv_kernel_size - 1) // 2, 0],
            stride=1,
        )
        # |conv_out_layer| = (64, 2304, 100)
        conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
            batch_size, -1, self.all_head_size, self.conv_kernel_size
        )
        # |conv_out_layer| = (bs, n, hs/2(all_attn_h_size), conv_kernal_size)
        conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        # |conv_out_layer| = (51200, 32, 9)
        # Q X K와 V가 결합되는 부분
        conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        # |conv_out_layer| = (51200, 32, 1)
        conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        # |conv_out_layer| = (6400, 256)

        ##############
        # self_attn layer #
        ##############
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # |attention_scores| = (bs, n_attn_head, n, n) = (64, 8, 100, 100)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # |attention_scores| = (bs, n_attn_head, n, n) = (64, 8, 100, 100)

        #############
        # dist func #
        #############
        total_effect = self.dist_func(attention_scores, td, mask, self.gammas)
        # |total_effect| = (bs, n_attn_head, n, n) = (64, 8, 100, 100)
        attention_scores = attention_scores * total_effect
        # |attention_scores| = (bs, n_attn_head, n, n) = (64, 8, 100, 100)

        # |mask| = (bs, n)
        attention_mask = self.get_extended_attention_mask(mask)
        # |attention_mask| = (bs, n_attn_head, n, n) = (64, 8, 100, 100)
        # 기존 코드에서는 원하는 위치는 0, 마스크 위치에는 -10000.0을 두어서 처리하려 함
        # attention_scores = attention_scores + attention_mask
        # 여기서는 attention_mask를 아래처럼 처리함
        attention_scores.masked_fill_(attention_mask == 0, -1e8)
        # |attention_scores| = (bs, n_attn_head, n, n) = (64, 8, 100, 100)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # |attention_probs| = (bs, n_attn_head, n, n) = (64, 8, 100, 100)
        attention_probs = self.dropout(attention_probs)
        # |attention_probs| = (bs, n_attn_head, n, n) = (64, 8, 100, 100)

        context_layer = torch.matmul(attention_probs, value_layer)
        # |context_layer| = (bs, n_attn_head, n, attn_head_size) = (64, 8, 100, 32)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # |context_layer| = (bs, n, n_attn_head, attn_head_size) = (64, 100, 8, 32)
        
        #####
        # conv와 self_attn이 concat
        #####

        conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
        # |conv_out| = (bs, n, n_attn_head, attn_head_size) = (64, 100, 8, 32)

        context_layer = torch.cat([context_layer, conv_out], 2)
        # |context_layer| = (bs, n, n_attn_head * 2, attn_head_size) = (64, 100, 16, 32)
        
        new_context_layer_shape = context_layer.size()[:-2] + \
             (self.head_ratio * self.all_head_size,)
        # new_context_layer_shape = (bs, n, hs)
        context_layer = context_layer.view(*new_context_layer_shape)
        # |context_layer| = (bs, n, hs)

        outputs = context_layer # 필요하면 함께 출력하기, attention_probs
        # |context_layer| = (bs, n, hs)
        # |attention_probs| = (bs, n_attn_head, n, n) = (64, 8, 100, 100)

        # |outputs| = (bs, n, hs)
        return outputs

    @torch.no_grad()
    def dist_func(self, attention_scores, td, mask, gamma):

        attention_mask = self.get_extended_attention_mask(mask)

        # nomal monotonic
        scores = attention_scores
        bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
        x1 = torch.arange(seqlen).expand(seqlen, -1)
        x2 = x1.transpose(0, 1).contiguous()

        scores_ = scores.masked_fill_(attention_mask == 0, -1e8)
        scores_ = F.softmax(scores_, dim=-1)  # (batch_size, 8, sq, sq)
        scores_ = scores_ * attention_mask.float()
        # [batch_size, 8, seqlen, seqlen]
        distcum_scores = torch.cumsum(scores_, dim=-1)
        # [batch_size, 8, seqlen, 1]
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        device = distcum_scores.get_device()
        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(
            torch.FloatTensor
        )  # [1, 1, seqlen, seqlen]
        position_effect = position_effect.to(device)
        # [batch_size, 8, seqlen, seqlen] positive distance
        # dist_score => d(t, tau)

        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()
        m = nn.Softplus()
        # 1,8,1,1  gamma is \theta in the paper (learnable decay rate parameter)
        gamma = -1.0 * m(gamma).unsqueeze(0)
        # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e-5
        total_effect = torch.clamp(
            torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
        )
        # |total_effect| = (bs, n_attn_head, n, n) = (64, 8, 100, 100)

        # td_effect
        td = F.normalize(td, dim=-1)
        td_scores = self.get_extended_attention_td(td)
        #|scores| = (bs, n_attn_head, n, n)
        bs, head, seqlen = td_scores.size(0), td_scores.size(1), td_scores.size(2)
        td_scores_ = td_scores.masked_fill_(attention_mask == 0, -1e4)
        td_scores_ = F.softmax(td_scores_, dim=-1)
        #|scores_| = (bs, n_attn_head, n, n)

        td_device = td_scores_.get_device()
        
        upper_tri_mat = np.ones(shape=(bs, head, seqlen, seqlen))
        upper_tri_mat = np.triu(upper_tri_mat)
        scores_masking = torch.FloatTensor((upper_tri_mat == 0)).to(td_device)
        upper_tri_mat = torch.FloatTensor(upper_tri_mat).to(td_device)
    
        td_effect = (-1.0 * td_scores_) * scores_masking # + upper_tri_mat
        # 윗 삼각행렬이 0인 상태로 더해줘야 함

        # total_effect + td_effect(이미 -값이므로)
        total_effect = total_effect + td_effect

        return total_effect

    @torch.no_grad()
    def get_extended_attention_mask(self, mask):
        # |mask| = (bs, n)

        mask_shape = mask.size() + (mask.size(1), self.num_attention_heads)
        # mask_shape = (bs, n, n, n_attn_head)
        mask_enc = mask.unsqueeze(-1).expand(mask.size(0), mask.size(1), mask.size(1) * self.num_attention_heads).bool()
        #|mask_enc| = (bs, n, n * n_attn_head)

        mask_enc = mask_enc.view(*mask_shape)
        #|mask_enc| = (bs, n, n, n_attn_head) = (64, 100, 100, 8)

        return mask_enc.permute(0, 3, 2, 1)
        #|mask_enc| = (bs, n_attn_head, n, n)

    @torch.no_grad()
    def get_extended_attention_td(self, td):
        # |td| = (bs, n)
        td = td.unsqueeze(1).repeat(1, td.size(1), 1).unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)

        return td
        #|td| = (bs, n_attn_head, n, n)

    # attention 계산을 위해 마지막 차원을 n_attn_head의 수만큼 나누고, 새로운 차원으로 만들어줌
    def transpose_for_scores(self, x):
        # |x| = (bs, n, hs/2(all_attn_h_size))

        # 마지막 차원을 n_attn_head의 수만큼으로 나눔
        new_x_shape = x.size()[:-1] + \
             (self.num_attention_heads, self.attention_head_size)
        # |x.size()[:-1]| = (bs, n)
        # self.new_num_attention_heads = 8
        # self.attention_head_size = 32
        # |new_x_shape| = (bs, n, new_num_attention_heads, attention_head_size)

        x = x.view(*new_x_shape)
        # |x| = (bs, n, n_attn_head, attn_head_size) = (64, 100, 8, 32)

        return x.permute(0, 2, 1, 3)
        # |x| = (bs, n_attn_head, n, attn_head_size) = (64, 8, 100, 32)


class EncoderBlock(nn.Module):

    def __init__(
        self,
        hidden_size, #512
        n_splits,
        use_leakyrelu,
        max_seq_len,
        dropout_p=.1,
    ):
        super().__init__()

        self.use_leakyrelu = use_leakyrelu

        self.attn = ForgettingMonotonicConvBertSelfAttention(hidden_size, n_splits, dropout_p)
        self.attn_norm = nn.LayerNorm(hidden_size) #attention을 위한 layerNorm
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if self.use_leakyrelu else self.gelu(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, td, mask):
        # |x| = (bs, n, emb_size), torch.float32
        # |td| = (bs, n)
        # |mask| = (bs, n)

        # Pre-LN:
        z = self.attn_norm(x)
        # |z| = (bs, n, emb_size)

        # x+ means redisual connection
        z = x + self.attn_dropout(self.attn(Q=z,
                                            K=z,
                                            V=z,
                                            td=td,
                                            mask=mask))
        # |z| = (bs, n, hs)

        z = z + self.fc_dropout(self.fc(self.fc_norm(z)))
        # |z| = (bs, n, hs)

        return z, mask

    # upstage's gelu
    def gelu(x):
        """Upstage said:
            Implementation of the gelu activation function.
            For information: OpenAI GPT's gelu is slightly different
            (and gives slightly different results):
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
            (x + 0.044715 * torch.pow(x, 3))))
            Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ForgettingMonoConvBert4ktPlus(nn.Module):

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

        self.encoder = nn.ModuleList(
            [EncoderBlock(
                hidden_size,
                num_head,
                self.use_leakyrelu,
                self.max_seq_len,
                dropout_p,
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

    def forward(self, q, r, pid, td, mask): #td는 time_data
        # |q| = (bs, n)
        # |r| = (bs, n)
        # |td| = (bs, n)
        # |mask| = (bs, n)

        # 다음 값과의 시간 차이를 계산
        td_f = torch.cat([td[:, 1:], td[:, -1].unsqueeze(-1)], dim=-1)
        td_p = td
        # 최소값이 0이 되도록 조정
        td = torch.clamp(td_f - td_p, min=0)

        #차이에 대한 softmax 값

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


        # z, _ = self.encoder(z, td_n, mask)
        # # |z| = (bs, n, hs)

        for block in self.encoder:
            z, _ = block(z, td, mask)

        y_hat = self.generator(z)
        #|y_hat| = (bs, n, output_size=1)

        return y_hat