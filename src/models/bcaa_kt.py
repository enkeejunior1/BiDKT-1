import torch
import torch.nn as nn

#using monotonic attention

class MonotonicAttention(nn.Module):

    def __init__(self, device, dropout_p):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.gamma = nn.Parameter(torch.zeros(1, 1)) #사이즈 이후 조절
        torch.nn.init.xavier_uniform_(self.gamma)

    def forward(self, Q, K, V, mask=None, dk=64):
        # |Q| = (batch_size * n_splits, m, hidden_size / n_splits)
        # |K| = |V| = (batch_size * n_splits, m, hidden_size / n_splits)
        # |mask| = (batch_size * n_splits, m, hidden_size / n_splits)

        # w = attention energy
        w = torch.bmm(Q, K.transpose(1, 2))
        # |w| = (batch_size, n, n)
        if mask is not None:
            assert w.size() == mask.size()
            # mask를 -float('inf')로 만들어두니 overflow 문제 발생
            w.masked_fill_(mask, -1e8)

        # distance score, no_grad()
        d = self.distance_func(w, dk)

        theta = -1 * self.softplus(self.gamma)

        total_effect = torch.clamp(
            torch.clamp((d * theta).exp(), min=1e-5), 
            max = 1e5)

        w_n = w / (dk**.5)

        s = self.dropout(self.softmax(w_n * total_effect))
        # attention value
        c = torch.bmm(s, V)
        # |c| = (batch_size, m, hidden_size)

        return c

    # 거리함수는 grad를 받지 않음
    @torch.no_grad()
    def distance_func(self, w, dk):
        # |w| = (batch_size, n, n), torch.bmm(Q, K.transpose(1, 2))

        w = w / (dk**.5)
        # |w| = (batch_size, n, n)
        score = self.softmax(w)
        #score = score * mask.float().to(device) -> 원문인데, 필요있을까?

        x1 = torch.arange(w.size(2)).expand(w.size(2), -1).to(self.device)
        x2 = x1.transpose(0, 1).contiguous()

        distcum_score = torch.cumsum(score, dim=-1)
        disttotal_score = torch.sum( score, dim=-1, keepdim=True)
        position_effect = torch.abs(x1-x2)[None, :, :].type(torch.FloatTensor).to(self.device)

        dist_score = torch.clamp(
            (disttotal_score - distcum_score) * position_effect, min=0.
        )
        dist_score = dist_score.sqrt().detach()

        return dist_score


class MultiHead(nn.Module):

    def __init__(self, hidden_size, n_splits, device, dropout_p):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_splits = n_splits

        # Note that we don't have to declare each linear layer, separately.
        self.Q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn = MonotonicAttention(device, dropout_p)

    def forward(self, Q, K, V, mask=None):
        # |Q| = |K| = |V| = (bs, n, hs)
        # |mask| = (bs, n, bs)

        # 마지막 차원을 split함, 그러면 QWs는 리스트형태로 쌓이게 됨
        QWs = self.Q_linear(Q).split(self.hidden_size // self.n_splits, dim=-1)
        KWs = self.K_linear(K).split(self.hidden_size // self.n_splits, dim=-1)
        VWs = self.V_linear(V).split(self.hidden_size // self.n_splits, dim=-1)
        # |QW_i| = (batch_size, m, hidden_size / n_splits) -> 이게 리스트형태로 쌓이는 것임
        # |KW_i| = |VW_i| = (batch_size, n, hidden_size / n_splits)

        # By concatenating splited linear transformed results,
        # we can remove sequential operations,
        # like mini-batch parallel operations.
        # 위에서 split한 것을 0번째 차원을 기준으로 concat함
        QWs = torch.cat(QWs, dim=0)
        KWs = torch.cat(KWs, dim=0)
        VWs = torch.cat(VWs, dim=0)
        # |QWs| = (batch_size * n_splits, m, hidden_size / n_splits)
        # |KWs| = |VWs| = (batch_size * n_splits, n, hidden_size / n_splits)

        # 마스크도 위의 cat처럼 같은 차원으로 만들기
        if mask is not None:
            mask = torch.cat([mask for _ in range(self.n_splits)], dim=0)
            # |mask| = (batch_size * n_splits, m, n)

        c = self.attn(
            QWs, KWs, VWs,
            mask=mask,
            dk=self.hidden_size // self.n_splits,
        )
        # |c| = (batch_size * n_splits, m, hidden_size / n_splits)

        # 다시 분리해서 원래 차원으로 되돌림
        # We need to restore temporal mini-batchfied multi-head attention results.
        c = c.split(Q.size(0), dim=0)
        # |c_i| = (batch_size, m, hidden_size / n_splits)
        c = self.linear(torch.cat(c, dim=-1))
        # |c| = (batch_size, m, hidden_size)

        return c


class EncoderBlock(nn.Module):

    def __init__(
        self,
        hidden_size, #512
        n_splits,
        use_leakyrelu,
        device,
        dropout_p=.1,
    ):
        super().__init__()

        self.use_leakyrelu = use_leakyrelu

        self.attn = MultiHead(hidden_size, n_splits, device, dropout_p)
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


class BCAA_KT(nn.Module):

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

        # MySequential을 활용해 필요한만큼 encoder block을 만듦
        self.encoder = MySequential(
            *[EncoderBlock(
                hidden_size,
                num_head,
                self.use_leakyrelu,
                device,
                dropout_p,
              ) for _ in range(num_encoder)],
        )

        self.generator = nn.Sequential(
            nn.LayerNorm(hidden_size), # Only for Pre-LN Transformer.
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid() # binary
        )

    # positional embedding
    def _positional_embedding(self, q, r, pid):
        # |q| = (bs, n)
        # |r| = (bs, n)
        seq_len = q.size(1)
        # seq_len = (n,)
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand_as(q).to(self.device)
        # |pos| = (bs, n)
        
        emb = self.emb_q(q) + self.emb_r(r) + self.emb_p(pos) + self.emb_pid(pid)
        # |emb| = (bs, n, hs)

        return emb

    def forward(self, q, r, pid, mask):
        # |q| = (bs, n)
        # |r| = (bs, n)
        # |mask| = (bs, n)

        # Mask to prevent having attention weight on padding position.
        with torch.no_grad():
            mask_enc = mask.unsqueeze(-1).expand(mask.size(0), mask.size(1), mask.size(1)).bool()
             # |mask_enc| = (bs, n, n)

        emb = self._positional_embedding(q, r, pid)
        # |emb| = (bs, n, emb_size)

        z = self.emb_dropout(emb)
        # |z| = (bs, n, emb_size)

        # |mask_enc| = (bs, n, n)
        # |z| = (bs, n, emb_size)
        z, _ = self.encoder(z, mask_enc)
        # |z| = (bs, n, hs)

        y_hat = self.generator(z)
        #|y_hat| = (bs, n, output_size=1)

        return y_hat