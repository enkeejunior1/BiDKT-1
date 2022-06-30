import torch
import torch.nn as nn

"""
bert based context-aware attentive knowledge tracing model
"""

# 수정하기
class MonotonicAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.gamma = nn.Parameter(torch.zeros(1, 1)) #사이즈 이후 조절
        torch.nn.init.xavier_uniform_(self.gamma)

    def forward(self, Q, K, V, mask=None, dk=64):
        # |Q| = (batch_size, m, hidden_size)
        # |K| = |V| = (batch_size, n, hidden_size)
        # |mask| = (batch_size, m, n)

        # w = attention energy
        w = torch.bmm(Q, K.transpose(1, 2))

        # |w| = (batch_size, m, n)
        if mask is not None:
            assert w.size() == mask.size()
            # mask를 -float('inf')로 만들어두니 overflow 문제 발생
            w.masked_fill_(mask, -1e8)

        # distance score
        d = self.distance_func()
        # attention energy
        s = d * w / (dk**.5)
        # attention value
        a = self.softmax(s)
        c = torch.bmm(a, V)
        # |c| = (batch_size, m, hidden_size)

        return c

    # 거리함수는 grad를 받지 않음
    @torch.no_grad()
    def distance_func(self):
        theta = self.gamma
        theta = -1 * self.softplus(theta)

        dist_scores = 0
        pass


class MultiHead(nn.Module):

    def __init__(self, hidden_size, n_splits):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_splits = n_splits

        # Note that we don't have to declare each linear layer, separately.
        self.Q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn = MonotonicAttention()

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
        dropout_p=.1,
    ):
        super().__init__()

        self.use_leakyrelu = use_leakyrelu

        self.attn = MultiHead(hidden_size, n_splits)
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

class KnowledgRetrieverEncoderBlock(nn.Module):

    def __init__(
        self,
        hidden_size, #512
        n_splits,
        use_leakyrelu,
        dropout_p=.1,
    ):
        super().__init__()

        self.use_leakyrelu = use_leakyrelu

        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size) #attention을 위한 layerNorm
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if self.use_leakyrelu else nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x_hat, y_hat, mask):
        # |x| = (bs, n, emb_size), torch.float32
        # |mask| = (bs, n, n)

        # Pre-LN:
        x = self.attn_norm(x_hat)
        # |z| = (bs, n, emb_size)

        # Pre-LN:
        y = self.attn_norm(y_hat)
        # |z| = (bs, n, emb_size)   

        # x+ means redisual connection
        """
        residual을 x와 y 둘 다 넣어야 하나?
        """
        z = x + self.attn_dropout(self.attn(Q=x,
                                            K=x,
                                            V=y, 
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

        # question + response embedding
        self.emb_q = nn.Embedding(self.num_q, self.hidden_size).to(self.device)
        self.emb_r = nn.Embedding(self.num_r, self.hidden_size).to(self.device)

        self.diff_emb = nn.Embedding(self.num_pid, 1)

        # positional embedding
        self.emb_p = nn.Embedding(self.max_seq_len, self.hidden_size).to(self.device)
        self.emb_dropout = nn.Dropout(self.dropout_p)

        # MySequential을 활용해 필요한만큼 encoder block을 만듦
        self.q_encoder = MySequential(
            *[EncoderBlock(
                hidden_size,
                num_head,
                self.use_leakyrelu,
                dropout_p,
              ) for _ in range(num_encoder)],
        )

        self.k_encoder = MySequential(
            *[EncoderBlock(
                hidden_size,
                num_head,
                self.use_leakyrelu,
                dropout_p,
              ) for _ in range(num_encoder)],
        )

        self.kr_encoder = MySequential(
            *[KnowledgRetrieverEncoderBlock(
                hidden_size,
                num_head,
                self.use_leakyrelu,
                dropout_p,
              ) for _ in range(num_encoder)],
        )

        self.generator = nn.Sequential(
            nn.LayerNorm(hidden_size), # Only for Pre-LN Transformer.
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid() # binary
        )

    # rasch embedding
    def _rasch_embedding(self, q, r, pid):
        # |q| = (bs, n)
        # |r| = (bs, n)

        seq_len = q.size(1)
        # seq_len = (n,)
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand_as(q).to(self.device)
        # |pos| = (bs, n)

        # positional embedding
        pos_emb = self.emb_p(pos)

        #문항 난이도 정보
        diff_emb = self.diff_emb(pid)

        # 기존 akt의 모델처럼 qr 정보를 활용하기에는 mask와 pad가 있어서 문제가 발생함
        # 따라서 emb_qr은 emb_q와 emb_r의 element-wise로 하고, qr_diff만 넣어주는 형태로 사용함
        # emb_qr과 emb_qr_diff는 값은 같지만, 의미가 다르므로 다르게 표기

        # q emb
        emb_q = self.emb_q(q)
        emb_q_diff = self.emb_q(q)
        rasch_emb_q = emb_q + diff_emb * emb_q_diff

        rasch_emb_x = rasch_emb_q + pos_emb
        # |rasch_emb_x| = (bs, n, hs)

        # qr emb
        emb_qr = self.emb_q(q) + self.emb_r(r)
        emb_qr_diff = self.emb_q(q) + self.emb_r(r)
        rasch_emb_qr = emb_qr + diff_emb * emb_qr_diff

        rasch_emb_y = rasch_emb_qr + pos_emb
        # |rasch_emb_y| = (bs, n, hs)

        return rasch_emb_x, rasch_emb_y

    def forward(self, q, r, pid, mask):
        # |q| = (bs, n)
        # |r| = (bs, n)
        # |mask| = (bs, n)

        # Mask to prevent having attention weight on padding position.
        with torch.no_grad():
            mask_enc = mask.unsqueeze(-1).expand(mask.size(0), mask.size(1), mask.size(1)).bool()
             # |mask_enc| = (bs, n, n)

        rasch_emb_x, rasch_emb_y = self._rasch_embedding(q, r, pid)
        # |rasch_emb_x| = |rasch_emb_y| = (bs, n, emb_size)
        rasch_emb_x = self.emb_dropout(rasch_emb_x)
        # |rasch_emb_x| = (bs, n, emb_size)
        rasch_emb_y = self.emb_dropout(rasch_emb_y)
        # |rasch_emb_y| = (bs, n, emb_size)

        # 여기 아래에서 encoder를 두 개 사용
        # |mask_enc| = (bs, n, n)
        # |z| = (bs, n, emb_size)

        # question encoder
        x_hat, _ = self.q_encoder(rasch_emb_x, mask_enc)
        # |x| = (bs, n, hs)

        # knowledge encoder
        y_hat, _ = self.k_encoder(rasch_emb_y, mask_enc)
        # |y| = (bs, n, hs)

        # knowledge retreiver encoder
        h, _ = self.kr_encoder(x_hat, y_hat, mask_enc)
        # |h| = (bs, n, hs)

        # prediction network
        r_hat = self.generator(h, rasch_emb_x)
        #|y_hat| = (bs, n, output_size=1)

        return r_hat