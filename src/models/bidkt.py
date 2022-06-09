import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None, dk=64):
        # |Q| = (batch_size, m, hidden_size)
        # |K| = |V| = (batch_size, n, hidden_size)
        # |mask| = (batch_size, m, n)

        # w = attention energy
        w = torch.bmm(Q, K.transpose(1, 2))
        # |w| = (batch_size, m, n)
        if mask is not None:
            assert w.size() == mask.size()
            w.masked_fill_(mask, -float('inf')) #마스크 씌우기

        w = self.softmax(w / (dk**.5)) #attention값
        c = torch.bmm(w, V) #attention값과 Value값 행렬곱
        # |c| = (batch_size, m, hidden_size)

        return c


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

        self.attn = Attention()

    def forward(self, Q, K, V, mask=None):
        # 각 multihead의 위치에 따라 입력값들의 차원이 달라지게 됨
        # 따라서 각 차원이 들어오더라도 대응할 수 있도록 코드를 짜야함
        # |Q|    = (batch_size, m, hidden_size) m은 decoder의 time step 갯수
        # |K|    = (batch_size, n, hidden_size) n은 encoder의 time step 갯수
        # |V|    = |K|
        # |mask| = (batch_size, m, n)

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
        hidden_size,
        n_splits,
        dropout_p=.1,
    ):
        super().__init__()

        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size) #attention을 위한 layerNorm
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(), #reaky relu
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        # |x|    = (bs, n, hs)
        # |mask| = (bs, n, n)

        # Pre-LN:
        z = self.attn_norm(x)
        # x+ means redisual connection
        z = x + self.attn_dropout(self.attn(Q=z,
                                            K=z,
                                            V=z,
                                            mask=mask))
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


class Bidkt(nn.Module):

    def __init__(
        self,
        num_q,
        num_r,
        hidden_size,
        output_size,
        num_head,
        num_encoder,
        max_seq_len,
        dropout_p=.1,
    ):
        self.num_q = num_q + 2 # <PAD>와 <MASK>를 추가한만큼의 Emb값이 필요
        self.num_r = num_r + 2 # <PAD>와 <MASK>를 추가한만큼의 Emb값이 필요
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_head = num_head
        self.num_encoder = num_encoder
        self.max_seq_len = max_seq_len
        self.dropout_p = dropout_p

        super().__init__()

        # embedding
        # question embedding
        self.emb_q = nn.Embedding(num_q, hidden_size)
        # response embedding
        self.emb_r = nn.Embedding(num_r, hidden_size)
        # positional embedding
        self.emb_p = nn.Embedding(max_seq_len, hidden_size)
        self.emb_dropout = nn.Dropout(dropout_p)

        # MySequential을 활용해 필요한만큼 encoder block을 만듦
        self.encoder = MySequential(
            *[EncoderBlock(
                hidden_size,
                num_head,
                dropout_p,
              ) for _ in range(num_encoder)],
        )

        self.generator = nn.Sequential(
            nn.LayerNorm(hidden_size), # Only for Pre-LN Transformer.
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=-1),
        )

    # positional embedding
    def _positional_embedding(self, q, r):
        # |q| = (bs, n)
        # |r| = (bs, n)

        seq_len = q.size(1)
        # seq_len = (n,)

        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand_as(q)
        # |pos| = (bs, n)

        emb = self.emb_q(q) + self.emb_r(r) + self.emb_p(pos)
        # |emb| = (bs, n, hs)

        return emb


    def forward(self, q, r, mask):
        # |q| = (bs, n)
        # |r| = (bs, n)
        # |mask| = (bs, n)

        # Mask to prevent having attention weight on padding position.
        with torch.no_grad():
            mask_enc = mask.unsqueeze(1).expand(q.size(), mask.size(-1))
            # |mask_enc| = (bs, n, n)
        
        z = self.emb_dropout(self._positional_embedding(q, r))
        # |z| = (bs, n, hs)

        z, _ = self.encoder(z, mask_enc)
        # |z| = (bs, n, hs)

        y_hat = self.generator(z)
        # |y_hat| = (bs, n, output_size)

        return y_hat