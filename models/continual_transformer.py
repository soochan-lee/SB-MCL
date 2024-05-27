import math
from functools import lru_cache

import torch
import torch.nn as nn
from einops import rearrange, repeat, pack, unpack, reduce
from fast_transformers.feature_maps import Favor, ActivationFunctionFeatureMap
from torch import nn as nn, Tensor

from models import Model
from models.encoders.classification import ClassEncoder
from models.model import Output
from utils import OUTPUT_TYPE_TO_LOSS_FN, cross_entropy
from models.encoders import MlpEncoder, X_ENCODER


class ContinualAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        if config['tf_attn'] == 'vanilla':
            self.feature_map = None
        elif config['tf_attn'] == 'elu':
            # Linear transformer: elu(x) + 1 feature map with much better precision
            self.feature_map = ActivationFunctionFeatureMap.factory(
                lambda x: torch.where(x > 0, x + 1, torch.exp(torch.minimum(x, torch.ones([], device=x.device))))
            )(None)
        elif config['tf_attn'] == 'favor':
            # Performer
            self.feature_map = Favor(
                query_dimensions=config['qk_dim'], n_dims=config['favor_dim'],
                stabilize=config['favor_stabilize'], redraw=1)  # always redraw
            # Manually control redraws to avoid resampling random features when loading from checkpoint
            self.register_buffer('feature_map_calls', torch.zeros([], dtype=torch.long))
        else:
            raise ValueError(f'Unknown feature map {config["feature_map"]}')

    def forward(self, queries, keys, values, attach_test_after, train_len=0, past_state=None, return_state=False):
        """Compute the attention.

        Args:
            queries: [batch, q_len, heads, qk_dim]
            keys: [batch, kv_len, heads, qk_dim]
            values: [batch, kv_len, heads, v_dim]
            attach_test_after: where to attach the test examples in the train sequence [batch, test_num]
            train_len: the length of the train sequence
            past_state: [batch_size, 1, heads, f_dim, v_dim+1]
            return_state: whether to return the last state after train sequence
        """
        batch, q_len, heads, qk_dim = queries.shape
        batch, test_num = attach_test_after.shape
        test_chunk_size = (q_len - train_len) // test_num
        test_len = test_num * test_chunk_size
        assert test_len + train_len == q_len

        if self.feature_map is not None:
            if 'favor_redraw' in self.config:
                # Performer
                if self.feature_map_calls.item() % self.config['favor_redraw'] == 0:
                    self.feature_map.new_feature_map(queries.device)
                self.feature_map_calls += 1
            queries = self.feature_map.forward_queries(queries)
            keys = self.feature_map.forward_keys(keys)

        q = rearrange(queries, 'b l h d -> b h l d')
        k = rearrange(keys, 'b l h d -> b h l d')
        v = rearrange(values, 'b l h d -> b h l d')

        if past_state is not None:
            past_k, past_v = past_state
            k, _ = pack([past_k, k], 'b h * d')
            v, _ = pack([past_v, v], 'b h * d')

            past_len = past_k.shape[-2]
            attach_test_after = attach_test_after + past_len
            train_len += past_len

        train_len = k.shape[-2] - test_num * test_chunk_size
        aux_output = {
            'state': (k[:, :, :train_len], v[:, :, :train_len]) if return_state else None,
        }

        attn_logit = torch.einsum('bhmd,bhnd->bhmn', q, k)
        if self.feature_map is None:
            attn_logit = attn_logit / (qk_dim ** 0.5)

        # Build attention mask
        mask = get_continual_mask(*attn_logit.shape[-2:], test_num, test_chunk_size, device=q.device)
        mask = repeat(mask, 'q k -> b h q k', b=batch, h=1).contiguous()

        # Prevent some of the attention from test queries to train keys according to attach_test_after
        indices = rearrange(torch.arange(train_len, device=q.device), 'train_len -> () () () train_len')
        attach_test_after = rearrange(attach_test_after, 'b n -> b () n ()')
        test_q_train_k_mask = repeat(
            indices <= attach_test_after,
            'b h test_num train_len -> b h (test_num c) train_len', c=test_chunk_size
        ).float()
        mask[:, :, q_len - test_len:, :train_len] = test_q_train_k_mask

        if self.feature_map is None:
            # Vanilla attention
            mask = torch.zeros_like(mask).masked_fill(~mask.bool(), torch.finfo(attn_logit.dtype).min)
            attn_logit = attn_logit + mask
            attn = torch.softmax(attn_logit, dim=-1)
            aux_output['attn_logit'] = attn_logit
        else:
            # Efficient transformers
            attn_logit = attn_logit + 1e-9  # for stability
            attn_logit = attn_logit * mask
            attn = attn_logit / attn_logit.sum(dim=-1, keepdim=True)
            aux_output['attn'] = attn

        output = torch.einsum('bhmn,bhnd->bhmd', attn, v)
        output = rearrange(output, 'b h l d -> b l (h d)')
        return output, aux_output


@lru_cache(maxsize=8)
def get_continual_mask(query_num: int, key_num: int, test_num: int, test_chunk_size: int, device='cuda'):
    """Build mask that simulates each test example being added to the end of train sequence one by one.

    Example:
        query_num = 8
        key_num = 12
        test_num = 3
        test_chunk_size = 2
        mask = tensor([[1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
                       [1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0.],
                       [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0.],
                       [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0.],
                       [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1.]])
    """
    assert query_num <= key_num

    # Basic causal mask
    mask = torch.tril(torch.ones((query_num, key_num), device=device), diagonal=key_num - query_num)

    # Block-diagonal mask for test chunks
    test_chunks = [torch.tril(torch.ones([test_chunk_size, test_chunk_size], device=device), diagonal=0)] * test_num
    test_allowed = torch.block_diag(*test_chunks)

    # Paste the block-diagonal mask to the causal mask
    q_train = query_num - test_num * test_chunk_size
    k_train = key_num - test_num * test_chunk_size
    mask[q_train:, k_train:] = test_allowed
    return mask


class ContinualAttentionLayer(nn.Module):
    """Attention layer optimized for continual learning."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dropout = nn.Dropout(config['tf_dropout'])

        self.qkv_linear = nn.Linear(config['hidden_dim'], config['tf_heads'] * (2 * config['qk_dim'] + config['v_dim']))
        self.attn = ContinualAttention(config, layer_idx)
        self.proj_linear = nn.Linear(config['tf_heads'] * config['v_dim'], config['hidden_dim'])

    def forward(self, x, attach_test_after, train_len=0, past_state=None, return_state=False):
        # Linear projection for Q, K, V
        qkv = self.qkv_linear(x)
        q, k, v = unpack(qkv, [
            [self.config['tf_heads'] * self.config['qk_dim']],
            [self.config['tf_heads'] * self.config['qk_dim']],
            [self.config['tf_heads'] * self.config['v_dim']]
        ], 'b l *')
        q = rearrange(q, 'b l (h d) -> b l h d', h=self.config['tf_heads'])
        k = rearrange(k, 'b l (h d) -> b l h d', h=self.config['tf_heads'])
        v = rearrange(v, 'b l (h d) -> b l h d', h=self.config['tf_heads'])

        # Compute attention
        attn_output, state = self.attn(
            q, k, v,
            attach_test_after=attach_test_after, train_len=train_len,
            past_state=past_state, return_state=return_state)

        x = x + self.dropout(self.proj_linear(attn_output))
        return x, state


class ContinualTransformerLayer(nn.Module):
    """Transformer layer optimized for continual learning."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn_layer_norm = nn.LayerNorm(config['hidden_dim']) if config['tf_ln'] else nn.Identity()
        self.attn_layer = ContinualAttentionLayer(config, layer_idx)
        self.mlp_layer_norm = nn.LayerNorm(config['hidden_dim']) if config['tf_ln'] else nn.Identity()
        self.mlp_layer = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['tf_ff_dim']),
            nn.GELU(),
            nn.Linear(config['tf_ff_dim'], config['hidden_dim']),
            nn.Dropout(config['tf_dropout']),
        )

    def forward(self, xy_enc, attach_test_after, train_len=0, past_state=None, return_state=False):
        x, state = self.attn_layer(
            self.attn_layer_norm(xy_enc), attach_test_after=attach_test_after, train_len=train_len,
            past_state=past_state, return_state=return_state)
        x = x + self.mlp_layer(self.mlp_layer_norm(x))
        return x, state


def sample_test_attachment(train_y, test_y):
    """Sample where to attach each test example in the train sequence

    Each test example can be evaluated after the task it belongs to or after any task after that.
    E.g., a test example of task3 can be tested after training task3, task4, ..., or taskN

    This function requires minimal assumptions about the task structure.
    Each sequence in the batch can have a different number of tasks and a different number of examples per task.
    The only assumption is that each sequence in train_y should start from 0 and monotonically increase by 0 or 1.
    E.g., [0, 0, 1, 1, 1, 2] is valid, but [0, 0, 1, 2, 1, 2] is not.

    Args:
        train_y: [batch, train_num]
        test_y: [batch, test_num]

    Returns:
        test_attachment: train example index [batch, test_num]
    """
    # For each test example, randomly sample which task to test after (sample in range [test_y, tasks))
    tasks = train_y.max(dim=1, keepdim=True).values + 1  # [batch, 1]
    num_options = tasks - test_y  # [batch, test_num]
    test_after = tasks - 1 - (torch.rand(test_y.shape, device=test_y.device) * num_options).long()

    # Find the indices of the ends of tasks in train sequence
    # E.g., if train_y = [[0, 0, 1, 2, 2, 2], [0, 1, 1, 2, 2, 3]], then task_end_indices = [1, 2, 5, 0, 2, 4, 5]
    # The first three comes from the first row, and the last four comes from the second row
    is_task_end = nn.functional.pad(train_y[:, 1:] != train_y[:, :-1], [0, 1], value=True)  # [batch, train_num]
    idx_arange = repeat(torch.arange(train_y.shape[1], device=train_y.device), 'l -> b l', b=train_y.shape[0])
    task_end_indices = torch.masked_select(idx_arange, is_task_end)  # [total_number_of_tasks]

    # Convert batch-wise task indices in test_after to global task indices that can be used to index task_end_indices
    num_boundaries = is_task_end.sum(dim=1)  # [batch]
    global_task_idx_offset = torch.cumsum(num_boundaries, dim=0, dtype=torch.long)
    global_task_idx_offset = nn.functional.pad(global_task_idx_offset[:-1], [1, 0], value=0)
    global_task_idx_offset = rearrange(global_task_idx_offset, 'b -> b 1')
    test_after_global = test_after + global_task_idx_offset  # [batch, test_num]

    return task_end_indices[test_after_global]


class ContinualTransformer(Model):
    """Decoder-only transformer optimized for continual learning."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_type = config['input_type']
        self.output_type = config['output_type']

        # max_len = config['tasks'] * (
        #         config['train_shots'] * (1 + config['y_len']) +
        #         config['test_shots'] * config['y_len']
        # )
        max_len = 20000
        self.pos_enc = PositionalEncoding(config['hidden_dim'], max_len=max_len)
        if self.input_type == 'image':
            self.x_encoder = X_ENCODER[config['x_encoder']](config)
            self.x_proj = nn.Linear(256 * (config['x_shape'][1] // 16) * (config['x_shape'][2] // 16), config['hidden_dim'])
        elif self.input_type == 'vector':
            self.x_encoder = MlpEncoder(config, input_dim=math.prod(config['x_shape']))
        else:
            raise NotImplementedError

        if self.output_type == 'class':
            self.y_encoder = ClassEncoder(config)
            self.output = nn.Linear(config['hidden_dim'], config['y_vocab'])
        elif self.output_type in ['vector', 'angle']:
            self.y_encoder = MlpEncoder(config, input_dim=math.prod(config['y_shape']))
            self.output = nn.Linear(config['hidden_dim'], math.prod(config['y_shape']))
        elif self.output_type == 'image':
            self.y_encoder = X_ENCODER[config['x_encoder']](config)
            self.output = nn.Linear(config['hidden_dim'], math.prod(config['y_shape']))
        else:
            raise NotImplementedError
        self.loss_fn = OUTPUT_TYPE_TO_LOSS_FN[config['output_type']]

        self.tf_layers = nn.ModuleList([
            ContinualTransformerLayer(config, layer_idx) for layer_idx in range(config['tf_layers'])
        ])

    def forward(self, train_x, train_y, test_x, test_y, summarize, meta_split):
        if self.input_type == 'image':
            batch, train_num, c, h, w = train_x.shape
            batch, test_num, c, h, w = test_x.shape

            # Encode images
            x, _ = pack([
                rearrange(train_x, 'b l c h w -> (b l) c h w'),
                rearrange(test_x, 'b l c h w -> (b l) c h w'),
            ], '* c h w')
            x_enc = self.x_encoder(x)
            x_enc = rearrange(x_enc, 'bl c h w -> bl (c h w)')
            x_enc = self.x_proj(x_enc)
            train_x_enc = rearrange(x_enc[:batch * train_num], '(b l) h -> b l h', b=batch, l=train_num)
            test_x_enc = rearrange(x_enc[batch * train_num:], '(b l) h -> b l h', b=batch, l=test_num)
        elif self.input_type == 'vector':
            batch, train_num, x_dim = train_x.shape
            batch, test_num, x_dim = test_x.shape

            # Encode x vectors
            x, x_ps = pack([
                rearrange(train_x, 'b l d -> (b l) d'),
                rearrange(test_x, 'b l d -> (b l) d'),
            ], '* d')
            x_enc = self.x_encoder(x)
            train_x_enc, test_x_enc = unpack(x_enc, x_ps, '* h')
            train_x_enc = rearrange(train_x_enc, '(b l) h -> b l h', b=batch, l=train_num)
            test_x_enc = rearrange(test_x_enc, '(b l) h -> b l h', b=batch, l=test_num)
        else:
            raise NotImplementedError

        if self.output_type == 'class':
            batch, train_num = train_y.shape
            batch, test_num = test_y.shape

            # Encode labels
            y_codebook = self.y_encoder.sample_codebook(batch, device=train_y.device)
            batch, num_classes, y_len = y_codebook.shape
            train_y_code = self.y_encoder.y2code(train_y, y_codebook)  # [batch, train_num, y_len]
            test_y_code = self.y_encoder.y2code(test_y, y_codebook)  # [batch, test_num, y_len]
            train_y_enc = self.y_encoder.encode(train_y_code)  # [batch, train_num, y_len, hidden]
            test_y_enc = self.y_encoder.encode(test_y_code)  # [batch, test_num, y_len, hidden]
        elif self.output_type in ['vector', 'angle']:
            batch, train_num, y_dim = train_y.shape
            batch, test_num, y_dim = test_y.shape

            # Encode y vectors
            y, y_ps = pack([
                rearrange(train_y, 'b l d -> (b l) d'),
                rearrange(test_y, 'b l d -> (b l) d'),
            ], '* d')
            y_enc = self.y_encoder(y)
            train_y_enc, test_y_enc = unpack(y_enc, y_ps, '* d')
            y_len = 1
            train_y_enc = rearrange(train_y_enc, '(b l) h -> b l 1 h', b=batch, l=train_num)
            test_y_enc = rearrange(test_y_enc, '(b l) h -> b l 1 h', b=batch, l=test_num)
        elif self.output_type == 'image':
            batch, train_num, c, h, w = train_y.shape
            batch, test_num, c, h, w = test_y.shape

            # Encode y images
            y, y_ps = pack([
                rearrange(train_y, 'b l c h w -> (b l) c h w'),
                rearrange(test_y, 'b l c h w -> (b l) c h w'),
            ], '* c h w')
            y_enc = self.y_encoder(y)
            train_y_enc, test_y_enc = unpack(y_enc, y_ps, '* c h w')
            y_len = 1
            train_y_enc = rearrange(train_y_enc, '(b l) c h w -> b l 1 (c h w)', b=batch, l=train_num)
            test_y_enc = rearrange(test_y_enc, '(b l) c h w -> b l 1 (c h w)', b=batch, l=test_num)
        else:
            raise NotImplementedError

        # Interleave train_x_enc and train_y_enc to build train sequence
        train_xy_enc, _ = pack([train_x_enc, train_y_enc], 'b l * h')
        train_xy_enc = rearrange(train_xy_enc, 'b l chunk h -> b (l chunk) h', chunk=1 + y_len)

        # Add positional encoding to train sequence
        train_xy_enc = self.pos_enc(train_xy_enc)

        loss_weight = None
        if not summarize and self.config['distributed_loss']:
            # Sample where to attach each test example
            task_idx = torch.arange(self.config['tasks'], device=train_y.device)
            train_task = repeat(task_idx, 't -> b (t s)', b=batch, s=self.config['train_shots'])
            test_task = repeat(task_idx, 't -> b (t s)', b=batch, s=self.config['test_shots'])
            test_attachment = sample_test_attachment(train_task, test_task)  # [batch, test_num]
            if 'distributed_loss_weighted' in self.config and self.config['distributed_loss_weighted']:
                loss_weight_mean = (self.config['tasks'] + 1) / 2
                loss_weight = (self.config['tasks'] - test_task) / loss_weight_mean  # [batch, test_num]
        else:
            # Attach all test examples after the last train example
            test_attachment = repeat(
                torch.tensor(train_num - 1, dtype=torch.long, device=train_y.device),
                ' -> b n', b=batch, n=test_num)

        # Since test_attachment is the indices of train examples, convert it to token indices
        attach_test_after = test_attachment * (1 + y_len) + y_len

        test_xy_enc = self.build_test_xy(test_x_enc, test_y_enc, attach_test_after)

        xy_enc, _ = pack([train_xy_enc, test_xy_enc], 'b * h')
        train_len = train_num * (1 + y_len)
        hidden, aux_outputs = self.forward_tf(xy_enc, attach_test_after, train_len=train_len)

        test_hidden = hidden[:, train_len:]
        logit = self.output(test_hidden)
        if self.output_type == 'class':
            logit = rearrange(logit, 'b (n y) v -> b n y v', n=test_num, y=y_len)
            loss = self.loss_fn(logit, test_y_code)
        elif self.output_type == 'vector':
            loss = reduce(((logit - test_y) ** 2), 'b n h -> b n', 'mean')
        elif self.output_type == 'angle':
            loss = self.loss_fn(logit, test_y)
        elif self.output_type == 'image':
            logit = torch.tanh(logit).view_as(test_y)
            loss = reduce(((logit - test_y) ** 2), 'b n c h w -> b n', 'mean')
        else:
            raise NotImplementedError

        if loss_weight is not None:
            if len(loss.shape) == 3:
                loss_weight = repeat(loss_weight, 'b n -> b n y', y=loss.shape[-1])
            loss = loss_weight * loss

        output = Output()
        output[f'loss/meta_{meta_split}'] = reduce(loss, 'b ... -> b', 'mean')

        # Compute attention loss
        if self.config['attn_loss'] > 0:
            attn_losses = []
            for i, aux_output in enumerate(aux_outputs):
                if 'attn_logit' in aux_output:
                    # Get attention logit of test queries and train keys
                    attn_logit = aux_output['attn_logit'][:, :self.config['attn_loss_heads'], train_len:, :train_len]

                    # Compute log-sum-exp for train keys in each task
                    attn_logit = rearrange(
                        attn_logit, 'b h q (t l) -> b h q t l',
                        t=self.config['tasks'], l=(1 + y_len) * self.config['train_shots'])
                    task_logit = torch.logsumexp(attn_logit, dim=-1)
                elif 'attn' in aux_output:
                    # Get attention of test queries and train keys
                    attn = aux_output['attn'][:, :self.config['attn_loss_heads'], train_len:, :train_len]

                    # Compute sum for train keys in each task
                    task_attn = reduce(
                        attn, 'b h q (t l) -> b h q t', reduction='sum',
                        t=self.config['tasks'], l=(1 + y_len) * self.config['train_shots'])
                    task_logit = (task_attn + 1e-9).log()
                else:
                    raise RuntimeError('No attention logit or attention found in aux_output')

                task_gt = repeat(
                    torch.arange(self.config['tasks'], device=task_logit.device),
                    't -> b h (t s c)',
                    b=batch, h=self.config['attn_loss_heads'],
                    s=self.config['test_shots'], c=y_len)
                attn_loss = cross_entropy(task_logit, task_gt)
                attn_loss = reduce(attn_loss, 'b ... -> b', 'mean')
                output[f'attn_loss/meta_{meta_split}/{i}'] = attn_loss
                attn_losses.append(attn_loss)
            output[f'attn_loss_sum/meta_{meta_split}'] = sum(attn_losses)

        if not summarize:
            return output

        ############
        # Evaluate #
        ############

        if self.config['output_type'] == 'class':
            assert self.config['y_len'] == 1
            # Simple evaluation
            output.add_classification_summary(
                rearrange(logit, 'b n 1 v -> b n v'),
                rearrange(test_y_code, 'b n 1 -> b n'),
                meta_split)
        elif self.config['output_type'] == 'image':
            output.add_image_comparison_summary(test_x, test_y, test_x, logit, key=f'completion/meta_{meta_split}')

        return output

    def build_test_xy(self, test_x_enc, test_y_enc, attach_test_after):
        batch, test_num, y_len, hidden = test_y_enc.shape
        test_xy_enc, _ = pack([test_x_enc, test_y_enc[:, :, :-1]], 'b l * h')  # [batch, test_num, y_len, hidden]

        # Add positional encoding to test examples
        pos_idx = repeat(attach_test_after, f'b n -> b n {y_len}')
        pos_idx = pos_idx + rearrange(torch.arange(1, 1 + y_len, device=test_x_enc.device), 'l -> 1 1 l')
        test_pe = self.pos_enc.pe[pos_idx]  # [batch, test_num, y_len, hidden]
        test_xy_enc = test_xy_enc + test_pe
        test_xy_enc = rearrange(test_xy_enc, 'b n y h -> b (n y) h')

        return test_xy_enc

    def forward_tf(self, xy_enc, attach_test_after, train_len=0, past_states=None, return_states=False):
        if past_states is None:
            past_states = [None] * len(self.tf_layers)

        aux_outputs = []
        hidden = xy_enc
        for tf_layer, past_state in zip(self.tf_layers, past_states):
            hidden, aux_output = tf_layer(
                hidden, attach_test_after=attach_test_after, train_len=train_len,
                past_state=past_state, return_state=return_states)
            aux_outputs.append(aux_output)

        return hidden, aux_outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len, requires_grad=False):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.pe = None
        self.requires_grad = requires_grad
        self.build_pe(max_len)

    def build_pe(self, max_len):
        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(max_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(pe, requires_grad=self.requires_grad)

    def forward(self, x: Tensor, offset=0) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch, seq_len, hidden]
            offset: int, offset of the first position
        """
        x_len = x.size(-2)
        return x + self.pe[offset:offset + x_len]
