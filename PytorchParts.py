# encoderlayer imports (only necessary ones)
import torch
from copy import copy
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

# multiheaded attention imports except for those already in encoderlayer (only necessary ones)
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
# hopefully its looking for the same Module,F, and Linear as encoderlayer is:
# from .module import Module
# from .. import functional as F
# from . import Linear 

from torch.nn.functional import linear, softmax, dropout # needed for multi_head_attention_forward


def get_qkv(query,key,value,in_proj_weight,in_proj_bias,embed_dim):
	if torch.equal(query, key) and torch.equal(key, value):
		# self-attention
		q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

	elif torch.equal(key, value):
		# encoder-decoder attention
		# This is inline in_proj function with in_proj_weight and in_proj_bias
		_b = in_proj_bias
		_start = 0
		_end = embed_dim
		_w = in_proj_weight[_start:_end, :]
		if _b is not None:
			_b = _b[_start:_end]
		q = linear(query, _w, _b)

		if key is None:
			assert value is None
			k = None
			v = None
		else:

			# This is inline in_proj function with in_proj_weight and in_proj_bias
			_b = in_proj_bias
			_start = embed_dim
			_end = None
			_w = in_proj_weight[_start:, :]
			if _b is not None:
				_b = _b[_start:]
			k, v = linear(key, _w, _b).chunk(2, dim=-1)

	else:
		# This is inline in_proj function with in_proj_weight and in_proj_bias
		_b = in_proj_bias
		_start = 0
		_end = embed_dim
		_w = in_proj_weight[_start:_end, :]
		if _b is not None:
			_b = _b[_start:_end]
		q = linear(query, _w, _b)

		# This is inline in_proj function with in_proj_weight and in_proj_bias
		_b = in_proj_bias
		_start = embed_dim
		_end = embed_dim * 2
		_w = in_proj_weight[_start:_end, :]
		if _b is not None:
			_b = _b[_start:_end]
		k = linear(key, _w, _b)

		# This is inline in_proj function with in_proj_weight and in_proj_bias
		_b = in_proj_bias
		_start = embed_dim * 2
		_end = None
		_w = in_proj_weight[_start:, :]
		if _b is not None:
			_b = _b[_start:]
		v = linear(value, _w, _b)
	return q,k,v

def add_bias_if_given(k,v,bias_k,bias_v,static_k,static_v,attn_mask,key_padding_mask):
	if bias_k is not None and bias_v is not None:
		if static_k is None and static_v is None:
			k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
			v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
			if attn_mask is not None:
				attn_mask = torch.cat([attn_mask,
									  torch.zeros((attn_mask.size(0), 1),
												  dtype=attn_mask.dtype,
												  device=attn_mask.device)], dim=1)
			if key_padding_mask is not None:
				key_padding_mask = torch.cat(
					[key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
												   dtype=key_padding_mask.dtype,
												   device=key_padding_mask.device)], dim=1)
		else:
			assert static_k is None, "bias cannot be added to static key."
			assert static_v is None, "bias cannot be added to static value."
	else:
		assert bias_k is None
		assert bias_v is None
	return k,v,attn_mask,key_padding_mask

def use_static_kv_if_given(k,v,static_k,static_v):
	if static_k is not None:
		assert static_k.size(0) == bsz * num_heads
		assert static_k.size(2) == head_dim
		k = static_k

	if static_v is not None:
		assert static_v.size(0) == bsz * num_heads
		assert static_v.size(2) == head_dim
		v = static_v
	return k,v

def sort_dimensions(q,k,v,tgt_len,bsz,num_heads,head_dim):
	q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
	if k is not None:
		k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
	if v is not None:
		v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
	return q,k,v

def add_zero_attn_if_asked(add_zero_attn,src_len,k,v,attn_mask,key_padding_mask):
	if add_zero_attn:
		src_len += 1
		k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
		v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
		if attn_mask is not None:
			attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
														  dtype=attn_mask.dtype,
														  device=attn_mask.device)], dim=1)
		if key_padding_mask is not None:
			key_padding_mask = torch.cat(
				[key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
											   dtype=key_padding_mask.dtype,
											   device=key_padding_mask.device)], dim=1)
	return src_len,k,v,attn_mask,key_padding_mask

def add_masks(attn_mask,key_padding_mask,attn_output_weights,bsz,num_heads,tgt_len,src_len):
	if attn_mask is not None:
		attn_mask = attn_mask.unsqueeze(0)
		attn_output_weights += attn_mask

	if key_padding_mask is not None:
		attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
		attn_output_weights = attn_output_weights.masked_fill(
			key_padding_mask.unsqueeze(1).unsqueeze(2),
			float('-inf'),
		)
		attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
	return attn_mask, attn_output_weights


# taken from torch.nn.functional, specifically: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
def multi_head_attention_forward(query,                           # type: Tensor
								 key,                             # type: Tensor
								 value,                           # type: Tensor
								 embed_dim_to_check,              # type: int
								 num_heads,                       # type: int
								 in_proj_weight,                  # type: Tensor
								 in_proj_bias,                    # type: Tensor
								 bias_k,                          # type: Optional[Tensor]
								 bias_v,                          # type: Optional[Tensor]
								 add_zero_attn,                   # type: bool
								 dropout_p,                       # type: float
								 out_proj_weight,                 # type: Tensor
								 out_proj_bias,                   # type: Tensor
								 training=True,                   # type: bool
								 key_padding_mask=None,           # type: Optional[Tensor]
								 need_weights=True,               # type: bool
								 attn_mask=None,                  # type: Optional[Tensor]
								 q_proj_weight=None,              # type: Optional[Tensor]
								 k_proj_weight=None,              # type: Optional[Tensor]
								 v_proj_weight=None,              # type: Optional[Tensor]
								 static_k=None,                   # type: Optional[Tensor]
								 static_v=None,                   # type: Optional[Tensor]
								 get_attn_internals=False,        # type: bool
								 get_differentiable_attn=False,
								 ):
	# type: (...) -> Tuple[Tensor, Optional[Tensor]]
	r"""
	Args:
		query, key, value: map a query and a set of key-value pairs to an output.
			See "Attention Is All You Need" for more details.
		embed_dim_to_check: total dimension of the model.
		num_heads: parallel attention heads.
		in_proj_weight, in_proj_bias: input projection weight and bias.
		bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
		add_zero_attn: add a new batch of zeros to the key and
					   value sequences at dim=1.
		dropout_p: probability of an element to be zeroed.
		out_proj_weight, out_proj_bias: the output projection weight and bias.
		training: apply dropout if is ``True``.
		key_padding_mask: if provided, specified padding elements in the key will
			be ignored by the attention. This is an binary mask. When the value is True,
			the corresponding value on the attention layer will be filled with -inf.
		need_weights: output attn_output_weights.
		attn_mask: mask that prevents attention to certain positions. This is an additive mask
			(i.e. the values will be added to the attention layer).
		q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
		static_k, static_v: static key and value used for attention operators.


	Shape:
		Inputs:
		- query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
		  the embedding dimension.
		- key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
		  the embedding dimension.
		- value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
		  the embedding dimension.
		- key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
		- attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
		- static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
		  N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
		- static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
		  N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

		Outputs:
		- attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
		  E is the embedding dimension.
		- attn_output_weights: :math:`(N, L, S)` where N is the batch size,
		  L is the target sequence length, S is the source sequence length.
	"""
	tgt_len, bsz, embed_dim = query.size()
	assert embed_dim == embed_dim_to_check
	assert key.size() == value.size()
	assert not (get_attn_internals and need_weights) # can only return at most one of them

	head_dim = embed_dim // num_heads
	assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
	scaling = float(head_dim) ** -0.5


	q,k,v = get_qkv(query,key,value,in_proj_weight,in_proj_bias,embed_dim)
	q = q * scaling

	k,v,attn_mask,key_padding_mask = \
		add_bias_if_given(k,v,bias_k,bias_v,static_k,static_v,attn_mask,key_padding_mask)

	q,k,v = sort_dimensions(q,k,v,tgt_len,bsz,num_heads,head_dim)

	k,v = use_static_kv_if_given(k,v,static_k,static_v)

	src_len = k.size(1)

	if key_padding_mask is not None:
		assert key_padding_mask.size(0) == bsz
		assert key_padding_mask.size(1) == src_len

	src_len, k,v,attn_mask,key_padding_mask = \
		add_zero_attn_if_asked(add_zero_attn,src_len,k,v,attn_mask,key_padding_mask)

	attn_output_weights = torch.bmm(q, k.transpose(1, 2))
	assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

	attn_mask, attn_output_weights = \
		add_masks(attn_mask,key_padding_mask,attn_output_weights,bsz,num_heads,tgt_len,src_len)
	
	if get_attn_internals or get_differentiable_attn:
		attn_internals = AttnInternals(bsz,num_heads,tgt_len,src_len,keep_differentiable_attn=get_differentiable_attn)
		attn_internals.set_score(attn_output_weights) 
		# have to do this before the softmax ruins the scores
	attn_output_weights = softmax(
		attn_output_weights, dim=-1)
	attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

	attn_output = torch.bmm(attn_output_weights, v)
	assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
	attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
	attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

	if get_attn_internals:
		attn_internals.set_distributions(attn_output_weights)
		attn_internals.set_q_k(q,k)
		return attn_output, attn_internals
	if need_weights:
		# average attention weights over heads # why on EARTH would i want the AVERAGE??
		attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
		return attn_output, attn_output_weights.sum(dim=1) / num_heads
	else:
		return attn_output, None

class AttnInternals:
	def __init__(self,bsz,num_heads,tgt_len,src_len,keep_differentiable_attn=False):
		self.bsz = bsz
		self.num_heads = num_heads
		self.tgt_len = tgt_len
		self.src_len = src_len
		self.keep_differentiable_attn = keep_differentiable_attn
	def _reshape_scores_or_distrs(self,s):
		return s.view(self.bsz,self.num_heads,self.tgt_len,self.src_len)
	def set_score(self,scores):
		self.scores = copy(self._reshape_scores_or_distrs(scores).data)
	def set_distributions(self,distrs):
		if self.keep_differentiable_attn:
			self.differentiable_distributions = self._reshape_scores_or_distrs(distrs)
		self.distributions = copy(self._reshape_scores_or_distrs(distrs).data)
	def set_q_k(self,qs,ks):
		self.qs = copy(qs.data)
		self.ks = copy(ks.data)

# taken from pytorch source code, torch.nn.modules.activation. want to mess with it
# original version location: https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html
class MultiheadAttention(Module): 
	r"""Allows the model to jointly attend to information
	from different representation subspaces.
	See reference: Attention Is All You Need

	.. math::
		\text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
		\text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

	Args:
		embed_dim: total dimension of the model.
		num_heads: parallel attention heads.
		dropout: a Dropout layer on attn_output_weights. Default: 0.0.
		bias: add bias as module parameter. Default: True.
		add_bias_kv: add bias to the key and value sequences at dim=0.
		add_zero_attn: add a new batch of zeros to the key and
					   value sequences at dim=1.
		kdim: total number of features in key. Default: None.
		vdim: total number of features in key. Default: None.

		Note: if kdim and vdim are None, they will be set to embed_dim such that
		query, key, and value have the same number of features.

	Examples::

		>>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
		>>> attn_output, attn_output_weights = multihead_attn(query, key, value)
	"""
	__annotations__ = {
		'bias_k': torch._jit_internal.Optional[torch.Tensor],
		'bias_v': torch._jit_internal.Optional[torch.Tensor],
	}
	__constants__ = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']

	def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
					keep_differentiable_attn=False):
		super(MultiheadAttention, self).__init__()
		self.embed_dim = embed_dim
		self.kdim = embed_dim
		self.vdim = embed_dim

		self.num_heads = num_heads
		self.dropout = dropout
		self.head_dim = embed_dim // num_heads
		assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

		self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
		self.register_parameter('q_proj_weight', None)
		self.register_parameter('k_proj_weight', None)
		self.register_parameter('v_proj_weight', None)

		if bias:
			self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
		else:
			self.register_parameter('in_proj_bias', None)
		self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

		if add_bias_kv:
			self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
			self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
		else:
			self.bias_k = self.bias_v = None

		self.add_zero_attn = add_zero_attn
		self.keep_differentiable_attn = keep_differentiable_attn
		self._reset_parameters()


	def _reset_parameters(self):
		xavier_uniform_(self.in_proj_weight)

		if self.in_proj_bias is not None:
			constant_(self.in_proj_bias, 0.)
			constant_(self.out_proj.bias, 0.)
		if self.bias_k is not None:
			xavier_normal_(self.bias_k)
		if self.bias_v is not None:
			xavier_normal_(self.bias_v)

	def __setstate__(self, state):
		super(MultiheadAttention, self).__setstate__(state)

	def forward(self, query, key, value, key_padding_mask=None,
				need_weights=True, attn_mask=None,get_attn_internals=False):
		# type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
		r"""
	Args:
		query, key, value: map a query and a set of key-value pairs to an output.
			See "Attention Is All You Need" for more details.
		key_padding_mask: if provided, specified padding elements in the key will
			be ignored by the attention. This is an binary mask. When the value is True,
			the corresponding value on the attention layer will be filled with -inf.
		need_weights: output attn_output_weights.
		attn_mask: mask that prevents attention to certain positions. This is an additive mask
			(i.e. the values will be added to the attention layer).

	Shape:
		- Inputs:
		- query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
		  the embedding dimension.
		- key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
		  the embedding dimension.
		- value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
		  the embedding dimension.
		- key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
		- attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.

		- Outputs:
		- attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
		  E is the embedding dimension.
		- attn_output_weights: :math:`(N, L, S)` where N is the batch size,
		  L is the target sequence length, S is the source sequence length.
		"""
		return multi_head_attention_forward(
			query, key, value, self.embed_dim, self.num_heads,
			self.in_proj_weight, self.in_proj_bias,
			self.bias_k, self.bias_v, self.add_zero_attn,
			self.dropout, self.out_proj.weight, self.out_proj.bias,
			training=self.training,
			key_padding_mask=key_padding_mask, need_weights=need_weights,
			attn_mask=attn_mask,get_attn_internals=get_attn_internals,
			get_differentiable_attn=self.keep_differentiable_attn)


# copied directly from pytorch code, just want to use it with a different multiheadattention is all
# original version location: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
class TransformerEncoderLayer(Module): 
	r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
	This standard encoder layer is based on the paper "Attention Is All You Need".
	Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
	Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
	Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
	in a different way during application.

	Args:
		d_model: the number of expected features in the input (required).
		nhead: the number of heads in the multiheadattention models (required).
		dim_feedforward: the dimension of the feedforward network model (default=2048).
		dropout: the dropout value (default=0.1).
		activation: the activation function of intermediate layer, relu or gelu (default=relu).

	Examples::
		>>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
		>>> src = torch.rand(10, 32, 512)
		>>> out = encoder_layer(src)
	"""

	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
		super(TransformerEncoderLayer, self).__init__()
		self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
		# Implementation of Feedforward model
		self.linear1 = Linear(d_model, dim_feedforward)
		self.dropout = Dropout(dropout)
		self.linear2 = Linear(dim_feedforward, d_model)

		self.norm1 = LayerNorm(d_model)
		self.norm2 = LayerNorm(d_model)
		self.dropout1 = Dropout(dropout)
		self.dropout2 = Dropout(dropout)

		self.activation = _get_activation_fn(activation)
		self.keep_attn_internals = False
		self.keep_vals = False

	def set_keeps(self,b):
		self.keep_vals = b
		self.keep_attn_internals = b

	def clear_keeps(self):
		self.attn_internals = None
		self.stored_in = None
		self.stored_out = None

	def training_attn(self,b):
		self.self_attn.keep_differentiable_attn = b

	def forward(self, src, src_mask=None, src_key_padding_mask=None):
		r"""Pass the input through the encoder layer.

		Args:
			src: the sequnce to the encoder layer (required).
			src_mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).

		Shape:
			see the docs in Transformer class.
		"""
		if self.keep_vals:
			self.stored_in = copy(src.data)

		keep_attn_internals = self.keep_attn_internals or self.self_attn.keep_differentiable_attn

		src2, attn_internals = self.self_attn(src, src, src, attn_mask=src_mask,
							  key_padding_mask=src_key_padding_mask,
							  need_weights=False,get_attn_internals=keep_attn_internals)
							  # it was calling it with its default, i.e. need_weights=True, 
							  # computing all those averages on the scores for nothing???????????? (never using them)
		if keep_attn_internals:
			self.attn_internals = attn_internals
			

		src = src + self.dropout1(src2)
		src = self.norm1(src)
		if hasattr(self, "activation"):
			src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
		else:  # for backward compatibility
			src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
		src = src + self.dropout2(src2)
		src = self.norm2(src)

		if self.keep_vals:
			self.stored_out = copy(src.data)
		
		return src

def _get_activation_fn(activation):
	if activation == "relu":
		return F.relu
	elif activation == "gelu":
		return F.gelu
	else:
		raise RuntimeError("activation should be relu/gelu, not %s." % activation)