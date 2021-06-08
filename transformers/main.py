from requests import head
import torch as t
import torch.nn as nn

# multi headed attention block to be used
# masking functionality is also implemented for better abstraction
class SelfAttention(nn.Module):
	def __init__(self, embed_size, heads):
		super(SelfAttention, self).__init__()

		self.embed_size = embed_size
		self.heads = heads
		self.head_dim = embed_size // heads

		assert (self.head_dim * self.heads == self.embed_size), "Embed size should be div by heads"

		# inputs of the Multi Head Attention
		self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.quaries = nn.Linear(self.head_dim, self.head_dim, bias=False)

		# outputs of the MHA
		self.fc_out = nn.Linear(embed_size, embed_size)

	# forward pass of the model
	def forward(self, values, keys, quaries, mask):
		N = quaries.shape[0]
		value_len, key_len, quary_len = values.shape[1], keys.shape[1], quaries.shape[1]

		values = values.reshape(N, value_len, self.heads, self.head_dim)
		keys = keys.reshape(N, key_len, self.heads, self.head_dim)
		quaries = quaries.reshape(N, quary_len, self.heads, self.head_dim)

		values = self.values(values)
		keys = self.keys(keys)
		quaries = self.quaries(quaries)

		# using einsum to calculate energy
		"""
		quaries shape: (N, quary_len, self.heads, self.head_dim)
		keys shape: (N, key_len, self.heads, self.head_dim)
		
		energy shape: (N, heads, quary_len, key_len)
		"""
		energy = t.einsum("nqhd,nkhd->nhqk", [quaries, keys])

		# apply masking is required
		if mask is not None:
			energy = energy.masked_fill(mask == 0, float("-1e20"))

		# Scaled dot product attention
		# dim=3 normalises data with key_len
		attention = t.softmax(energy / (self.embed_size ** (1/2)), dim=3)

		# final MatMul using einsum
		# attention shape: (N, heads, quary_len, key_len)
		# values shape: (N, value_len, heads, heads_dim)
		# out shape: (N, quary_len, heads, heads_dim)
		# after the einsum flatten the last two dimentions
		out = t.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, quary_len, self.heads * self.head_dim)
		out = self.fc_out(out)
		return out



# Transformer Block
# one repititive block of the encoder
# this is used in decoder too
class TransformerBlock(nn.Module):
	def __init__(self, embed_size, heads, dropout, forward_expansion):
		super(TransformerBlock, self).__init__()

		# attention mechanism
		self.attention = SelfAttention(embed_size, heads)

		# normalisation of attention
		self.norm1 = nn.LayerNorm(embed_size)

		# feed forward network
		self.feed_forward = nn.Sequential(
			nn.Linear(embed_size, forward_expansion * embed_size),
			nn.ReLU(),
			nn.Linear(embed_size * forward_expansion, embed_size)
		)	

		# normalisation of feed forward network
		self.norm2 = nn.LayerNorm(embed_size)

		self.dropout = nn.Dropout(dropout)

	def forward(self, value, key, quary, mask):
		attention = self.attention(value, key, quary, mask)

		# normalisation with skip connection
		x = self.dropout(self.norm1(attention + quary))

		# feedforwarding 
		forward = self.feed_forward(x)

		# geting norm with skip connection 
		out = self.dropout(self.norm2(forward + x))

		return out




class Encoder(nn.Module):
	def __init__(
		self,
		src_vocab_size,
		embed_size,
		num_layers,
		heads, 
		device,
		forward_expansion,
		dropout, 
		max_length
	):
		super(Encoder, self).__init__()

		self.embed_size = embed_size
		self.device = device

		# embedder
		self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
		self.position_embedding = nn.Embedding(max_length, embed_size)

		# actual Tranformer block
		self.layers = nn.ModuleList(
			[
				TransformerBlock(
					embed_size, 
					heads, 
					dropout=dropout,
					forward_expansion=forward_expansion
				) for _ in range(num_layers)
			]
		)

		self.dropout = nn.Dropout(dropout)

	def forward(self, x, mask):
		N, seq_length = x.shape
		positions = t.arange(0, seq_length).expand(N, seq_length).to(self.device)

		# embedding
		out = self.word_embedding(x) + self.position_embedding(positions)

		for layer in self.layers:
			out = layer(out, out, out, mask)

		return out



# one repetitive block of decoder 
class DecoderBlock(nn.Module):
	def __init__(self, embed_size, heads, forward_expansion, dropout, device):
		super(DecoderBlock, self).__init__()

		# attention part of the decoder (masked multi headed attention)
		self.attention = SelfAttention(embed_size, heads)

		# normalisation of attention output
		self.norm = nn.LayerNorm(embed_size)

		# second part of the decoder is just another transformer block
		self.tranformer_block = TransformerBlock(
			embed_size, heads, dropout, forward_expansion
		)

		self.dropout = nn.Dropout(dropout)

	def forward(self, x, value, key, src_mask, trg_mask):
		# masked multi headed attention
		attention = self.attention(x, x, x, trg_mask)

		# query from the masked multi headed attention, normalised and skipped
		query = self.dropout(self.norm(attention + x))

		# final transformer block
		out = self.tranformer_block(value, key, query, src_mask)

		return out


# decoder of the transformer
class Decoder(nn.Module):
	def __init__(
			self, 
			trg_vocab_size,
			embed_size,
			num_layers,
			heads,
			forward_expansion, 
			droput,
			device, 
			max_length):
	    	
		super(Decoder, self).__init__()

		self.device = device

		# embedding
		self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
		self.position_embedding = nn.Embedding(max_length, embed_size)

		# actual multiple transformer blocks
		self.layers = nn.ModuleList(
			[
				DecoderBlock(
					embed_size, 
					heads, 
					forward_expansion,
					droput,
					device
				) for _ in range(num_layers)
			]
		)

		# output of the decoder block
		self.fc_out = nn.Linear(embed_size, trg_vocab_size)
		self.dropout = nn.Dropout(droput)

	def forward(self, x, enc_out, src_mask, trg_mask):
		N, seq_length = x.shape 
		
		# embedder portion of the Decoder
		position = t.arange(0, seq_length).expand(N, seq_length).to(self.device)
		x = self.dropout(self.word_embedding(x)+self.position_embedding(position))

		# running through all the decoder blocks
		for layer in self.layers:
			x = layer(x, enc_out, enc_out, src_mask, trg_mask)

		out = self.fc_out(x)

		return out


class Tranformer(nn.Module):
	def __init__(
		self,
		src_vocab_size,
		trg_vocab_size,
		src_pad_idx,
		trg_pad_idx,
		embed_size = 256,
		num_layers = 6,
		forward_expansion = 4,
		heads = 8,
		dropout = 0,
		device = "cpu",
		max_length = 100
	):
	    super().__init__()

	    # encoder part
	    self.encoder = Encoder(
		    src_vocab_size,
		    embed_size,
		    num_layers,
		    heads,
		    device,
		    forward_expansion,
		    dropout,
		    max_length
	    )

	    # decoder part
	    self.decoder = Decoder(
		    trg_vocab_size,
		    embed_size,
		    num_layers,
		    heads,
		    forward_expansion,
		    dropout,
		    device,
		    max_length
	    )

	    self.trg_pad_idx = trg_pad_idx
	    self.src_pad_idx = src_pad_idx
	    self.device = device

	def make_src_mask(self, src:t.Tensor):
		src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
		# (N, 1, 1, src_len)
		return src_mask.to(self.device)

	def make_trg_mask(self, trg:t.Tensor):
		N, trg_len = trg.shape
		trg_mask = t.tril(t.ones((trg_len, trg_len))).expand(
			N, 1, trg_len, trg_len
		)
		return trg_mask.to(self.device)

	def forward(self, src: t.Tensor, trg: t.Tensor):
		src_mask = self.make_src_mask(src)
		trg_mask = self.make_trg_mask(trg)

		enc_src = self.encoder(src, src_mask)
		out = self.decoder(trg, enc_src, src_mask, trg_mask)

		return out

if __name__ == '__main__':
	device = t.device('cpu')

	x = t.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
	trg = t.tensor([[1, 7, 4, 3, 5, 9, 2, 0],[1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

	model = Tranformer(10, 10, 0, 0).to(device)

	out = model(x, trg[:, :-1])

	print(out.shape)