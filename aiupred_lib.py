import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.functional import pad
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import logging
import os

PATH = os.path.dirname(os.path.abspath(__file__))
AA_CODE = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
WINDOW = 100


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = 32
        self.pos_encoder = PositionalEncoding(self.d_model)
        encoder_layers = TransformerEncoderLayer(self.d_model, 2, 64, 0, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)
        self.encoder = nn.Embedding(21, self.d_model)
        self.decoder = nn.Linear((WINDOW + 1) * self.d_model, 1)

    def forward(self, src: Tensor, embed_only=False) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)  # (Batch x Window+1 x Embed_dim
        embedding = self.transformer_encoder(src)
        if embed_only:
            return embedding
        output = torch.flatten(embedding, 1)
        output = self.decoder(output)
        return torch.squeeze(output)


class DecoderModel(nn.Module):
    def __init__(self, pred_type):
        super().__init__()
        input_dim = (WINDOW + 1) * (WINDOW + 1) * 32
        output_dim = 1
        current_dim = input_dim
        if pred_type == 'disorder':
            layer_architecture = [64, 64, 64, 64, 16]
        else:
            layer_architecture = [64, 64, 64, 16]
        self.layers = nn.ModuleList()
        for hdim in layer_architecture:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        output = torch.sigmoid(self.layers[-1](x))
        return torch.squeeze(output)


@torch.no_grad()
def tokenize(sequence, device):
    """
    Tokenize an amino acid sequence. Non-standard amino acids are treated as X
    :param sequence: Amino acid sequence in string
    :param device: Device to run on. CUDA{x} or CPU
    :return: Tokenized tensors
    """
    return torch.tensor([AA_CODE.index(aa) if aa in AA_CODE else 20 for aa in sequence], device=device)


def predict(sequence, embedding_model, decoder_model, device, smoothing=None):
    _tokens = tokenize(sequence, device)
    _padded_token = pad(_tokens, (WINDOW // 2, WINDOW // 2), 'constant', 0)
    _unfolded_tokes = _padded_token.unfold(0, WINDOW + 1, 1)
    _token_embedding = embedding_model(_unfolded_tokes, embed_only=True)
    _padded_embed = pad(_token_embedding, (0, 0, 0, 0, WINDOW // 2, WINDOW // 2), 'constant', 0)
    _unfolded_embedding = _padded_embed.unfold(0, WINDOW + 1, 1)
    _decoder_input = _unfolded_embedding.permute(0, 2, 1, 3)
    _prediction = decoder_model(_decoder_input).detach().cpu().numpy()
    if smoothing:
        _prediction = smoothing(_prediction)
    return _prediction


def low_memory_predict(sequence, embedding_model, decoder_model, device, smoothing=None, chunk_len=1000):
    overlap = 100
    if chunk_len <= overlap:
        raise ValueError("Chunk len must be bigger than 200!")
    overlapping_predictions = []
    for chunk in range(0, len(sequence), chunk_len-overlap):
        overlapping_predictions.append(predict(
            sequence[chunk:chunk+chunk_len],
            embedding_model,
            decoder_model,
            device
        ))
    prediction = np.concatenate((overlapping_predictions[0], *[x[overlap:] for x in overlapping_predictions[1:]]))
    return prediction


def multifasta_reader(file_handler):
    """
    (multi) FASTA reader function
    :return: Dictionary with header -> sequence mapping from the file
    """
    if type(file_handler) == str:
        file_handler = open(file_handler)
    sequence_dct = {}
    header = None
    for line in file_handler:
        if line.startswith('>'):
            header = line.strip()
            sequence_dct[header] = ''
        elif line.strip():
            sequence_dct[header] += line.strip()
    file_handler.close()
    return sequence_dct


def init_models(prediction_type, force_cpu=False, gpu_num=0):
    """
    Initialize networks and device to run on
    :param prediction_type: Prediction to carry out. Either disorder or binding
    :param force_cpu: Force the method to run on CPU only mode
    :param gpu_num: Index of the GPU to use, default=0
    :return: Tuple of (embedding_model, regression_model, device)
    """
    if prediction_type not in ['disorder', 'binding']:
        raise ValueError('Prediction type must either be disorder or binding!')
    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
    if force_cpu:
        device = 'cpu'
    logging.debug(f'Running on {device}')
    if device == 'cpu':
        print('# Warning: No GPU found, running on CPU. It is advised to run AIUPred on a GPU')

    embedding_model = TransformerModel()
    embedding_model.load_state_dict(torch.load(f'{PATH}/data/embedding.pt', map_location=device))
    embedding_model.to(device)
    embedding_model.eval()

    decoder_model = DecoderModel(prediction_type)
    decoder_model.load_state_dict(torch.load(f'{PATH}/data/{prediction_type}_decoder.pt', map_location=device))
    decoder_model.to(device)
    decoder_model.eval()

    logging.debug("Networks initialized")

    return embedding_model, decoder_model, device


def aiupred_disorder(sequence, force_cpu=False, gpu_num=0):
    """
    Library function to carry out single sequence analysis
    :param sequence: Amino acid sequence in a string
    :param force_cpu: Force the method to run on CPU only mode
    :param gpu_num: Index of the GPU to use, default=0
    :return: Numpy array with disorder propensities for each position
    """
    embedding_model, decoder, device = init_models('disorder', force_cpu, gpu_num)
    return predict(sequence, embedding_model, decoder, device)


def aiupred_binding(sequence, force_cpu=False, gpu_num=0):
    """
    Library function to carry out single sequence analysis
    :param sequence: Amino acid sequence in a string
    :param force_cpu: Force the method to run on CPU only mode
    :param gpu_num: Index of the GPU to use, default=0
    :return: Numpy array with disorder propensities for each position
    """
    embedding_model, decoder, device = init_models('binding', force_cpu, gpu_num)
    return predict(sequence, embedding_model, decoder, device)


def main(multifasta_file, prediction_type, force_cpu=False, gpu_num=0):
    """
    Main function to be called from aiupred.py
    :param prediction_type: Prediction to carry out. Either disorder or binding
    :param multifasta_file: Location of (multi) FASTA formatted sequences
    :param force_cpu: Force the method to run on CPU only mode
    :param gpu_num: Index of the GPU to use, default=0
    :return: Dictionary with parsed sequences and predicted results
    """
    if prediction_type not in ['disorder', 'binding']:
        raise ValueError('Prediction type must either be disorder or binding!')
    embedding_model, reg_model, device = init_models(prediction_type, force_cpu, gpu_num)
    sequences = multifasta_reader(multifasta_file)
    logging.debug("Sequences read")
    logging.info(f'{len(sequences)} sequences read')
    if not sequences:
        raise ValueError("FASTA file is empty")
    results = {}
    logging.StreamHandler.terminator = ""
    for num, (ident, sequence) in enumerate(sequences.items()):
        results[ident] = {}
        results[ident]['aiupred'] = predict(sequence, embedding_model, reg_model, device)
        results[ident]['sequence'] = sequence
        logging.debug(f'{num}/{len(sequences)} sequences done...\r')
    logging.StreamHandler.terminator = '\n'
    logging.debug('Analysis done, writing output')
    return results


if __name__ == '__main__':
    print("This is a library, this should not be called directly! Please refer to readme.md for more information!")
