<pre>          _____ _    _ _____              _ 
    /\   |_   _| |  | |  __ \            | |
   /  \    | | | |  | | |__) | __ ___  __| |
  / /\ \   | | | |  | |  ___/ '__/ _ \/ _` |
 / ____ \ _| |_| |__| | |   | | |  __/ (_| |
/_/    \_\_____|\____/|_|   |_|  \___|\__,_|</pre>

Made by Gabor Erdos and Zsuzsanna Dosztanyi \
CAID3 version

# Requirements

AIPred is only dependent on PyTorch. In order to install it use

`pip install torch`

or

`pip install -r requirements.txt`

AIUPred was tested on PyTorch v 2.2.2 and Python 3.10.12

AIUPred uses on an average 2.5GB RAM for a protein with 1000 residues. This number includes the loaded network(a) as well!

AIUPred can utilize a dedicated GPU (NVidia) to seep up the prediction immensely. By default AIUPred searches for available GPUs and will use the fist indexed one. In case no GPU is found, AIUPred will fall back to CPU usage with a warning.
# How to use

## Using the `aiupred.py` script

AIUPred is very fast, the time limiting step is the reading of the network itself, so it is recommended to load the network into memory, and keep it there while analyzing files

The `aiupred.py` script allows the analysation of multi FASTA formatted files while only loading the network data once.

Example usage: `python3 aiupred.py -i data/test.fasta`

Expected output:

```
# ...
# Gabor Erdos, Zsuzsanna Dosztanyi
# v1.5, CAID3 edition

>sp|P04637|P53_HUMAN Cellular tumor antigen p53 OS=Homo sapiens OX=9606 GN=TP53 PE=1 SV=4
1       M       0.2416
2       E       0.1395
3       E       0.1538
4       P       0.0940
...


>sp|P35222|CTNB1_HUMAN Catenin beta-1 OS=Homo sapiens OX=9606 GN=CTNNB1 PE=1 SV=1
1       M       0.7144
2       A       0.7177
3       T       0.7407
4       Q       0.7362
...
```

Other options:

```
usage: aiupred.py [-h] -i INPUT_FILE [-o OUTPUT_FILE] [-v] [-g GPU] [-m MODE] [--force-cpu]

AIUPred disorder prediction method v1.5
Developed by Gabor Erdos and Zsuzsanna Dosztanyi
Version generated for CAID3

options:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input_file INPUT_FILE
                        Input file in (multi) FASTA format
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Output file
  -v, --verbose         Increase output verbosity
  -g GPU, --gpu GPU     Index of GPU to use, default=0
  -m MODE, --mode MODE  Mode or prediction type. Either 'disorder' or 'binding'. Default is 'disorder'
  --force-cpu           Force the network to only utilize the CPU. Calculation will be very slow, not recommended

```

## Output format
Each sequence inside the multi FASTA will be analysed sequentially. For each sequence the result will contain the header of the sequence followed by a tab separated table like structure in the format of
`Position Residue Prediction`

Prediction values are ranging from 0 to 1 in a continuous scale. 

## Using the `aiupred_lib.py` library

AIUPred also come with an importable python3 library called `aiupred_lib`. In order to use the library it first should be added to the PYTHON_PATH environment variable.

`export PYTHONPATH="${PYTHONPATH}:/path/to/aiupred/folder"`

After reloading the shell aiupred_lib should be importable in your python environment.

```python
import aiupred_lib
# Load the models and let AIUPred find if a GPU is available.     
embedding_model, regression_model, device = aiupred_lib.init_models('disorder')
# Predict disorder of a sequence
sequence = 'THISISATESTSEQENCE'
prediction = aiupred_lib.predict(sequence, embedding_model, regression_model, device)
```

# Low memory prediction

In case sufficient memory is not available (AIUPred requires ~2.5GB of memory for 1000 residues) the aiupred_lib.py file contains a low memory prediction function.

This function splits the sequence into `chunk_len` segments which overlap in 100 positions, carries out an iterative prediction and concatenates the results.

Due to padding during the decoder network the results are not 100% identical to full sequence predictions, but the difference is negligible. 
The `chunk_len` optional keyword argument can be set to lower the memory usage. By default it is set to 1000. Lowering this variable decreases memory usage but increases running time and lowers precision.

Example:

```python
sequence = ['A', 'C', 'T', 'F', 'Q'] * 2000  # 10000 residue long sequence
embedding, decoder, device = init_models('disorder')
prediction = low_memory_predict(sequence, embedding, decoder, device)
```

# Method

AIUPred uses an energy embedding transformer network followed by a specific fully connected decoder network for disorder and binding.
The transformer was trained on the calculated energies of high resolution monomer structures from the PDB
The disorder decoder was trained on the AlphaFold2 RSA values
The binding decoder was trained on AlphaMissense scores combined with Alphafold2 RSA values
