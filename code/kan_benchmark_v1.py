"""
KAN Models Benchmark with Multiple Embeddings.

For 20th iSAI-NLP IEEE Paper "Enhancing Burmese News Classification with Kolmogorov-Arnold Network Head Fine-tuning (Thura Aung, Eaint Kay Khaing Kyaw, Ye Kyaw Thu, Thazin Myint Oo, Thepchai Supnithi)"

This module provides a comprehensive benchmarking framework for comparing different
KAN (Kolmogorov-Arnold Network) implementations with various text embedding strategies
on the Myanmar News classification dataset.

Usage:
    python kan_benchmark_v1.py --embed tfidf --model all --epochs 15
    python kan_benchmark_v1.py --embed mbert --tune --transformer xlm-roberta-base
"""

import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time
from fasterkan.fasterkan import FasterKAN
from efficient_kan import KAN
from torchkan import KAL_Net
from fastkan.fastkan import FastKAN as FastKANORG
import fasttext
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel

# --- Embedding Classes ---
class TfidfEmbedding:
    """TF-IDF text embedding generator.
    
    Converts text documents into TF-IDF (Term Frequency-Inverse Document Frequency)
    vector representations using unigrams and bigrams.
    
    Args:
        max_features (int, optional): Maximum number of features to extract. Defaults to 1000.
    
    Attributes:
        vectorizer (TfidfVectorizer): Sklearn TF-IDF vectorizer instance.
        embedding_dim (int): Dimensionality of output embeddings.
    """
    
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.embedding_dim = max_features
    
    def fit_transform(self, texts):
        """Fit vectorizer on texts and transform them to TF-IDF vectors.
        
        Args:
            texts (list of str): Training text documents.
            
        Returns:
            numpy.ndarray: TF-IDF matrix of shape (n_samples, max_features).
        """
        return self.vectorizer.fit_transform(texts).toarray()
    
    def transform(self, texts):
        """Transform texts to TF-IDF vectors using fitted vectorizer.
        
        Args:
            texts (list of str): Text documents to transform.
            
        Returns:
            numpy.ndarray: TF-IDF matrix of shape (n_samples, max_features).
        """
        return self.vectorizer.transform(texts).toarray()

class RandomEmbedding:
    """Random initialized word embeddings.
    
    Creates randomly initialized word embeddings and averages word vectors
    for each document. Useful as a baseline embedding method.
    
    Args:
        vocab_size (int, optional): Maximum vocabulary size. Defaults to 10000.
        embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 300.
    
    Attributes:
        vocab_size (int): Maximum number of words in vocabulary.
        embedding_dim (int): Dimension of word embeddings.
        word2idx (dict): Mapping from words to vocabulary indices.
        embedding (nn.Embedding): PyTorch embedding layer.
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=300):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word2idx = {}
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def _build_vocab(self, texts):
        """Build vocabulary from training texts.
        
        Args:
            texts (list of str): Training text documents.
        """
        words = set()
        for text in texts:
            words.update(text.split())
        self.word2idx = {word: idx for idx, word in enumerate(list(words)[:self.vocab_size-1])}
        self.word2idx['<UNK>'] = self.vocab_size - 1
    
    def _text_to_indices(self, text):
        """Convert text to list of vocabulary indices.
        
        Args:
            text (str): Input text document.
            
        Returns:
            list of int: Vocabulary indices for each word.
        """
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in text.split()]
        return indices
    
    def fit_transform(self, texts):
        """Build vocabulary and transform texts to averaged embeddings.
        
        Args:
            texts (list of str): Training text documents.
            
        Returns:
            numpy.ndarray: Document embeddings of shape (n_samples, embedding_dim).
        """
        self._build_vocab(texts)
        embeddings = []
        for text in texts:
            indices = self._text_to_indices(text)
            if len(indices) == 0:
                embeddings.append(torch.zeros(self.embedding_dim))
            else:
                text_emb = self.embedding(torch.tensor(indices)).mean(dim=0)
                embeddings.append(text_emb)
        return torch.stack(embeddings).detach().numpy()
    
    def transform(self, texts):
        """Transform texts to averaged embeddings using fitted vocabulary.
        
        Args:
            texts (list of str): Text documents to transform.
            
        Returns:
            numpy.ndarray: Document embeddings of shape (n_samples, embedding_dim).
        """
        embeddings = []
        for text in texts:
            indices = self._text_to_indices(text)
            if len(indices) == 0:
                embeddings.append(torch.zeros(self.embedding_dim))
            else:
                text_emb = self.embedding(torch.tensor(indices)).mean(dim=0)
                embeddings.append(text_emb)
        return torch.stack(embeddings).detach().numpy()

class FastTextEmbedding:
    def __init__(self):
        print("Loading FastText model...")
        model_path = hf_hub_download(repo_id="facebook/fasttext-my-vectors", filename="model.bin")
        self.model = fasttext.load_model(model_path)
        self.embedding_dim = self.model.get_dimension()
        print(f"FastText loaded. Vocabulary size: {len(self.model.words)}, Embedding dim: {self.embedding_dim}")
    
    def _get_text_embedding(self, text):
        words = text.split()
        if len(words) == 0:
            return np.zeros(self.embedding_dim)
        embeddings = [self.model[word] for word in words]
        return np.mean(embeddings, axis=0)
    
    def fit_transform(self, texts):
        return np.array([self._get_text_embedding(text) for text in texts])
    
    def transform(self, texts):
        return np.array([self._get_text_embedding(text) for text in texts])

class TransformerEmbedding:
    def __init__(self, model_name='google-bert/bert-base-multilingual-cased', device='cpu'):
        print(f"Loading transformer model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.embedding_dim = self.model.config.hidden_size
        print(f"Transformer loaded. Embedding dim: {self.embedding_dim}")
    
    def _get_embeddings(self, texts, batch_size=32):
        self.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def fit_transform(self, texts):
        return self._get_embeddings(texts)
    
    def transform(self, texts):
        return self._get_embeddings(texts)

# --- Load Myanmar News dataset ---
def load_myanmar_news(device='cpu', embed_type='tfidf', transformer_name=None):
    dataset = load_dataset("ThuraAung1601/myanmar_news")
    texts_train = list(dataset['train']['text'])
    labels_train = list(dataset['train']['label'])
    texts_test = list(dataset['test']['text'])
    labels_test = list(dataset['test']['label'])

    # Map string labels to integers
    unique_labels = sorted(set(labels_train + labels_test))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    labels_train_int = [label2id[l] for l in labels_train]
    labels_test_int = [label2id[l] for l in labels_test]

    # Select embedding method
    if embed_type == 'tfidf':
        embedder = TfidfEmbedding(max_features=1000)
    elif embed_type == 'random':
        embedder = RandomEmbedding(vocab_size=10000, embedding_dim=300)
    elif embed_type == 'fasttext':
        embedder = FastTextEmbedding()
    elif embed_type == 'mbert':
        model_name = transformer_name or 'google-bert/bert-base-multilingual-cased'
        embedder = TransformerEmbedding(model_name=model_name, device=device)
    else:
        raise ValueError(f"Unknown embedding type: {embed_type}")
    
    # Get embeddings
    print(f"Generating {embed_type} embeddings...")
    X_train = embedder.fit_transform(texts_train)
    X_test = embedder.transform(texts_test)
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(labels_train_int, dtype=torch.long, device=device)
    y_test = torch.tensor(labels_test_int, dtype=torch.long, device=device)

    print(f"Embedding dimension: {X_train.shape[1]}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    return {
        'train_input': X_train, 'train_label': y_train,
        'test_input': X_test, 'test_label': y_test,
        'embedder': embedder
    }, len(unique_labels)

# --- Simple MLP with optional embedding layer ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, has_embedding=False, embedder=None, tune_embedding=False):
        super().__init__()
        self.has_embedding = has_embedding
        self.tune_embedding = tune_embedding
        
        if has_embedding and embedder is not None and isinstance(embedder, RandomEmbedding):
            self.embedding = embedder.embedding
            if not tune_embedding:
                for param in self.embedding.parameters():
                    param.requires_grad = False
        else:
            self.embedding = None
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# --- Training function ---
def train_model(model, dataset, device, epochs=20, batch_size=32, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(dataset['train_input'], dataset['train_label'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# --- Speed benchmark ---
def benchmark_speed(model, dataset, device, batch_size=64, reps=10):
    model.eval()
    forward_times, backward_times = [], []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Warm up
    for _ in range(3):
        idxs = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
        x = dataset['train_input'][idxs]
        y = dataset['train_label'][idxs]
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.zero_grad()

    for _ in range(reps):
        idxs = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
        x = dataset['train_input'][idxs]
        y = dataset['train_label'][idxs]

        torch.cuda.synchronize() if device == 'cuda' else None
        t0 = time.time()
        pred = model(x)
        torch.cuda.synchronize() if device == 'cuda' else None
        t1 = time.time()
        forward_times.append((t1 - t0) * 1000)

        optimizer.zero_grad()
        torch.cuda.synchronize() if device == 'cuda' else None
        t2 = time.time()
        loss = loss_fn(pred, y)
        loss.backward()
        torch.cuda.synchronize() if device == 'cuda' else None
        t3 = time.time()
        backward_times.append((t3 - t2) * 1000)

    return {
        'forward_ms': np.mean(forward_times),
        'backward_ms': np.mean(backward_times)
    }

# --- Classification evaluation ---
def evaluate_model(model, dataset, device):
    model.eval()
    with torch.no_grad():
        logits = model(dataset['test_input'])
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_true = dataset['test_label'].cpu().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# --- Count parameters ---
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# --- Run all models ---
def run_all_models(args):
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    print(f"Embedding type: {args.embed}")
    print(f"Tune embedding: {args.tune}")
    if args.embed == 'mbert':
        print(f"Transformer model: {args.transformer}")
    
    print("\nLoading dataset...")
    dataset, num_classes = load_myanmar_news(
        device=device, 
        embed_type=args.embed,
        transformer_name=args.transformer
    )
    input_dim = dataset['train_input'].shape[1]
    
    model_constructors = {
        'mlp': lambda: MLP(input_dim, args.hidden_size, num_classes),
        'fasterkan': lambda: FasterKAN(
            layers_hidden=[input_dim, args.hidden_size, num_classes],
            grid_min=-2.0, grid_max=2.0, num_grids=8, exponent=2,
            train_grid=True, train_inv_denominator=True
        ),
        'fastkanorg': lambda: FastKANORG(
            layers_hidden=[input_dim, args.hidden_size, num_classes],
            grid_min=-2.0, grid_max=2.0, num_grids=8
        ),
        'efficientkan': lambda: KAN(
            layers_hidden=[input_dim, args.hidden_size, num_classes],
            grid_size=5, spline_order=3
        ),
        'kalnet': lambda: KAL_Net(
            layers_hidden=[input_dim, args.hidden_size, num_classes],
            polynomial_order=3, base_activation=nn.SiLU
        )
    }

    results = {}
    
    # Filter models if specific model is requested
    if args.model != 'all':
        if args.model not in model_constructors:
            print(f"Error: Unknown model '{args.model}'. Available: {list(model_constructors.keys())}")
            return
        model_constructors = {args.model: model_constructors[args.model]}
    
    for model_name, constructor in model_constructors.items():
        print(f"\n{'='*50}")
        print(f"Running model: {model_name.upper()}")
        print(f"{'='*50}")
        
        try:
            model = constructor().to(device)
            total_params, trainable_params = count_params(model)
            print(f"Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
            
            # Train the model
            print("Training...")
            train_start = time.time()
            train_model(model, dataset, device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
            train_time = time.time() - train_start
            
            # Evaluate performance
            print("Evaluating...")
            metrics = evaluate_model(model, dataset, device)
            
            # Speed benchmark
            print("Speed benchmark...")
            speed_metrics = benchmark_speed(model, dataset, device, batch_size=args.batch_size)
            
            # Combine results
            result = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'train_time_sec': train_time,
                **metrics,
                **speed_metrics
            }
            
            results[model_name] = result
            
            print(f"Results: {result}")
            
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Embedding: {args.embed}, Tune: {args.tune}")
    print(f"{'Model':<12} {'Accuracy':<8} {'F1':<6} {'Params':<8} {'Train(s)':<8} {'Fwd(ms)':<7} {'Bwd(ms)':<7}")
    print("-" * 60)
    
    for name, result in results.items():
        if 'error' not in result:
            print(f"{name:<12} {result['accuracy']:<8.3f} {result['f1']:<6.3f} "
                  f"{result['total_params']:<8,} {result['train_time_sec']:<8.1f} "
                  f"{result['forward_ms']:<7.2f} {result['backward_ms']:<7.2f}")
        else:
            print(f"{name:<12} ERROR: {result['error']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark KAN models with different embeddings')
    
    # Embedding options
    parser.add_argument('--embed', type=str, default='tfidf',
                        choices=['tfidf', 'random', 'fasttext', 'mbert'],
                        help='Embedding type to use')
    parser.add_argument('--tune', action='store_true', default=False,
                        help='Fine-tune embedding layer (only for random/mbert)')
    parser.add_argument('--transformer', type=str, default='google-bert/bert-base-multilingual-cased',
                        help='Transformer model name for mbert embedding')
    
    # Model options
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'mlp', 'fasterkan', 'fastkanorg', 'efficientkan', 'kalnet'],
                        help='Which model to run')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Hidden layer size')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    
    args = parser.parse_args()
    run_all_models(args)