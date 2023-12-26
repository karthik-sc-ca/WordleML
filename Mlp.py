from typing import Set, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, vocab_size: int, context_size: int, embedding_dim: int, tanh_dim: int):
        """
        Initializes an MLP-based language model.

        Args:
        - vocab_size (int): Size of the vocabulary
        - context_size (int): Context size for the language model
        - embedding_dim (int): Dimension of the embeddings
        - tanh_dim (int): Dimension of the hidden layer (Tanh layer)

        Initializes the model architecture with specified parameters.
        """
        super(MLP, self).__init__()
        self.context_size = context_size
        self.C = nn.Embedding(vocab_size, embedding_dim)
        self.W1 = nn.Linear(context_size * embedding_dim, tanh_dim)
        self.b1 = nn.Parameter(torch.randn(tanh_dim))
        self.W2 = nn.Linear(tanh_dim, vocab_size)
        self.b2 = nn.Parameter(torch.randn(vocab_size))

        self.stoi = {}
        self.itos = {}
        self.words = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the MLP model.

        Args:
        - x (torch.Tensor): Input tensor

        Returns:
        - torch.Tensor: Output logits from the model
        """
        emb = self.C(x)
        emb = emb.view(-1, emb.size(1) * emb.size(2))
        h = torch.tanh(self.W1(emb) + self.b1)
        logits = self.W2(h) + self.b2
        return logits

    def read_words(self, file_name: str) -> None:
        """
        Reads words from a file, processes them, and builds character mappings.

        Args:
        - file_name (str): Name of the file containing words
        """
        self.words = open(file_name, "r").read().lower().splitlines()
        chars = list(sorted(set(''.join(self.words))))
        self.stoi = {s: i + 1 for i, s in enumerate(chars)}
        self.stoi['.'] = 0
        self.itos = {i: s for s, i in self.stoi.items()}

    def build_datasets(self) -> None:
        """
        Builds training datasets (X and Y) from the processed words.
        """
        self.X, self.Y = [], []
        for w in self.words:
            context = [0] * self.context_size
            for ch in w + '.':
                ix = self.stoi[ch]
                self.X.append(context)
                self.Y.append(ix)
                context = context[1:] + [ix]
        self.X = torch.tensor(self.X)
        self.Y = torch.tensor(self.Y)

    def train(self, batch_size: int, num_epochs: int) -> None:
        """
        Trains the MLP model using stochastic gradient descent.

        Args:
        - batch_size (int): Size of each training batch
        - num_epochs (int): Number of training epochs
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        for epoch in range(num_epochs):
            for i in range(0, len(self.X), batch_size):
                optimizer.zero_grad()
                inputs = self.X[i:i + batch_size]
                targets = self.Y[i:i + batch_size]
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def evaluate(self) -> float:
        """
        Evaluates the model's performance.

        Returns:
        - float: Loss value obtained during evaluation
        """
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            outputs = self.forward(self.X)
            loss = criterion(outputs, self.Y)
        return loss.item()

    def sample(self) -> str:
        """
        Generates text based on the trained model until encountering the end token.

        Returns:
        - str: Generated text
        """
        generated_str = ""
        context = [0] * self.context_size
        while True:
            inputs = torch.tensor(context).unsqueeze(0)
            logits = self.forward(inputs)
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            generated_str += self.itos[ix]
        return generated_str

    def sample_length(self, length: int) -> str:
        """
        Generates text with a specified length.

        Args:
        - length (int): Desired length of the generated text

        Returns:
        - str: Generated text with the specified length
        """
        generated_str = ""
        context = [0] * self.context_size
        for _ in range(length):
            inputs = torch.tensor(context).unsqueeze(0)
            ix = 0
            while ix == 0:
                logits = self.forward(inputs)
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            generated_str += self.itos[ix]
        return generated_str

    def sample_length_with_constraints(self, length: int, in_position_chars: Dict[int, str],
                                       out_position_chars: Dict[int, Set[str]],
                                       excluded_chars: Set[str],
                                       previous_samples: List[str] = []) -> str:
        """
                Generates text with a specific length while meeting certain constraints.

                Args:
                - length (int): Desired length of the generated text
                - in_position_chars (Dict[int, str]): Characters to be placed at specific positions
                - out_position_chars (Dict[int, List[str]]): Characters not to be placed at specific positions
                - excluded_chars (Set[str]): Characters to be excluded from the generated text
                - previous_samples (List[str], optional): Previously generated samples
        """

        flag = True
        while flag:
            generated_str = ""
            context = [0] * self.context_size
            for i in range(length):
                if i in in_position_chars.keys():
                    ix = self.stoi[in_position_chars[i]]
                else:
                    inputs = torch.tensor(context).unsqueeze(0)
                    ix = 0
                    logits = self.forward(inputs)
                    probs = F.softmax(logits, dim=1)
                    while ix == 0 or self.itos[ix] in excluded_chars or (
                            i in out_position_chars.keys() and self.itos[ix] in out_position_chars[i]):
                        ix = torch.multinomial(probs, num_samples=1).item()
                context = context[1:] + [ix]
                generated_str += self.itos[ix]

            if generated_str in previous_samples:
                continue

            check_string = ''.join(['.' if i in in_position_chars else ch for i, ch in enumerate(generated_str)])

            flag = any(not set(chars).issubset(set(check_string)) for chars in out_position_chars.values())
            if flag:
                previous_samples.append(generated_str)

        return generated_str


if __name__ == '__main__':
    model = MLP(vocab_size=27, context_size=4, embedding_dim=10, tanh_dim=200)
    model.read_words("names.txt")
    model.build_datasets()
    model.train(batch_size=32, num_epochs=20)
    dev_loss = model.evaluate()
    print(f"Validation Loss: {dev_loss}")

    print(f"Sampling 10 Indian Names:")
    for _ in range(10):
        print(model.sample())

    english_model = MLP(vocab_size=27, context_size=4, embedding_dim=10, tanh_dim=200)
    english_model.read_words("english.txt")
    english_model.build_datasets()
    english_model.train(batch_size=32, num_epochs=10)
    dev_loss = english_model.evaluate()
    print(f"Validation Loss: {dev_loss}")

    print(f"Sampling 10 5 letter English Words:")
    for _ in range(10):
        print(english_model.sample_length(length=5))

    print(f"Sampling 10 5 letters English Words starting with 'ap' , having 'e' as last char, not having 'd' and "
          f"having 'l' but not as third character:")
    for _ in range(10):
        print(english_model.sample_length_with_constraints(length=5, in_position_chars={0: 'a', 1: 'p', 4: 'e'},
                                                           out_position_chars={2: {'l'}}, excluded_chars={'d'}))
    print("done")
