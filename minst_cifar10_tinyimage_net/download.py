import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm


# ============================================================
# 1. Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# ============================================================
# 2. Transforms
#    Convert MNIST to RGB 32x32 so both datasets have same shape
# ============================================================

mnist_train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

mnist_test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

cifar_train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761)
    )
])

cifar_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761)
    )
])


# ============================================================
# 3. Dataset Wrappers
# ============================================================

class MNISTProxyVLMDataset(Dataset):
    def __init__(self, train=True, root="./data"):
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=mnist_train_transform if train else mnist_test_transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # MNIST labels stay 0-9
        unified_label = label

        return image, unified_label


class CIFAR100ProxyVLMDataset(Dataset):
    def __init__(self, train=True, root="./data"):
        self.dataset = datasets.CIFAR100(
            root=root,
            train=train,
            download=True,
            transform=cifar_train_transform if train else cifar_test_transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # CIFAR-100 labels are shifted by 10
        # MNIST: 0-9
        # CIFAR-100: 10-109
        unified_label = label + 10

        return image, unified_label


# ============================================================
# 4. Load Datasets
# ============================================================

mnist_train = MNISTProxyVLMDataset(train=True)
mnist_test = MNISTProxyVLMDataset(train=False)

cifar_train = CIFAR100ProxyVLMDataset(train=True)
cifar_test = CIFAR100ProxyVLMDataset(train=False)

train_dataset = ConcatDataset([mnist_train, cifar_train])
test_dataset = ConcatDataset([mnist_test, cifar_test])

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True if device == "cuda" else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True if device == "cuda" else False
)


# ============================================================
# 5. Build Unified Class Prompts
# ============================================================

cifar_class_names = datasets.CIFAR100(
    root="./data",
    train=True,
    download=False
).classes

mnist_prompts = [
    f"a photo of digit {i}"
    for i in range(10)
]

cifar_prompts = [
    f"a photo of a {name.replace('_', ' ')}"
    for name in cifar_class_names
]

all_prompts = mnist_prompts + cifar_prompts

all_class_names = (
    [f"digit_{i}" for i in range(10)]
    + cifar_class_names
)

num_classes = len(all_prompts)

print("Total classes:", num_classes)
print("First 10 prompts:", all_prompts[:10])
print("Example CIFAR prompts:", all_prompts[10:20])


# ============================================================
# 6. Text Processor
# ============================================================

class ProxyVLMTextProcessor:
    def __init__(self, prompts):
        self.prompts = prompts

        vocab = {
            "<pad>": 0,
            "<unk>": 1
        }

        for prompt in prompts:
            for word in prompt.lower().split():
                if word not in vocab:
                    vocab[word] = len(vocab)

        self.token_to_id = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)
        self.max_len = max(len(prompt.split()) for prompt in prompts)

    def encode_prompt(self, prompt):
        words = prompt.lower().split()

        ids = [
            self.token_to_id.get(word, self.token_to_id["<unk>"])
            for word in words
        ]

        while len(ids) < self.max_len:
            ids.append(self.token_to_id["<pad>"])

        return torch.tensor(ids, dtype=torch.long)

    def build_all_class_token_ids(self):
        encoded = [
            self.encode_prompt(prompt)
            for prompt in self.prompts
        ]

        return torch.stack(encoded, dim=0)


text_processor = ProxyVLMTextProcessor(all_prompts)

class_token_ids = text_processor.build_all_class_token_ids().to(device)

print("Vocabulary size:", text_processor.vocab_size)
print("Max prompt length:", text_processor.max_len)


# ============================================================
# 7. Shared Image Encoder
# ============================================================

class SharedImageEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 x 16 x 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 x 8 x 8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 128 x 4 x 4
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim)
        )

    def forward(self, images):
        x = self.features(images)
        x = self.proj(x)
        x = F.normalize(x, dim=-1)
        return x


# ============================================================
# 8. Tiny Text Encoder
# ============================================================

class TinyTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, text_hidden_dim=128, pad_id=0):
        super().__init__()

        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(
            vocab_size,
            text_hidden_dim,
            padding_idx=pad_id
        )

        self.proj = nn.Sequential(
            nn.Linear(text_hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)

        mask = (token_ids != self.pad_id).unsqueeze(-1).float()
        x = x * mask

        summed = x.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)

        x = summed / counts

        x = self.proj(x)
        x = F.normalize(x, dim=-1)

        return x


# ============================================================
# 9. Proxy VLM Model
# ============================================================

class MultiDatasetProxyVLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, text_hidden_dim=128, pad_id=0):
        super().__init__()

        self.image_encoder = SharedImageEncoder(embed_dim=embed_dim)

        self.text_encoder = TinyTextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            text_hidden_dim=text_hidden_dim,
            pad_id=pad_id
        )

        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, images):
        return self.image_encoder(images)

    def encode_text(self, token_ids):
        return self.text_encoder(token_ids)

    def forward(self, images, class_token_ids):
        image_features = self.encode_image(images)
        text_features = self.encode_text(class_token_ids)

        scale = self.logit_scale.exp().clamp(max=100)

        logits = scale * image_features @ text_features.T

        return logits
