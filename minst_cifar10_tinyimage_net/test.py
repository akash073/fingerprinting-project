



# ============================================================
# Raspberry Pi Inference/Test Code
# Proxy VLM: MNIST + CIFAR-100 + TinyImageNet-200
# CPU-only, low-memory version
# ============================================================

import os
import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm


# ============================================================
# 1. Raspberry Pi CPU Settings
# ============================================================

device = "cpu"

# Raspberry Pi 4/5 usually works well with 4 threads
torch.set_num_threads(4)

print("Using device:", device)
print("Torch threads:", torch.get_num_threads())


# ============================================================
# 2. Check Model Class Exists
# ============================================================

try:
    MultiDatasetProxyVLM
except NameError:
    raise NameError(
        "MultiDatasetProxyVLM is not defined. "
        "Please run/import the model-definition code first."
    )


# ============================================================
# 3. Load Saved Model
# ============================================================

checkpoint_path = "best_proxy_vlm_mnist_cifar100_tinyimagenet.pth"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}\n"
        "Place the .pth file in the same folder as this script."
    )

checkpoint = torch.load(checkpoint_path, map_location=device)

model = MultiDatasetProxyVLM(
    vocab_size=checkpoint["vocab_size"],
    embed_dim=checkpoint["embed_dim"],
    text_hidden_dim=checkpoint["text_hidden_dim"],
    pad_id=checkpoint["token_to_id"]["<pad>"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

class_token_ids = checkpoint["class_token_ids"].to(device).long()
all_prompts = checkpoint["all_prompts"]
all_class_names = checkpoint["all_class_names"]

print("Model loaded successfully.")
print("Saved epoch:", checkpoint["epoch"])
print("Best saved accuracy:", checkpoint.get("best_total_acc", "N/A"))
print("Total classes:", len(all_class_names))


# ============================================================
# 4. Precompute Text Features Once
# Important for Raspberry Pi speed
# ============================================================

with torch.no_grad():
    text_features = model.encode_text(class_token_ids)
    text_features = text_features.to(device)
    logit_scale = model.logit_scale.exp().clamp(max=100)

print("Text features precomputed.")
print("Text feature shape:", text_features.shape)


# ============================================================
# 5. Test Transforms
# Must match training test transforms
# ============================================================

IMG_SIZE = 64

mnist_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

cifar_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761)
    )
])

tiny_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4802, 0.4481, 0.3975),
        std=(0.2302, 0.2265, 0.2262)
    )
])


# ============================================================
# 6. Test Dataset Wrappers
# Unified label mapping:
# MNIST: 0-9
# CIFAR-100: 10-109
# TinyImageNet-200: 110-309
# ============================================================

class MNISTTestDataset(Dataset):
    def __init__(self, root="./data"):
        self.dataset = datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=mnist_test_transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        unified_label = label
        return image, unified_label


class CIFAR100TestDataset(Dataset):
    def __init__(self, root="./data"):
        self.dataset = datasets.CIFAR100(
            root=root,
            train=False,
            download=True,
            transform=cifar_test_transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        unified_label = label + 10
        return image, unified_label


class TinyImageNetTestDataset(Dataset):
    def __init__(self, root="./data"):
        val_dir = os.path.join(root, "tiny-imagenet-200", "val")

        if not os.path.exists(val_dir):
            raise FileNotFoundError(
                f"TinyImageNet validation folder not found: {val_dir}\n"
                "Expected structure:\n"
                "./data/tiny-imagenet-200/val/class_id/image.png"
            )

        self.dataset = datasets.ImageFolder(
            root=val_dir,
            transform=tiny_test_transform
        )

        if len(self.dataset.classes) != 200:
            print(
                f"Warning: TinyImageNet val has {len(self.dataset.classes)} folders. "
                "Expected 200 folders."
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        unified_label = label + 110
        return image, unified_label


# ============================================================
# 7. Load Test Datasets
# ============================================================

mnist_test = MNISTTestDataset(root="./data")
cifar100_test = CIFAR100TestDataset(root="./data")
tinyimagenet_test = TinyImageNetTestDataset(root="./data")


# ============================================================
# Optional: Test Small Subset First
# Recommended on Raspberry Pi
# ============================================================

USE_SUBSET = False
SUBSET_SIZE = 500

if USE_SUBSET:
    mnist_test = Subset(mnist_test, range(min(SUBSET_SIZE, len(mnist_test))))
    cifar100_test = Subset(cifar100_test, range(min(SUBSET_SIZE, len(cifar100_test))))
    tinyimagenet_test = Subset(tinyimagenet_test, range(min(SUBSET_SIZE, len(tinyimagenet_test))))

    print(f"Using subset size: {SUBSET_SIZE} samples per dataset")
else:
    print("Using full test datasets")


# ============================================================
# 8. DataLoaders for Raspberry Pi
# ============================================================

batch_size = 8       # Use 4 if Raspberry Pi becomes slow or memory-heavy
num_workers = 0      # Important for Raspberry Pi
pin_memory = False   # CPU-only

mnist_loader = DataLoader(
    mnist_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)

cifar100_loader = DataLoader(
    cifar100_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)

tinyimagenet_loader = DataLoader(
    tinyimagenet_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)

print("MNIST test samples:", len(mnist_test))
print("CIFAR-100 test samples:", len(cifar100_test))
print("TinyImageNet-200 test samples:", len(tinyimagenet_test))


# ============================================================
# 9. Fast CPU Inference Function
# Uses precomputed text features
# ============================================================

def test_model_on_pi(
    model,
    loader,
    text_features,
    logit_scale,
    dataset_name,
    device="cpu"
):
    model.eval()

    correct = 0
    total = 0

    start_time = time.time()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Testing {dataset_name}"):
            images = images.to(device)
            labels = labels.to(device).long()

            # Only encode images during inference
            image_features = model.encode_image(images)

            # Compare image embeddings with precomputed text embeddings
            logits = logit_scale * image_features @ text_features.T

            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    end_time = time.time()
    elapsed_time = end_time - start_time

    accuracy = 100 * correct / total
    avg_time_per_image = elapsed_time / total

    print(f"\n{dataset_name} Results")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct / Total: {correct}/{total}")
    print(f"Total inference time: {elapsed_time:.2f} seconds")
    print(f"Average time per image: {avg_time_per_image:.4f} seconds")

    return accuracy, elapsed_time, avg_time_per_image


# ============================================================
# 10. Test Each Dataset Separately
# ============================================================

mnist_acc, mnist_time, mnist_tpi = test_model_on_pi(
    model=model,
    loader=mnist_loader,
    text_features=text_features,
    logit_scale=logit_scale,
    dataset_name="MNIST",
    device=device
)

cifar_acc, cifar_time, cifar_tpi = test_model_on_pi(
    model=model,
    loader=cifar100_loader,
    text_features=text_features,
    logit_scale=logit_scale,
    dataset_name="CIFAR-100",
    device=device
)

tiny_acc, tiny_time, tiny_tpi = test_model_on_pi(
    model=model,
    loader=tinyimagenet_loader,
    text_features=text_features,
    logit_scale=logit_scale,
    dataset_name="TinyImageNet-200",
    device=device
)


# ============================================================
# 11. Final Summary
# ============================================================

print("\n================ Raspberry Pi Final Test Results ================")
print(f"MNIST Accuracy:               {mnist_acc:.2f}%")
print(f"CIFAR-100 Accuracy:           {cifar_acc:.2f}%")
print(f"TinyImageNet-200 Accuracy:    {tiny_acc:.2f}%")
print()
print(f"MNIST avg time/image:         {mnist_tpi:.4f} sec")
print(f"CIFAR avg time/image:         {cifar_tpi:.4f} sec")
print(f"TinyImageNet avg time/image:  {tiny_tpi:.4f} sec")
print("=================================================================")