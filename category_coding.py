import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from model_RDM_extraction import process_video
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)

def set_seed(seed=42):
    """Sets the seed for all random processes to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True  # Ensures reproducibility on GPUs
    torch.backends.cudnn.benchmark = False


set_seed(42)  # Set seed for reproducibility


def plot_confusion(all_targets, all_preds, classnames, title):
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots()
    cm = cm[[1, 2, 4, 5, 3, 0], :]
    cm = cm[:, [1, 2, 4, 5, 3, 0]]
    cax = ax.matshow(cm, cmap="Blues")
    plt.title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(classnames)))
    ax.set_yticks(np.arange(len(classnames)))
    ax.set_xticklabels(classnames[[1, 2, 4, 5, 3, 0]])
    ax.set_yticklabels(classnames[[1, 2, 4, 5, 3, 0]])

    # Label cells with counts
    for i in range(len(classnames)):
        for j in range(len(classnames)):
            ax.text(j, i, str(cm[i, j]), va="center", ha="center", color="black")

    plt.tight_layout()
    plt.show()
    plt.close()


class VideoDataset(Dataset):
    def __init__(self, model_name, path):
        if model_name == "slow_r50":
            self.model_name = "slowfast_r50"
        else:
            self.model_name = model_name

        self.samples = []
        self.path = path

        for fname in sorted(os.listdir(self.path)):
            if fname.startswith("processed_"):
                label_str = fname[len("processed_") :].split("_")[0]
                self.samples.append((os.path.join(self.path, fname), label_str))

        self.label_to_idx = {
            label: idx
            for idx, label in enumerate(sorted(set(label for _, label in self.samples)))
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label_str = self.samples[idx]
        video_tensor = process_video(self.model_name, video_path)
        label = self.label_to_idx[label_str]
        return video_tensor, label

    def get_labels(self):
        return [label_str for _, label_str in self.samples]

def get_feature(model_name, dataset, device):
    if model_name == "res_r50":
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", "slow_r50", pretrained=True
        )
        custom_pool = torch.nn.AvgPool3d(
            kernel_size=(8, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0)
        )
        backbone = torch.nn.Sequential(*list(model.blocks[:-1]), custom_pool).to(device)

    else:
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True
        )
        backbone = torch.nn.Sequential(*list(model.blocks[:-1])).to(device)

    backbone.eval()

    if model_name == "slow_r50":
        for module_name, module in backbone.named_modules():
            if "multipathway_fusion" in module_name:
                for param in module.parameters():
                    param.requires_grad = False
                    if isinstance(module, torch.nn.Conv3d):
                        module.weight.data.fill_(0)
                    elif isinstance(module, torch.nn.BatchNorm3d):
                        module.weight.data.fill_(0)
                        module.bias.data.fill_(0)

    features, labels = [], []

    with torch.no_grad():
        for video, label in tqdm(dataset):
            if isinstance(video, list):
                video = [v.unsqueeze(0).to(device) for v in video]
            else:
                video = video.unsqueeze(0).to(device)
            feat = backbone(video).view(1, -1)
            features.append(feat.squeeze(0).cpu())
            labels.append(label)

    torch.save((features, labels), f"trained_models/{path}_{model_name}.pt")
    return features, labels


if __name__ == "__main__":
    model_name = "slow_r50"
    device = "cpu"  # or "cuda" if using GPU

    batch_size = 36
    if not os.path.exists(f"trained_models/{model_name}/best_model_fold1.pt"):
        path = "generated_videos"

        dataset = VideoDataset(model_name, path)

        if os.path.exists(f"trained_models/{path}_{model_name}.pt"):
            features, labels = torch.load(
                f"trained_models/{path}_{model_name}.pt", weights_only=True
            )
        else:
            features, labels = get_feature(model_name, dataset, device)

        classnames = np.unique(dataset.get_labels())
        features = torch.stack(features)
        labels = torch.tensor(labels)
        num_classes = len(classnames)

        k = 5
        max_epochs = 100
        patience = 10
        learning_rate = 1e-3
        criterion = torch.nn.CrossEntropyLoss()

        skf = StratifiedKFold(n_splits=k, shuffle=True)

        for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
            X_train, y_train = features[train_idx], labels[train_idx]
            X_val, y_val = features[val_idx], labels[val_idx]

            train_loader = DataLoader(
                TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

            classifier = torch.nn.Sequential(
                torch.nn.Linear(features.size(1), 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, num_classes),
            ).to(device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

            best_val_acc = 0.0
            epochs_no_improve = 0
            best_state_dict = None

            for epoch in range(max_epochs):
                classifier.train()
                correct = 0
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    outputs = classifier(xb.to(device))
                    loss = criterion(outputs, yb.to(device))
                    loss.backward()
                    optimizer.step()
                    correct += (outputs.argmax(1) == yb.to(device)).sum().item()

                train_acc = correct / len(train_loader.dataset)

                # Validation
                classifier.eval()
                correct = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        preds = classifier(xb.to(device)).argmax(dim=1)
                        correct += (preds == yb.to(device)).sum().item()
                val_acc = correct / len(val_loader.dataset)
                print(
                    f"Epoch {epoch + 1:02d}: Train Acc = {train_acc:.2%}, Val Acc = {val_acc:.2%}"
                )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state_dict = classifier.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break

            print(f"Fold {fold + 1} Best Val Acc: {best_val_acc:.2%}")
            os.makedirs(f"trained_models/{model_name}/", exist_ok=True)
            torch.save(
                best_state_dict, f"trained_models/{model_name}/best_model_fold{fold}.pt"
            )

    path = "stimuli"

    dataset = VideoDataset(model_name, path)

    if os.path.exists(f"trained_models/{path}_{model_name}.pt"):
        features, labels = torch.load(
            f"trained_models/{path}_{model_name}.pt", weights_only=True
        )
    else:
        features, labels = get_feature(model_name, dataset, device)

    features = torch.stack(features)
    labels = torch.tensor(labels)
    classnames = np.unique(dataset.get_labels())
    num_classes = len(classnames)

    classifier = torch.nn.Sequential(
        torch.nn.Linear(features.size(1), 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, num_classes),
    ).to(device)

    CM = np.zeros((5, num_classes, num_classes))
    for folds in range(5):
        weights = torch.load(
            f"trained_models/{model_name}/best_model_fold{folds}.pt",
            weights_only=True,
        )
        classifier.load_state_dict(weights)
        classifier.eval()

        test_loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size)
        correct = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for xb, yb in test_loader:
                preds = classifier(xb.to(device)).argmax(dim=1)
                correct += (preds == yb.to(device)).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(yb.cpu().numpy())

        test_acc = correct / len(test_loader.dataset)
        print(f"Fold {folds + 1} Test Accuracy: {test_acc:.2%}")
        # plot_confusion(
        #     all_targets,
        #     all_preds,
        #     classnames,
        #     f"Fold {folds + 1} - Test Confusion Matrix",
        # )
        cm = confusion_matrix(all_targets, all_preds)
        CM[folds, :, :] = cm

    with open(f"result/confusions/{model_name}.pkl", "wb") as File:
        pickle.dump(CM, File)