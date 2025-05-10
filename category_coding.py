import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import KFold
from model_RDM_extraction import process_video
from torch.utils.data import Subset
import copy


# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)


class VideoDataset(Dataset):
    def __init__(self, model_name):
        self.model_name = model_name
        self.samples = []

        # Create list of files and extract labels
        for fname in sorted(os.listdir("generated_videos")):
            if fname.startswith("processed_"):
                label_str = fname[len("processed_") :].split("_")[0]  # e.g. "ball"
                self.samples.append(
                    (os.path.join("generated_videos", fname), label_str)
                )

        # Map string labels to integers
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


def get_feature(model_name, device):
    model = torch.hub.load("facebookresearch/pytorchvideo", model_name, pretrained=True)
    backbone = torch.nn.Sequential(*list(model.blocks[:-1]))
    backbone = backbone.to(device).eval()

    dataset = VideoDataset(model_name)
    features = []
    labels = []

    with torch.no_grad():
        for video, label in tqdm(dataset):
            if isinstance(video, list):
                video = [v.unsqueeze(0).to(device) for v in video]
            else:
                video = video.unsqueeze(0).to(device)
            feat = backbone(video)
            feat = feat.view(feat.size(0), -1)
            features.append(feat.squeeze(0).cpu())
            labels.append(label)

    torch.save((features, labels), "cached_features.pt")

    return (features, labels)


if __name__ == "__main__":
    model_name = "slowfast_r50"
    device = "mps"

    if os.path.exists("cached_features.pt"):
        features, labels = torch.load("cached_features.pt", weights_only=True)
    else:
        features, labels = get_feature(model_name, device)

    features = torch.stack(features)  # shape: [N, C]
    labels = torch.tensor(labels)

    dataset = TensorDataset(features, labels)

    # Parameters
    k = 6
    max_epochs = 100
    batch_size = 36
    patience = 10
    learning_rate = 1e-3
    criterion = torch.nn.CrossEntropyLoss()

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
        print(f"\n=== Fold {fold + 1} ===")

        train_dataset = TensorDataset(features[train_idx], labels[train_idx])
        val_dataset = TensorDataset(features[val_idx], labels[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1)

        classifier = torch.nn.Sequential(
            torch.nn.Linear(features.size(1), 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 6),
        ).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

        best_val_acc = 0.0
        epochs_no_improve = 0
        best_model_path = f"trained_models/best_model_fold{fold + 1}.pt"

        for epoch in range(max_epochs):
            classifier.train()
            total_loss = 0
            correct = 0

            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = classifier(x_batch.to(device))
                loss = criterion(outputs, y_batch.to(device))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch.to(device)).sum().item()

            train_acc = correct / len(train_dataset)

            # Validation
            classifier.eval()
            val_correct = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    outputs = classifier(x_batch.to(device))
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == y_batch.to(device)).item()

            val_acc = val_correct / len(val_dataset)
            print(
                f"Epoch {epoch + 1:02d}: Train Acc = {train_acc:.2%}, Val Acc = {val_acc:.2%}"
            )

            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(classifier.state_dict(), best_model_path)
                print(f"🧠 New best model saved for fold {fold + 1}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch + 1}")
                    break

        # Load and evaluate the best model
        classifier.load_state_dict(torch.load(best_model_path, weights_only=True))
        classifier.eval()
        final_correct = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = classifier(x_batch.to(device))
                preds = outputs.argmax(dim=1)
                final_correct += (preds == y_batch.to(device)).item()

        acc = final_correct / len(val_dataset)
        print(f"✅ Best Validation Accuracy for Fold {fold + 1}: {acc:.2%}")
        accuracies.append(acc)

    # Summary
    mean_acc = sum(accuracies) / len(accuracies)
    print(f"\n🎯 Final K-Fold Accuracy: {mean_acc:.2%}")