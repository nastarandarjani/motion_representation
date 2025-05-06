import os
import torch
from tqdm import tqdm
from model_RDM_extraction import process_video
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneOut
from torch.utils.data import Subset


# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)


class VideoDataset(Dataset):
    def __init__(self, model_name):
        self.model_name = model_name
        self.samples = []

        # Create list of files and extract labels
        for fname in sorted(os.listdir("stimuli")):
            if fname.startswith("processed_"):
                label_str = fname[len("processed_") :].split("_")[0]  # e.g. "ball"
                self.samples.append((os.path.join("stimuli", fname), label_str))

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


if __name__ == "__main__":
    # Specify the desired model name ('slowfast_r50', 'R2PLUS1D', 'x3d_m', 'slow_r50', 'res_r50' or 'dorsalnet')
    model_name = "slowfast_r50"
    dataset = "k400"  # k400, ssv2
    batch_size = int(36 / 2)
    num_classes = 6
    num_epochs = 10
    lr = 1e-4
    device = "cpu"

    full_dataset = VideoDataset(model_name)
    all_indices = list(range(len(full_dataset)))
    loo = LeaveOneOut()
    total_correct = 0

    for fold, (train_idx, test_idx) in enumerate(loo.split(full_dataset)):
        print(f"\n--- Fold {fold + 1}/{len(full_dataset)} ---")

        train_loader = DataLoader(
            Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            Subset(full_dataset, test_idx), batch_size=1, shuffle=False
        )

        # Load pretrained model
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", model_name, pretrained=True
        )
        model.blocks[-1].proj = torch.nn.Linear(
            model.blocks[-1].proj.in_features, num_classes
        )

        # Freeze all layers except final classifier
        for param in model.parameters():
            param.requires_grad = False
        for param in model.blocks[-1].proj.parameters():
            param.requires_grad = True

        model = model.to(device)
        optimizer = torch.optim.Adam(model.blocks[-1].proj.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        # === TRAIN ===
        model.train()
        for epoch in tqdm(range(num_epochs)):
            for videos, labels in train_loader:
                if "slowfast" in model_name:
                    inputs = [v.to(device) for v in videos]
                else:
                    inputs = torch.stack(videos).to(device)

                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # === Save Trained Model ===
        save_path = f"trained_models/slowfast_fold{fold + 1}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved model for fold {fold + 1} to {save_path}")

        # === TEST ===
        model.eval()
        with torch.no_grad():
            for videos, labels in test_loader:
                if "slowfast" in model_name:
                    inputs = [v.to(device) for v in videos]
                else:
                    inputs = videos[0].unsqueeze(0).to(device)

                labels = labels.to(device)
                outputs = model(inputs)
                pred = outputs.argmax(dim=1)
                correct = (pred == labels).item()
                total_correct += correct

                print(
                    f"Test label: {labels.item()}, Predicted: {pred.item()}, Correct: {correct}"
                )

    # === FINAL RESULT ===
    print(
        f"\nFinal LOOCV Accuracy: {total_correct}/{len(full_dataset)} = {100 * total_correct / len(full_dataset):.2f}%"
    )
