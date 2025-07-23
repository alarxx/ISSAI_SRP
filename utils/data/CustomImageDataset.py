import os
from PIL import Image
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

# Like ImageFolder
# dataset = datasets.ImageFolder(root, transform)

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir # path
        self.transform = transform

        self.image_paths = []
        self.labels = [] # хранит в виде индексов классов
        self.classes = [] # idx_to_class
        self.class_to_idx = {} # isn't it redundant?

        # assign index to each class folder
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path): # is it necessary?
                self.classes.append(class_name)
                self.class_to_idx[class_name] = idx # ???
                for fname in os.listdir(class_path):
                    if fname.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, fname))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def train_test_split(self, test_size=0.25, random_state=42):
        indices = list(range(len(self)))
        train_idx, test_idx = train_test_split(indices,
                                               test_size=test_size,
                                               shuffle=True,
                                               random_state=random_state,
                                               stratify=self.labels)
        return Subset(self, train_idx), Subset(self, test_idx)
