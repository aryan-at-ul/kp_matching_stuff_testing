import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as GeometricData
from torch_geometric.nn import GCNConv
import random
import os
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from torch_geometric.utils import to_dense_batch
from PIL import Image
import torchvision.models as models
from torch.nn import Linear as Lin, BatchNorm1d as BN
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import LayerNorm as LN

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

NUM_KEYPOINTS = 20
BATCH_SIZE = 2
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
SHIFT = (1, 1)
RESIZE = (224, 224)
SPARSITY_K = 1
CONSENSUS_ITERATIONS = 10
RANDOM_FUNCTIONS = 3
CORRESPONDENCE_THRESHOLD = None
LAMBDA_REG = 0.2

NUM_KEYPOINTS = 100

resnet = models.resnet18(pretrained=True)
resnet_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
resnet_feature_extractor = resnet_feature_extractor.to(device)
resnet_feature_extractor.eval()

class ShiftedTinyImageNet(Dataset):
    def __init__(self, root=os.environ.get('DATA_PATH', '/data/tiny-imagenet-200'), train=True, transform=None, shift=(1, 1), resize=(224, 224), subset_fraction=1):
        self.root = root
        self.train = train
        self.transform = transform
        self.shift = shift
        self.resize = resize
        self.data = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        if self.train:
            dataset_path = os.path.join(root, 'train')
            classes = os.listdir(dataset_path)
            for idx, cls in enumerate(classes):
                class_path = os.path.join(dataset_path, cls, 'images')
                for img_file in os.listdir(class_path):
                    self.data.append(os.path.join(class_path, img_file))
                    self.labels.append(cls)
                self.label_to_idx[cls] = idx
                self.idx_to_label[idx] = cls
        else:
            dataset_path = os.path.join(root, 'val')
            val_annotations_path = os.path.join(dataset_path, 'val_annotations.txt')
            with open(val_annotations_path, 'r') as file:
                val_annotations = file.readlines()
            for line in val_annotations:
                parts = line.split('\t')
                self.data.append(os.path.join(dataset_path, 'images', parts[0]))
                self.labels.append(parts[1].strip())
                if parts[1].strip() not in self.label_to_idx:
                    idx = len(self.label_to_idx)
                    self.label_to_idx[parts[1].strip()] = idx
                    self.idx_to_label[idx] = parts[1].strip()
        self.labels = [self.label_to_idx[label] for label in self.labels]
        total_samples = len(self.data)
        subset_size = int(total_samples * subset_fraction)
        self.indices = random.sample(range(total_samples), subset_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path = self.data[actual_idx]
        label = self.labels[actual_idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.resize)
        img_np = np.array(img)
        keypoints_before = detect_keypoints(img_np)
        center = (img_np.shape[1] // 2, img_np.shape[0] // 2)
        angle = 10
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img_np = cv2.warpAffine(img_np, M, (img_np.shape[1], img_np.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        keypoints_after = apply_rotation_to_keypoints(keypoints_before, M)
        rotated_img_pil = Image.fromarray(rotated_img_np)
        if self.transform:
            img = self.transform(img)
            rotated_img = self.transform(rotated_img_pil)
        else:
            rotated_img = rotated_img_pil
        return img, rotated_img, label, keypoints_before, keypoints_after

NUM_KEYPOINTS = 100

def detect_keypoints(img_np):
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img_gray, None)
    keypoints = sorted(keypoints, key=lambda x: -x.response)[:NUM_KEYPOINTS]
    if len(keypoints) == 0:
        return np.empty((0, 2), dtype=np.float32)
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return points

def apply_rotation_to_keypoints(keypoints, M):
    if keypoints.size == 0:
        return keypoints
    keypoints_homogeneous = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
    keypoints_transformed = M @ keypoints_homogeneous.T
    keypoints_transformed = keypoints_transformed[:2].T
    return keypoints_transformed

def compute_dense_descriptors(img, keypoints, patch_size=32):
    if keypoints.size == 0:
        return np.zeros((0, 512), dtype=np.float32)
    descriptors = []
    half_patch = patch_size // 2
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        x1, y1 = max(x - half_patch, 0), max(y - half_patch, 0)
        x2, y2 = min(x + half_patch, img.shape[1]), min(y + half_patch, img.shape[0])
        patch = img[y1:y2, x1:x2, :]
        patch_tensor = transforms.ToTensor()(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            resnet_features = resnet_feature_extractor(patch_tensor).squeeze()
            resnet_features = resnet_features.cpu().numpy()
        descriptors.append(resnet_features)
    return np.array(descriptors)

from scipy.spatial.distance import cdist

def build_graph_with_descriptors(descriptors, keypoints):
    # trade off bw homophily and spatial features
    num_nodes = keypoints.shape[0]
    edge_index = []
    edge_attr = []
    if num_nodes == 0:
        return GeometricData(x=torch.empty((0, descriptors.shape[1]), dtype=torch.float),
                             edge_index=torch.empty((2, 0), dtype=torch.long),
                             edge_attr=torch.empty((0,), dtype=torch.float))
    k = 4
    spatial_distances = cdist(keypoints, keypoints, metric='euclidean')
    descriptor_distances = cdist(descriptors, descriptors, metric='cosine')
    alpha = 0.1
    beta = 1 - alpha
    combined_distances = alpha * spatial_distances + beta * descriptor_distances
    for i in range(num_nodes):
        neighbors = combined_distances[i].argsort()[1:k+1]
        for j in neighbors:
            edge_index.append([i, j])
            edge_attr.append(combined_distances[i, j])
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    node_features = torch.tensor(descriptors, dtype=torch.float)
    graph = GeometricData(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return graph

def sinkhorn_normalization(S, num_iters=10, epsilon=1e-9):
    S = torch.exp(S)
    for i in range(num_iters):
        S = S / (S.sum(dim=1, keepdim=True) + epsilon)
        S = S / (S.sum(dim=0, keepdim=True) + epsilon)
        if torch.isnan(S).any():
            print(f"Sinkhorn iteration {i}: NaN encountered in S.")
            S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    return S



def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out

def to_sparse(x, mask):
    return x[mask]

def to_dense(x, mask):
    out = x.new_zeros(tuple(mask.size()) + (x.size(-1), ))
    out[mask] = x
    return out



class RelConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(RelConv, self).__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = Lin(in_channels, out_channels, bias=False)
        self.lin2 = Lin(in_channels, out_channels, bias=False)
        self.root = Lin(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.root.reset_parameters()

    def forward(self, x, edge_index):
        self.flow = 'source_to_target'
        out1 = self.propagate(edge_index, x=self.lin1(x))
        self.flow = 'target_to_source'
        out2 = self.propagate(edge_index, x=self.lin2(x))
        return self.root(x) + out1 + out2

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class GNNForEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=False,
                 cat=True, lin=True, dropout=0.0):
        super(GNNForEmbedding, self).__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cat = cat
        self.lin = lin
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RelConv(in_channels, out_channels))
            self.batch_norms.append(LN(out_channels))
            in_channels = out_channels
        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels
        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels
        self.reset_parameters()

    def reset_parameters(self):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            conv.reset_parameters()
            batch_norm.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, *args):
        xs = [x]
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(xs[-1], edge_index)
            x = batch_norm(F.relu(x)) if self.batch_norm else F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = self.final(x) if self.lin else x
        return x

class GNNForConsensus(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNNForConsensus, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class UpdateCorrespondenceMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(UpdateCorrespondenceMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, di):
        out = self.fc1(di)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def compute_initial_correspondence(Hs, Ht, k=1):
    S_hat = torch.matmul(Hs, Ht.T)
    topk_values, topk_indices = torch.topk(S_hat, k, dim=1, largest=True, sorted=True)
    mask = torch.zeros_like(S_hat).scatter_(1, topk_indices, 1)
    S_hat_sparse = S_hat * mask
    S_hat_normalized = torch.nn.functional.softmax(S_hat_sparse, dim=1) * mask
    return S_hat_normalized

def visualize_correspondences(img1, img2, kp1, kp2, correspondence, epoch, sample_idx):
    img1 = cv2.resize(img1, (RESIZE[1], RESIZE[0]))
    img2 = cv2.resize(img2, (RESIZE[1], RESIZE[0]))
    kp1 = np.array(kp1, dtype=int)
    kp2 = np.array(kp2, dtype=int)
    combined_image = np.hstack((img1, img2))
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    for i, j in enumerate(correspondence):
        if j != -1 and i < len(kp1) and j < len(kp2):
            x1, y1 = kp1[i]
            x2, y2 = kp2[j]
            x2 += img1.shape[1]
            plt.plot(x1, y1, 'ro')
            plt.plot(x2, y2, 'bo')
            plt.plot([x1, x2], [y1, y2], 'g-', linewidth=1)
    plt.title(f"Epoch {epoch+1} Sample {sample_idx} Keypoint Correspondences")
    plt.show()

def compute_ground_truth_correspondences(kp1, kp2_transformed, threshold=None):
    if kp1.size == 0 or kp2_transformed.size == 0:
        return torch.full((kp1.shape[0],), -1, dtype=torch.long)
    cost_matrix = np.linalg.norm(kp1[:, np.newaxis, :] - kp2_transformed[np.newaxis, :, :], axis=2)
    penalty = 1e6
    for i in range(min(len(kp1), len(kp2_transformed))):
        cost_matrix[i, :i] = penalty
        cost_matrix[i, i+1:] = penalty
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    correspondence = torch.full((kp1.shape[0],), -1, dtype=torch.long)
    if threshold is not None:
        valid_matches = cost_matrix[row_ind, col_ind] <= threshold
    else:
        valid_matches = np.ones(len(row_ind), dtype=bool)
    correspondence[row_ind[valid_matches]] = torch.tensor(col_ind[valid_matches], dtype=torch.long)
    return correspondence

def generate_y(kp1, kp2_transformed):
    correspondence = compute_ground_truth_correspondences(kp1, kp2_transformed)
    return correspondence

class KeypointMatchingModel(nn.Module):
    def __init__(self, embedding_dim=512, consensus_dim=512, random_functions=3, hidden_dim=512):
        super(KeypointMatchingModel, self).__init__()
        self.gnn_embed = GNNForEmbedding(in_channels=512, out_channels=embedding_dim, num_layers=1, 
                                batch_norm=True, cat=True, lin=True, dropout=0.2)
        self.gnn_consensus = GNNForEmbedding(in_channels=512, out_channels=embedding_dim, num_layers=1, 
                                batch_norm=True, cat=True, lin=True, dropout=0.2)
        self.update_mlp = UpdateCorrespondenceMLP(input_dim=consensus_dim)
        self.num_steps = CONSENSUS_ITERATIONS
        self.detach = True

    def forward(self, x_s, edge_index_s, edge_attr_s, batch_s,
                x_t, edge_index_t, edge_attr_t, batch_t):
        Hs = self.gnn_embed(x_s, edge_index_s)
        Ht = self.gnn_embed(x_t, edge_index_t)
        Hs, Ht = (Hs.detach(), Ht.detach()) if self.detach else (Hs, Ht)
        Hs, s_mask = to_dense_batch(Hs, batch_s, fill_value=0)
        Ht, t_mask = to_dense_batch(Ht, batch_t, fill_value=0)
        assert Hs.size(0) == Ht.size(0), 'Encountered unequal batch sizes, graph loader messes up'
        (B, N_s, C_out), N_t = Hs.size(), Ht.size(1)
        R_in, R_out = self.gnn_consensus.in_channels, self.gnn_consensus.out_channels
        S_hat = Hs @ Ht.transpose(-1, -2)
        S_mask = s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t)
        S_0 = masked_softmax(S_hat, S_mask, dim=-1)[s_mask]
        for _ in range(self.num_steps):
            S = masked_softmax(S_hat, S_mask, dim=-1)
            R_s = torch.randn((B, N_s, R_in), dtype=Hs.dtype, device=Hs.device)
            R_t = S.transpose(-1, -2) @ R_s
            R_s, R_t = to_sparse(R_s, s_mask), to_sparse(R_t, t_mask)
            O_s = self.gnn_consensus(R_s, edge_index_s, edge_attr_s)
            O_t = self.gnn_consensus(R_t, edge_index_t, edge_attr_t)
            O_s, O_t = to_dense(O_s, s_mask), to_dense(O_t, t_mask)
            D = O_s.view(B, N_s, 1, R_out) - O_t.view(B, 1, N_t, R_out)
            S_hat = S_hat + self.update_mlp(D).squeeze(-1).masked_fill(~S_mask, 0)
        S_L = masked_softmax(S_hat, S_mask, dim=-1)[s_mask]
        return S_0, S_L

    def loss_fn(self, S, y):
        criterion = nn.CrossEntropyLoss()
        log_S = torch.log(S + 1e-8)
        loss = criterion(log_S, y)
        return loss

    def acc(self, S, y, reduction='sum'):
        S_np = S.cpu().detach().numpy()
        row_ind, col_ind = linear_sum_assignment(-S_np)
        pred = torch.full_like(y, -1)
        for r, c in zip(row_ind, col_ind):
            if r < S.size(0) and c < S.size(1):
                pred[r] = c
        correct = (pred == y).sum().float()
        if reduction == 'mean':
            return correct / y.size(0)
        return correct

def train_model(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    total_valid_pairs = 0
    for batch_idx, (img1, img2, label, keypoints_before, keypoints_after) in enumerate(train_loader):
        batch_size_current = img1.size(0)
        optimizer.zero_grad()
        batch_loss = 0.0
        valid_pairs = 0
        for i in range(batch_size_current):
            img1_np = (img1[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img2_np = (img2[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            kp1 = keypoints_before[i].cpu().numpy()
            kp2_transformed = keypoints_after[i].cpu().numpy()
            img_width, img_height = RESIZE
            valid_indices = (
                (kp2_transformed[:, 0] >= 0) & (kp2_transformed[:, 0] < img_width) &
                (kp2_transformed[:, 1] >= 0) & (kp2_transformed[:, 1] < img_height)
            )
            kp1 = kp1[valid_indices]
            kp2_transformed = kp2_transformed[valid_indices]
            descriptors1 = compute_dense_descriptors(img1_np, kp1)
            descriptors2 = compute_dense_descriptors(img2_np, kp2_transformed)
            graph1 = build_graph_with_descriptors(descriptors1, kp1)
            graph2 = build_graph_with_descriptors(descriptors2, kp2_transformed)
            if graph1.num_nodes == 0 or graph2.num_nodes == 0:
                print(f"[Epoch {epoch+1}, Batch {batch_idx+1}, Sample {i+1}] Skipped due to no keypoints after transformation.")
                continue
            graph1 = graph1.to(device)
            graph2 = graph2.to(device)
            S_0, S_L = model(graph1.x, graph1.edge_index, graph1.edge_attr, graph1.batch,
                            graph2.x, graph2.edge_index, graph2.edge_attr, graph2.batch)
            correspondence_source_to_target = generate_y(kp1, kp2_transformed).to(device)
            if torch.any(correspondence_source_to_target != -1):
                loss = model.loss_fn(S_L, correspondence_source_to_target)
                reg_loss = torch.mean((S_L.sum(dim=0) - 1) ** 2)
                loss = loss + LAMBDA_REG * reg_loss
                batch_loss += loss
                valid_pairs += 1
                print(f"[Epoch {epoch+1}, Batch {batch_idx+1}, Sample {i+1}] Loss: {loss.item():.4f}")
                S_final_np = S_L.cpu().detach().numpy()
                row_ind, col_ind = linear_sum_assignment(-S_final_np)
                predicted_correspondence = np.full(S_final_np.shape[0], -1, dtype=int)
                for r, c in zip(row_ind, col_ind):
                    if r < S_final_np.shape[0] and c < S_final_np.shape[1]:
                        predicted_correspondence[r] = c
                unique_mappings = len(set(predicted_correspondence)) - (1 if -1 in predicted_correspondence else 0)
                if unique_mappings != len(predicted_correspondence[predicted_correspondence != -1]):
                    print(f"[Epoch {epoch+1}, Batch {batch_idx+1}, Sample {i+1}] Warning: One-to-one correspondence violated.")


                if batch_idx % 10 == 0:  # Visualize only in the first two epochs to reduce overhead
                    visualize_correspondences(img1_np, img2_np, kp1, kp2_transformed, predicted_correspondence.tolist(), epoch, i+1)

        if valid_pairs > 0:
            batch_loss = batch_loss / valid_pairs
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item() * valid_pairs
            total_valid_pairs += valid_pairs
        if (batch_idx + 1) % 10 == 0 and valid_pairs > 0:
            avg_loss = total_loss / total_valid_pairs
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Avg Loss: {avg_loss:.4f}")

@torch.no_grad()
def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    num_examples = 0
    kk_once = True
    for batch_idx, (img1, img2, label, keypoints_before, keypoints_after) in enumerate(test_loader):
        batch_size_current = img1.size(0)
        for i in range(batch_size_current):
            img1_np = (img1[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img2_np = (img2[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            kp1 = keypoints_before[i].cpu().numpy()
            kp2_transformed = keypoints_after[i].cpu().numpy()
            img_width, img_height = RESIZE
            valid_indices = (
                (kp2_transformed[:, 0] >= 0) & (kp2_transformed[:, 0] < img_width) &
                (kp2_transformed[:, 1] >= 0) & (kp2_transformed[:, 1] < img_height)
            )
            kp1 = kp1[valid_indices]
            kp2_transformed = kp2_transformed[valid_indices]
            descriptors1 = compute_dense_descriptors(img1_np, kp1)
            descriptors2 = compute_dense_descriptors(img2_np, kp2_transformed)
            graph1 = build_graph_with_descriptors(descriptors1, kp1)
            graph2 = build_graph_with_descriptors(descriptors2, kp2_transformed)
            if graph1.num_nodes == 0 or graph2.num_nodes == 0:
                print(f"[Test Epoch {epoch+1}, Batch {batch_idx+1}, Sample {i+1}] Skipped due to no keypoints after transformation.")
                continue
            graph1 = graph1.to(device)
            graph2 = graph2.to(device)
            S_0, S_L = model(graph1.x, graph1.edge_index, graph1.edge_attr, graph1.batch,
                            graph2.x, graph2.edge_index, graph2.edge_attr, graph2.batch)
            correspondence_source_to_target = generate_y(kp1, kp2_transformed).to(device)
            acc = model.acc(S_L, correspondence_source_to_target, reduction='sum')
            correct += acc.item()
            num_examples += (correspondence_source_to_target != -1).sum().item()
            if epoch < 10:
                S_final_np = S_L.cpu().detach().numpy()
                row_ind, col_ind = linear_sum_assignment(-S_final_np)
                predicted_correspondence = np.full(S_final_np.shape[0], -1, dtype=int)
                for r, c in zip(row_ind, col_ind):
                    if r < S_final_np.shape[0] and c < S_final_np.shape[1]:
                        predicted_correspondence[r] = c
                if kk_once:
                    visualize_correspondences(img1_np, img2_np, kp1, kp2_transformed, predicted_correspondence.tolist(), epoch, i+1)
                    kk_once = False
            if num_examples >= 1000:
                break
        if num_examples >= 1000:
            break
    accuracy = correct / num_examples if num_examples > 0 else 0
    print(f"Test Epoch [{epoch+1}/{NUM_EPOCHS}], Accuracy: {accuracy:.4f}")
    return accuracy

def custom_collate_fn(batch):
    img1_batch, img2_batch, labels, keypoints_before_batch, keypoints_after_batch = [], [], [], [], []
    for item in batch:
        img1_batch.append(item[0])
        img2_batch.append(item[1])
        labels.append(item[2])
        keypoints_before_batch.append(torch.tensor(item[3], dtype=torch.float32))
        keypoints_after_batch.append(torch.tensor(item[4], dtype=torch.float32))
    img1_batch = torch.stack(img1_batch)
    img2_batch = torch.stack(img2_batch)
    labels = torch.tensor(labels)
    return img1_batch, img2_batch, labels, keypoints_before_batch, keypoints_after_batch

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

def main():
    print("Starting the keypoint matching training pipeline.")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    os.makedirs('./data', exist_ok=True)
    train_dataset = ShiftedTinyImageNet(train=True, transform=transform, shift=SHIFT, resize=RESIZE, subset_fraction=0.1)
    test_dataset = ShiftedTinyImageNet(train=False, transform=transform, shift=SHIFT, resize=RESIZE, subset_fraction=0.1)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True, collate_fn=custom_collate_fn)
    model = KeypointMatchingModel().to(device)
    print_model_size(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_model(model, train_loader, optimizer, epoch)
        test_model(model, test_loader, epoch)
    print("Training completed.")

if __name__ == '__main__':
    main()
