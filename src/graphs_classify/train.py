import os, random, torch, numpy as np
from hydra import main
from omegaconf import DictConfig
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from graphs_classify.models.gin import GINNet
# аналогично импорт по имени модели из конфига

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(cfg, num_features, num_classes):
    if cfg.model.name == "GIN":
        return GINNet(num_features, cfg.model.hidden_dim, num_classes,
                      num_layers=cfg.model.num_layers, dropout=cfg.model.dropout)
    # elif ... GCN/GAT/GraphSAGE/PNA
    raise ValueError("Unknown model")

@main(config_path="../../configs", config_name="config", version_base=None)
def run(cfg: DictConfig):
    seed_everything(cfg.seed)
    ds = TUDataset(root=cfg.dataset.root, name=cfg.dataset.name)
    num_features = ds.num_features or 1
    num_classes = ds.num_classes
    model = get_model(cfg, num_features, num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # простая K-fold CV
    idx = np.arange(len(ds))
    folds = np.array_split(np.random.permutation(idx), cfg.train.k_folds)
    metrics = []
    for k in range(cfg.train.k_folds):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(cfg.train.k_folds) if i != k])
        train_loader = DataLoader(ds[train_idx.tolist()], batch_size=cfg.train.batch_size, shuffle=True)
        test_loader  = DataLoader(ds[test_idx.tolist()],  batch_size=cfg.train.batch_size)

        opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        best_acc, patience, bad = 0, cfg.train.patience, 0

        for epoch in range(cfg.train.max_epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = torch.nn.functional.cross_entropy(logits, batch.y)
                opt.zero_grad(); loss.backward(); opt.step()

            # val == test (для краткости; в реале выдели валидацию)
            model.eval(); y_true=[]; y_pred=[]
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    probs = model(batch)
                    y_pred.extend(probs.argmax(1).cpu().tolist())
                    y_true.extend(batch.y.cpu().tolist())
            acc = accuracy_score(y_true, y_pred)
            if acc > best_acc:
                best_acc, bad = acc, 0
            else:
                bad += 1
            if bad >= patience: break

        f1 = f1_score(y_true, y_pred, average="macro")
        metrics.append((acc, f1))

    acc_mean = float(np.mean([m[0] for m in metrics]))
    f1_mean = float(np.mean([m[1] for m in metrics]))
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/summary.csv","a") as f:
        f.write(f"{cfg.dataset.name},{cfg.model.name},{acc_mean:.4f},{f1_mean:.4f}\n")
    print({"acc": acc_mean, "f1_macro": f1_mean})

if __name__ == "__main__":
    run()
