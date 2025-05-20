
import random, time, math, os
from pathlib import Path
from typing import List

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms

from PIL import Image, ImageFile
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from torch.optim.lr_scheduler import CosineAnnealingLR  
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ——————————————————  konfiguracja  ——————————————————
CSV_PATH   = "./data_gathering/nft_dataset.csv"
IMG_DIR    = Path("data_gathering/images")
TEST_SIZE  = 0.20
BATCH_SIZE = 32              
EPOCHS_HEAD, EPOCHS_FULL = 3, 7
EARLY_STOP_PATIENCE = 3
LR_HEAD      = 3e-4
LR_BACKBONE  = 1e-4
FREEZE_UP_TO = "features.4"
SEED         = 42

# powtarzalność
random.seed(SEED);  torch.manual_seed(SEED)

# Dostosowanie obiążenia RAM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    BATCH_SIZE = 8    
print(f"💻  device: {device} | batch: {BATCH_SIZE}")

# wczytanie CSV + walidacja obrazów
df = pd.read_csv(CSV_PATH)
def safe(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-").replace("\\", "-").replace("#","")

paths: List[Path]; prices: List[float]; colls: List[str]
paths, prices, colls, bad = [], [], [], []
ImageFile.LOAD_TRUNCATED_IMAGES = True 

for _, r in df.iterrows():
    n = r["name"] if pd.notnull(r["name"]) else str(r["token_id"])
    p = IMG_DIR / f"{r['collection']}_{safe(n)}.png"
    if not p.is_file(): continue
    try:
        with Image.open(p) as im: im.verify()
        paths.append(p); prices.append(r["last_sale_price"]); colls.append(r["collection"])
    except Exception: continue

data = pd.DataFrame({"path":paths,"price":prices,"collection":colls})
kwantyl = data["price"].quantile(0.75)
data["label"] = (data["price"] >= kwantyl).astype(int)


# split po kolekcjach
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
tr_idx, val_idx = next(gss.split(data, groups=data["collection"]))
tr_df, val_df   = data.iloc[tr_idx], data.iloc[val_idx]

# przystosowanie i zwiększenie róźnorodności danych
train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, (0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3,0.3,0.3,0.1),
        transforms.ToTensor(),
])
val_tf   = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

class NFT(Dataset):
    def __init__(self, frame, tf):
        self.df, self.tf = frame.reset_index(drop=True), tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        try:
            img = Image.open(row.path).convert("RGB")
            img = self.tf(img)
            return img, torch.tensor(row.label, dtype=torch.float32)
        except Exception: return None

def collate(batch):
    batch=[b for b in batch if b]; 
    return None if not batch else torch.utils.data.dataloader.default_collate(batch)


class_bal = tr_df["label"].value_counts()
weights   = tr_df["label"].apply(lambda x: 1./class_bal[x]).values
sampler   = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

tr_loader = DataLoader(NFT(tr_df,train_tf), batch_size=BATCH_SIZE,
                       sampler=sampler, num_workers=0,
                       pin_memory=(device.type=="cuda"), collate_fn=collate)
val_loader= DataLoader(NFT(val_df,val_tf),   batch_size=BATCH_SIZE,
                       shuffle=False, num_workers=0,
                       pin_memory=(device.type=="cuda"), collate_fn=collate)


####################################################################################################


# model
weights = EfficientNet_B0_Weights.DEFAULT
model   = efficientnet_b0(weights=weights)

# zamrożenie części warstw
unfreeze=False
for n,p in model.named_parameters():
    if FREEZE_UP_TO in n: unfreeze=True
    p.requires_grad = unfreeze

# label‑smoothing
in_f = model.classifier[1].in_features
class SmoothedLinear(nn.Linear):
    def forward(self, x, eps=0.1):
        out = super().forward(x)
        return out * (1-eps) + eps/2   
    
# przypisanie nowej głowy
model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        SmoothedLinear(in_f, 1)

)

model.to(device)

# Niestandardowa funkcja straty - Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2): super().__init__(); self.g=gamma
    def forward(self, logit, target):
        p = torch.sigmoid(logit)
        pt = p*target + (1-p)*(1-target)
        w  = (1-pt).pow(self.g)
        bce= nn.functional.binary_cross_entropy_with_logits(logit, target, reduction='none')
        return (w*bce).mean()
    
criterion = FocalLoss(gamma=2)

# konfiguracja learning rate
head_params = [p for n,p in model.named_parameters() if 'classifier' in n]
bb_params   = [p for n,p in model.named_parameters() if 'classifier' not in n]
opt = torch.optim.AdamW([
        {'params': head_params, 'lr': LR_HEAD},
        {'params': bb_params,   'lr': LR_BACKBONE}
    ], weight_decay=1e-4)
sched = CosineAnnealingLR(opt, T_max=EPOCHS_HEAD + EPOCHS_FULL) 

# trenowanie lub testowanie etapu (funkcja pomocnicza)
def epoch(loader, train):
    model.train(train)
    tot,n,yt,yh = 0,0,[],[]
    for batch in loader:
        if batch is None: continue
        xb,yb = [t.to(device) for t in batch];  yb=yb.unsqueeze(1)
        with torch.set_grad_enabled(train):
            lo = model(xb);  loss = criterion(lo,yb)
            if train: opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*len(xb); n+=len(xb)
        yt.append(yb.cpu()); yh.append(torch.sigmoid(lo).detach().cpu())
    if n==0: return math.nan,0,0
    yt=torch.cat(yt); yh=torch.cat(yh); preds=(yh>=0.5).int()
    return tot/n, accuracy_score(yt,preds), f1_score(yt,preds)

best_f1, patience = 0, EARLY_STOP_PATIENCE
phase = 'head'

# trenowanie
for ep in range(1,EPOCHS_HEAD+EPOCHS_FULL+1):
    t0=time.time()
    tr_l,tr_a,tr_f=epoch(tr_loader,True)
    vl_l,vl_a,vl_f=epoch(val_loader,False)
    sched.step()

    print(f"[{ep:02}] {phase:5} | "
          f"tr loss {tr_l:.3f} acc {tr_a:.3f} f1 {tr_f:.3f} || "
          f"val loss {vl_l:.3f} acc {vl_a:.3f} f1 {vl_f:.3f} | "
          f"{time.time()-t0:.1f}s")

    # early stop
    if vl_f > best_f1+1e-3:
        best_f1, patience = vl_f, EARLY_STOP_PATIENCE
    else:
        patience -= 1
        if patience==0:
            print("early‑stopping – brak poprawy F1"); break

    # po epokach head‑only odmrażamy cały model
    if ep == EPOCHS_HEAD:
        print("odblokowuję cały backbone")
        for p in model.parameters(): p.requires_grad=True
        phase='full'

# raport końcowy
model.eval(); logits,labels=[],[]
with torch.no_grad():
    for batch in val_loader:
        if batch is None: continue
        xb,yb=[t.to(device) for t in batch]
        logits.append(torch.sigmoid(model(xb)).cpu()); labels.append(yb)
y_val=torch.cat(labels).numpy(); probs=torch.cat(logits).numpy()
y_pred=(probs>=0.5).astype(int)

print("classification report (val, thr=0.5):\n")
print(classification_report(y_val, y_pred, digits=3))

# zapisanie modelu
torch.save(model.state_dict(), "nft_classifier.pt")
print(" Model zapisany jako nft_classifier.pt")
