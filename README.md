# Model przewidujący popularność NFT  
EfficientNet-B0 do przewidywania, czy NFT osiągnie cenę z górnego kwartyla

## Spis treści
1. [Opis projektu](#opis-projektu)  
2. [Struktura repozytorium](#struktura-repozytorium)  
3. [Uruchamianie treningu](#uruchamianie-treningu)  
---

## Opis projektu <a id="opis-projektu"></a>

Skrypt **`script.py`** trenuje wstępnie wytrenowany model
EfficientNet-B0 (PyTorch/Torchvision) na obrazach NFT, aby
sklasyfikować obraz jako **popularny**
lub **niepopularny**.



---

## Struktura repozytorium <a id="struktura-repozytorium"></a>

```text
AI_PROJEKT/
├── data_gathering/
│   ├── images/              # pliki PNG (wejście do treningu)
│   └── nft_dataset.csv      # metadane z ceną, kolekcją, nazwą
├── script.py                # główny skrypt treningowy
└──  nft_classifier.pt        # wagi przykładowego modelu (output)
```
## Uruchamianie treningu <a id="uruchamianie-treningu"></a>

```bash
pip install (ręcznie)
python script.py
```
