# GeoVision — Project Report

## 1. Problem statement
Land-use and land-cover (LULC) maps feed urban planning, agriculture subsidies, deforestation monitoring, flood-risk modelling and insurance pricing. Sentinel-2 photographs the whole Earth every 5 days, but turning those pixels into labelled maps is still largely manual. **Goal:** build a deployable system that classifies satellite tiles into 10 land-use classes with high accuracy. Every prediction should be explainable, and the system should detect when live data stops looking like the training data.

**Success criteria:**
- Test macro-F1 of at least 0.97
- Calibrated confidences (ECE below 0.03)
- Under 50 ms per image on CPU
- An automatic drift alert when the input distribution shifts

## 2. Data
- **Source:** EuroSAT RGB. It has 27,000 labelled 64×64 Sentinel-2 tiles from 34 European countries.
- **Classes:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake. Each class has between 2,000 and 3,000 images.
- **Validation** (`data.scan_images`): every file is opened and verified, and unreadable files are dropped and counted. Image sizes are tabulated, and the result is written to `data/data_report.json`.
- **Split:** stratified 70/15/15 with a fixed seed. The split is written to `data/manifest.csv`, so every experiment uses identical test images.
- **Reference statistics:** the per-image brightness, contrast and channel means of the training set, plus the training class distribution, are saved. These become the baseline for production drift detection.

## 3. Preprocessing & augmentation
| Step | Why |
|---|---|
| Input size 64 px (native) | The reported model uses the native 64×64 tiles (final feature map 2×2). Upsampling to 128 px (4×4 feature map) is configurable and recommended when a GPU is available. |
| ImageNet normalisation | Matches the statistics the pretrained weights expect |
| H/V flip and 90° rotations | Overhead imagery has no canonical orientation, so these augmentations never change the label. |
| RandomResizedCrop (0.7–1.0) | Scale invariance |
| Colour jitter | Robustness to season, haze and sensor differences |
| MixUp (α = 0.2) | Regularisation and better calibration |
| Label smoothing (0.1) | Prevents over-confident logits |

## 4. Models
1. **Baseline CNN (from scratch):** 4 conv blocks (Conv-BN-ReLU ×2 followed by MaxPool) and global average pooling. This baseline measures how much transfer learning actually adds.
2. **ResNet-50:** residual connections. A classic strong baseline.
3. **EfficientNet-B0** (default): compound-scaled, about 4 M parameters. It gives the best accuracy per FLOP and is the right choice for a 4 GB GPU and for CPU serving.
4. **ConvNeXt-Tiny:** a modern CNN with transformer-style design choices.

The new head on each pretrained backbone is Dropout followed by a Linear layer. All four architectures are implemented behind one interface; **the reported results come from EfficientNet-B0 only** — the other three were not trained for this report (`scripts/compare_models.py` runs the comparison).

## 5. Training strategy
- **Two-stage fine-tuning.** In epoch 1 the backbone is frozen and only the randomly initialised head trains. This stops large early gradients from destroying the pretrained features. After that, the whole network is unfrozen.
- **Discriminative learning rates.** The backbone LR is 0.1 times the head LR.
- **Optimiser:** AdamW (decoupled weight decay) with a cosine annealing schedule.
- **Mixed precision (AMP):** enabled automatically on CUDA GPUs (about 2× faster, half the memory). It was inactive in our run, which used the CPU because the GPU driver did not support CUDA 12.6.
- **Run used for the reported model:** 5 epochs, 64 px, batch 64, CPU, 22 minutes in total; validation accuracy 73.0% → 88.3% → 89.9% → 92.6% → 92.8%.
- **Gradient clipping** at a norm of 5.
- **Model selection:** validation **macro-F1**, which is robust to the mild class imbalance. Training uses early stopping with a patience of 4, and only the best checkpoint is kept. The checkpoint stores weights, class names, image size, normalisation, metrics and the full config, so the model is self-describing.

## 6. Evaluation
`artifacts/test_metrics.json` reports:
- Accuracy, macro and weighted F1, top-3 accuracy
- Per-class precision and recall
- **Expected Calibration Error**, plus a reliability diagram
- A confusion matrix
- The most-confident mistakes
- Single-image latency

TTA averages the predictions from the original image, an H-flip and a V-flip.

**Observed error analysis.** The five most frequent confusions account for 100 of the 221 test errors:
- *Industrial* → *Residential* (30): both are built-up areas with rooftops and roads.
- *HerbaceousVegetation* → *Forest* (21) and *Pasture* → *Forest* (15): green cover at 10 m/pixel.
- *AnnualCrop* → *PermanentCrop* (19) and *PermanentCrop* → *HerbaceousVegetation* (15): field textures that depend on season.

Grad-CAM on the Industrial example concentrates on the large factory roofs, and the two misclassified tiles in the gallery have low confidence (31% and 43%).

> **Measured results** (EfficientNet-B0, 5 epochs on CPU at 64 px, TTA on; `artifacts/test_metrics.json`):
> | Model | Test acc | Macro-F1 | Top-3 | ECE | ms/img |
> |---|---|---|---|---|---|
> | EfficientNet-B0 | 94.5% | 0.945 | 99.6% | 0.187 | 12.6 (CPU) |
>
> Largest confusions: Industrial→Residential (30), HerbaceousVegetation→Forest (21), AnnualCrop→PermanentCrop (19). ECE is high because the model is under-confident (mean confidence 76% vs 94.5% accuracy), a known side effect of MixUp + label smoothing; temperature scaling on the validation set is the standard fix.

## 7. Explainability
Grad-CAM is implemented from scratch in `explain.py`. Hooks capture the activations of the last conv block and the gradients of the predicted logit with respect to them. Each channel is weighted by its average gradient. The weighted channels are summed, passed through ReLU, upsampled to the input size and overlaid on the input. The API returns the heatmap with `?explain=true`.

## 8. Deployment
- **FastAPI:** 15 routes, including `/predict`, `/predict/batch`, `/feedback`, `/model-info`, `/model/report`, `/health`, `/monitoring/*`, `/samples`, `/artifacts/*` and `/metrics` (Prometheus). The service validates inputs (file type, 10 MB limit, at most 32 images per batch). It loads the model once at startup, and it returns a `503` error if the model is missing rather than crashing.
- **Web app** (served by FastAPI at `/`): classify by upload, drag-and-drop, paste or one-click test images; Grad-CAM viewer; feedback; batch processing with CSV export; live monitoring with traffic simulation; model report. A legacy Streamlit dashboard is also included.
- **ONNX export:** a framework-independent model for edge or cloud runtimes. A numerical parity check against PyTorch confirms the export.
- **Docker:** a Dockerfile for a slim CPU image (non-root user, healthcheck) and docker-compose are provided; they were not built as part of this project.
- **CI:** GitHub Actions runs all 20 tests on every push (passing).

## 9. Monitoring
| Signal | Method | Why it matters |
|---|---|---|
| Data drift | PSI of brightness, contrast and R/G/B means (recent window vs training reference) | Catches new sensors, haze, seasonal change and wrong band order before accuracy visibly drops |
| Prediction drift | PSI of the predicted-class mix vs the training label mix | Shows the model suddenly predicting one class for everything |
| Model health | Mean confidence and the share of predictions below 0.6 | Reported for context only — this model is deliberately under-confident, so it does not set the drift status |
| Live accuracy | Analysts submit true labels through `/feedback` | The only direct measure of real-world performance |
| Ops | Request and error counters, p50/p95 latency via `/metrics` | SLOs and alerting in Prometheus/Grafana |

The status uses **noise-corrected PSI**: raw PSI minus its expected value with no drift, (bins − 1) / n, so small windows of normal traffic do not raise false warnings. Thresholds: below 0.1 OK, 0.1–0.25 warning, above 0.25 alert. An alert recommends labelling recent traffic and retraining.

**Measured** on the running API (fixed random seeds): 60 normal test images → **Stable**, every corrected PSI 0.000, 56/60 correct. Adding 60 images darkened to 45% brightness → **Alert**: brightness 0.541, R 0.375, G 0.608, B 0.683, contrast 0.192, class mix 0.233; only 26/60 darkened images correct. The web app's *Simulate* buttons and `scripts/simulate_traffic.py --drift dark|haze|color` reproduce this.

## 10. Limitations & future work
- RGB only. Sentinel-2 has 13 spectral bands, and using all of them (EuroSAT-MS) would help separate the crop classes.
- Tile-level classification only. A natural extension is semantic segmentation (U-Net) for pixel-level maps.
- Retraining is triggered by hand. The next step is an automated retraining pipeline, for example with Airflow, gated by evaluation results.
- Domain shift: the training data covers Europe only, so imagery from other continents would need fine-tuning.

---

## Interview / viva questions you should be ready for

1. **Why transfer learning when the images are only 64×64?** Low-level filters such as edges and textures transfer across domains. With about 19k training images, a scratch model tends to overfit. The project includes a baseline CNN so the gap can be measured with `scripts/compare_models.py` (not run for this report).
2. **Why freeze the backbone for the first epoch?** The new head starts random, so early gradients are large and noisy. Back-propagating them into pretrained weights damages those weights, a problem known as catastrophic forgetting.
3. **Why macro-F1 rather than accuracy for model selection?** Macro-F1 weights every class equally, so a weak minority class can't hide behind the majority classes.
4. **What is ECE and why care?** ECE is the average gap between confidence and accuracy. Monitoring uses confidence as an accuracy proxy, and that only works if confidence is calibrated.
5. **How does Grad-CAM work?** See section 7. Its limitation is coarse resolution, because the final feature map is only a few cells wide.
6. **What is PSI?** It is defined as Σ (cur − ref) · ln(cur/ref) over bins. It measures how much a distribution has shifted, using quantile bins taken from the reference. Because PSI is biased upwards on small samples, we subtract its expected no-drift value (bins − 1) / n before applying thresholds.
7. **Data drift vs concept drift?** Data drift means P(x) changes, for example with a new satellite sensor. Concept drift means P(y|x) changes, for example when a class definition changes. PSI catches data drift, and catching concept drift needs labelled feedback.
8. **How would you scale this to all of India?** Tile the country with a sliding window and run batch inference with ONNX Runtime on GPU workers fed from a queue. Store the results as GeoTIFF/COG, and use the drift monitor per region and per season.
9. **What does AMP do?** It runs matmuls and convolutions in FP16 and keeps master weights in FP32. GradScaler scales the loss so small gradients don't underflow.
10. **Why stratified splitting?** It keeps the class proportions the same in every split, so validation and test metrics aren't biased by chance.
