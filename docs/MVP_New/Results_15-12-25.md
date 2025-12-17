(Stone) (base) [catman@catman-dell Stone]$ python -m src.observer_simple 2>&1
Simple Stone Observer initialized on cuda

============================================================
SETUP PHASE
============================================================

[1/5] Initializing BZ simulator...

[2/5] Training β-VAE for BZ patterns...
Generating 1000 BZ patterns for VAE training (batch_size=32)...
Generating batches: 100%|█████████████████████████████████████████████████████████████████████████| 32/32 [00:11<00:00,  2.72it/s]
Epoch 10/50: Loss=277.4454, Recon=277.0448, KL=0.1001
Epoch 20/50: Loss=97.0921, Recon=96.8492, KL=0.0607
Epoch 30/50: Loss=59.5900, Recon=59.0128, KL=0.1443
Epoch 40/50: Loss=46.6237, Recon=45.4796, KL=0.2860
Epoch 50/50: Loss=41.1836, Recon=39.1888, KL=0.4987
  VAE trained. Latent dim: 32

[3/5] Initializing graph generator...

[4/5] Creating target specification...
  Baseline P₀ = 0.000%

[5/5] Finding optimal θ* and training policy...
  Finding optimal θ* via evolution...
    Gen 10: best_hit=100.00%, mean_fit=60.67%
    Gen 20: best_hit=100.00%, mean_fit=87.83%
    Gen 30: best_hit=100.00%, mean_fit=93.67%
    Gen 40: best_hit=100.00%, mean_fit=99.00%
    Gen 50: best_hit=100.00%, mean_fit=100.00%
  Found θ* with hit rate: 100.00%
  Training policy (supervised)...
    Epoch 25: loss = 0.014008
    Epoch 50: loss = 0.001644
    Epoch 75: loss = 0.000510
    Epoch 100: loss = 0.000236
  Policy training complete!

============================================================
SETUP COMPLETE
============================================================

============================================================
INITIAL EVALUATION
============================================================
P_Φ (initial) = 100.000%

P₀ estimation:
  Quick estimate: 0.000000%
Precise P₀ estimation: 100%|█████████████████████████████████████████████████████████████| 100000/100000 [08:17<00:00, 203.15it/s]
  No hits in 100000 samples
Precise P₀ estimation: 100%|█████████████████████████████████████████████████████████████| 100000/100000 [08:17<00:00, 201.18it/s]
  Precise: 0/100000 = 0.000000%
  95% CI: [0.00e+00, 3.00e-05]
  Initial Improvement: ∞ (P₀ < 3.00e-05)

============================================================
TRAINING PHASE (Fine-tuning)
============================================================
Episode 100/500 | Ξ=0.000 | Hit=100.00% | λ=3.00e-04 | Time=33s
Episode 200/500 | Ξ=0.000 | Hit=100.00% | λ=3.00e-04 | Time=67s
Episode 300/500 | Ξ=0.000 | Hit=100.00% | λ=3.00e-04 | Time=100s
Episode 400/500 | Ξ=0.000 | Hit=100.00% | λ=3.00e-04 | Time=132s
Episode 500/500 | Ξ=0.000 | Hit=100.00% | λ=3.00e-04 | Time=164s

============================================================
FINAL EVALUATION
============================================================
P_Φ (final) = 100.000%
P₀ = 0.000000% (upper bound: 3.00e-05)
Improvement: > 33333x (P₀ upper bound used)

============================================================
SUMMARY RESULTS
============================================================
P₀ = 0.000000%
P₀ upper bound (95% CI) = 3.00e-05
P₀ samples = 100,000
P_Φ = 100.000%
Improvement = 33333x
Ξ (final) = 0.0000
(Stone) (base) [catman@catman-dell Stone]$ 