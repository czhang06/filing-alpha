# Do 8-K filings move prices?

- Events: **1,615** filings, **77** tickers, 2024-01-01 to 2025-12-31
- Observations: 6,460 (filing x event window)
- Market-model estimation window: [-250, -30] trading days; market proxy SPY
- Sentiment backend: none

Two different questions are asked of the same events:

1. **Direction** -- does the filing push the price up or down on average? 0 of 40 tests significant after Benjamini-Hochberg at q=0.05.
2. **Magnitude** -- does the filing make the price move *more*, in either direction? 24 of 40 tests significant at the same threshold.

The distinction matters. For earnings releases roughly half the surprises are good and half are bad, so the signed average can sit at zero even when every announcement moves the stock several percent. A mean test is structurally blind to that; the variance test is not.

## 1. Direction: mean cumulative abnormal return

`mde` is the smallest true effect this sample could have detected at 80% power. A null result is only meaningful next to it.

| Slice | Window | n | Mean CAR | 95% CI | % positive | BMP p | FDR sig | MDE |
|---|---|---:|---:|---|---:|---:|:-:|---:|
| All filings | [+0,+0] | 1,615 | +0.034% | [-0.176%, +0.250%] | 48.2% | 0.4094 | no | 0.303% |
| Item 1.01 (Material agreement entered) | [+0,+0] | 92 | +0.447% | [-0.072%, +1.119%] | 51.1% | 0.3236 | no | 0.868% |
| Item 2.02 (Results of operations) | [+0,+0] | 575 | +0.141% | [-0.399%, +0.684%] | 50.3% | 0.2354 | no | 0.777% |
| Item 2.03 (Direct financial obligation) | [+0,+0] | 52 | +0.043% | [-0.394%, +0.535%] | 44.2% | 0.6560 | no | 0.662% |
| Item 5.02 (Director/officer departure) | [+0,+0] | 309 | -0.186% | [-0.417%, +0.069%] | 43.7% | 0.3322 | no | 0.350% |
| Item 5.03 (Other) | [+0,+0] | 55 | +0.048% | [-0.357%, +0.472%] | 43.6% | 0.4589 | no | 0.604% |
| Item 5.07 (Other) | [+0,+0] | 136 | -0.008% | [-0.343%, +0.389%] | 48.5% | 0.7596 | no | 0.525% |
| Item 7.01 (Regulation FD disclosure) | [+0,+0] | 299 | +0.192% | [-0.321%, +0.732%] | 48.2% | 0.6635 | no | 0.750% |
| Item 8.01 (Other events) | [+0,+0] | 416 | +0.250% | [-0.090%, +0.613%] | 48.8% | 0.1871 | no | 0.502% |
| Item 9.01 (Other) | [+0,+0] | 1,240 | +0.100% | [-0.168%, +0.374%] | 48.5% | 0.2268 | no | 0.386% |
| All filings | [+0,+1] | 1,615 | +0.037% | [-0.199%, +0.275%] | 48.3% | 0.4246 | no | 0.341% |
| Item 1.01 (Material agreement entered) | [+0,+1] | 92 | +0.402% | [-0.325%, +1.294%] | 44.6% | 0.5913 | no | 1.166% |
| Item 2.02 (Results of operations) | [+0,+1] | 575 | +0.232% | [-0.355%, +0.829%] | 50.3% | 0.1437 | no | 0.844% |
| Item 2.03 (Direct financial obligation) | [+0,+1] | 52 | +0.363% | [-0.408%, +1.223%] | 44.2% | 0.6217 | no | 1.178% |
| Item 5.02 (Director/officer departure) | [+0,+1] | 309 | -0.374% | [-0.698%, -0.041%] | 42.7% | 0.0529 | no | 0.467% |
| Item 5.03 (Other) | [+0,+1] | 55 | +0.502% | [-0.156%, +1.192%] | 52.7% | 0.0541 | no | 0.970% |
| Item 5.07 (Other) | [+0,+1] | 136 | -0.042% | [-0.520%, +0.493%] | 50.0% | 0.8349 | no | 0.727% |
| Item 7.01 (Regulation FD disclosure) | [+0,+1] | 299 | +0.047% | [-0.543%, +0.655%] | 45.2% | 0.7110 | no | 0.850% |
| Item 8.01 (Other events) | [+0,+1] | 416 | +0.233% | [-0.159%, +0.654%] | 50.2% | 0.1373 | no | 0.588% |
| Item 9.01 (Other) | [+0,+1] | 1,240 | +0.134% | [-0.161%, +0.434%] | 49.7% | 0.1179 | no | 0.430% |
| All filings | [+0,+5] | 1,615 | -0.063% | [-0.357%, +0.232%] | 48.7% | 0.8627 | no | 0.420% |
| Item 1.01 (Material agreement entered) | [+0,+5] | 92 | +0.767% | [-0.414%, +2.034%] | 57.6% | 0.3027 | no | 1.758% |
| Item 2.02 (Results of operations) | [+0,+5] | 575 | +0.024% | [-0.628%, +0.686%] | 49.6% | 0.3838 | no | 0.944% |
| Item 2.03 (Direct financial obligation) | [+0,+5] | 52 | +0.793% | [-0.612%, +2.122%] | 63.5% | 0.2443 | no | 1.977% |
| Item 5.02 (Director/officer departure) | [+0,+5] | 309 | -0.296% | [-0.792%, +0.207%] | 42.1% | 0.2528 | no | 0.714% |
| Item 5.03 (Other) | [+0,+5] | 55 | -0.115% | [-1.115%, +0.885%] | 54.5% | 0.7409 | no | 1.441% |
| Item 5.07 (Other) | [+0,+5] | 136 | -0.251% | [-1.009%, +0.513%] | 48.5% | 0.5994 | no | 1.076% |
| Item 7.01 (Regulation FD disclosure) | [+0,+5] | 299 | -0.105% | [-0.858%, +0.660%] | 44.5% | 0.3699 | no | 1.088% |
| Item 8.01 (Other events) | [+0,+5] | 416 | -0.079% | [-0.614%, +0.447%] | 49.5% | 0.8535 | no | 0.756% |
| Item 9.01 (Other) | [+0,+5] | 1,240 | -0.045% | [-0.406%, +0.307%] | 49.0% | 0.9146 | no | 0.514% |
| All filings | [-1,+1] | 1,615 | +0.085% | [-0.163%, +0.339%] | 49.4% | 0.2048 | no | 0.364% |
| Item 1.01 (Material agreement entered) | [-1,+1] | 92 | +0.404% | [-0.454%, +1.361%] | 50.0% | 0.7457 | no | 1.299% |
| Item 2.02 (Results of operations) | [-1,+1] | 575 | +0.477% | [-0.120%, +1.086%] | 51.3% | 0.0181 | no | 0.855% |
| Item 2.03 (Direct financial obligation) | [-1,+1] | 52 | +0.541% | [-0.415%, +1.622%] | 53.8% | 0.4912 | no | 1.474% |
| Item 5.02 (Director/officer departure) | [-1,+1] | 309 | -0.522% | [-0.946%, -0.117%] | 44.3% | 0.0990 | no | 0.586% |
| Item 5.03 (Other) | [-1,+1] | 55 | +0.090% | [-0.597%, +0.779%] | 54.5% | 0.3589 | no | 0.995% |
| Item 5.07 (Other) | [-1,+1] | 136 | -0.417% | [-0.976%, +0.141%] | 46.3% | 0.2932 | no | 0.790% |
| Item 7.01 (Regulation FD disclosure) | [-1,+1] | 299 | +0.092% | [-0.569%, +0.739%] | 47.2% | 0.7352 | no | 0.931% |
| Item 8.01 (Other events) | [-1,+1] | 416 | +0.267% | [-0.156%, +0.712%] | 51.9% | 0.1327 | no | 0.629% |
| Item 9.01 (Other) | [-1,+1] | 1,240 | +0.209% | [-0.108%, +0.525%] | 51.0% | 0.0520 | no | 0.452% |

## 2. Magnitude: was the move bigger than a normal day?

Both ratios are 1.00 on an ordinary day. `|AR| ratio` is mean absolute standardized abnormal return against its no-event expectation; `var ratio` is mean squared standardized abnormal return, so 2.00 means twice the usual return variance. Tests are one-sided: an unusually *calm* day is not evidence of a reaction.

| Slice | Window | n | \|AR\| ratio | Var ratio | Variance p | FDR sig |
|---|---|---:|---:|---:|---:|:-:|
| All filings | [+0,+0] | 1,615 | 2.03x | 6.13x | 2.59e-38 | yes |
| Item 9.01 (Other) | [+0,+0] | 1,240 | 2.33x | 7.57x | 5.74e-38 | yes |
| All filings | [+0,+1] | 1,615 | 1.71x | 3.88x | 4.89e-37 | yes |
| Item 2.02 (Results of operations) | [+0,+0] | 575 | 3.68x | 14.08x | 5.61e-37 | yes |
| Item 9.01 (Other) | [+0,+1] | 1,240 | 1.91x | 4.65x | 2.33e-36 | yes |
| Item 2.02 (Results of operations) | [+0,+1] | 575 | 2.82x | 8.15x | 1.88e-35 | yes |
| All filings | [-1,+1] | 1,615 | 1.56x | 3.01x | 5.13e-33 | yes |
| Item 9.01 (Other) | [-1,+1] | 1,240 | 1.70x | 3.51x | 6.44e-32 | yes |
| Item 2.02 (Results of operations) | [-1,+1] | 575 | 2.32x | 5.62x | 1.21e-31 | yes |
| Item 2.02 (Results of operations) | [+0,+5] | 575 | 1.82x | 3.34x | 3.05e-28 | yes |
| Item 9.01 (Other) | [+0,+5] | 1,240 | 1.41x | 2.22x | 1.63e-27 | yes |
| All filings | [+0,+5] | 1,615 | 1.32x | 1.96x | 5.91e-27 | yes |
| Item 7.01 (Regulation FD disclosure) | [+0,+0] | 299 | 1.96x | 5.40x | 3.07e-10 | yes |
| Item 7.01 (Regulation FD disclosure) | [+0,+1] | 299 | 1.71x | 3.58x | 2.84e-09 | yes |
| Item 7.01 (Regulation FD disclosure) | [+0,+5] | 299 | 1.37x | 2.01x | 1.33e-07 | yes |
| Item 7.01 (Regulation FD disclosure) | [-1,+1] | 299 | 1.67x | 3.26x | 1.40e-07 | yes |
| Item 8.01 (Other events) | [+0,+5] | 416 | 1.16x | 1.55x | 4.58e-05 | yes |
| Item 8.01 (Other events) | [-1,+1] | 416 | 1.27x | 2.12x | 1.00e-04 | yes |
| Item 8.01 (Other events) | [+0,+0] | 416 | 1.48x | 3.84x | 1.19e-04 | yes |
| Item 8.01 (Other events) | [+0,+1] | 416 | 1.31x | 2.60x | 1.19e-04 | yes |
| Item 5.02 (Director/officer departure) | [+0,+1] | 309 | 1.11x | 1.64x | 3.10e-03 | yes |
| Item 5.02 (Director/officer departure) | [+0,+0] | 309 | 1.11x | 1.77x | 7.41e-03 | yes |
| Item 5.02 (Director/officer departure) | [-1,+1] | 309 | 1.11x | 1.64x | 2.43e-02 | yes |
| Item 5.02 (Director/officer departure) | [+0,+5] | 309 | 1.05x | 1.25x | 2.81e-02 | yes |
| Item 5.03 (Other) | [+0,+1] | 55 | 1.17x | 1.64x | 7.61e-02 | no |
| Item 1.01 (Material agreement entered) | [+0,+5] | 92 | 1.10x | 1.55x | 8.37e-02 | no |
| Item 1.01 (Material agreement entered) | [-1,+1] | 92 | 1.12x | 1.62x | 8.64e-02 | no |
| Item 5.07 (Other) | [+0,+1] | 136 | 1.08x | 1.48x | 1.04e-01 | no |
| Item 1.01 (Material agreement entered) | [+0,+1] | 92 | 1.12x | 1.99x | 1.06e-01 | no |
| Item 1.01 (Material agreement entered) | [+0,+0] | 92 | 1.06x | 2.04x | 1.67e-01 | no |
| Item 2.03 (Direct financial obligation) | [+0,+1] | 52 | 1.05x | 1.32x | 1.69e-01 | no |
| Item 2.03 (Direct financial obligation) | [+0,+5] | 52 | 1.13x | 1.28x | 1.72e-01 | no |
| Item 5.07 (Other) | [-1,+1] | 136 | 1.03x | 1.19x | 1.79e-01 | no |
| Item 5.07 (Other) | [+0,+0] | 136 | 0.98x | 1.46x | 2.17e-01 | no |
| Item 5.03 (Other) | [+0,+5] | 55 | 1.03x | 1.27x | 2.40e-01 | no |
| Item 5.03 (Other) | [+0,+0] | 55 | 0.95x | 1.24x | 2.67e-01 | no |
| Item 2.03 (Direct financial obligation) | [-1,+1] | 52 | 1.00x | 1.18x | 2.84e-01 | no |
| Item 5.07 (Other) | [+0,+5] | 136 | 0.96x | 1.11x | 3.08e-01 | no |
| Item 5.03 (Other) | [-1,+1] | 55 | 1.01x | 1.08x | 3.76e-01 | no |
| Item 2.03 (Direct financial obligation) | [+0,+0] | 52 | 0.95x | 0.99x | 5.14e-01 | no |

## How to read this

A row is evidence only if its p-value survives the FDR column. An uncorrected p below 0.05 means little with this many slices: several will clear that bar by chance alone.

Where the direction table is null but the variance table is not, the reading is that filings reliably move prices without being predictable in sign -- consistent with the market pricing the news in, rather than with a tradeable signal.

Nothing here is a strategy. Abnormal returns ignore spreads, market impact and borrow costs, and the variance result in particular is not directional.
