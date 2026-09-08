# Direct-Cost Inflation Methodology (V2)

## Objective and published measures

The model estimates the cost escalation this buyer experiences for recurring direct-cost inputs. It publishes realized, committed, and projected measures separately because receipts, open commitments, and future demand answer different questions.

| Output | Question | Eligible evidence |
|---|---|---|
| Realized inflation | What changed in prices actually received? | Positive received quantity and effective realized unit price |
| Committed-cost signal | What pressure is embedded in open POs? | Remaining open quantity matched to a prior realized price |
| Fixed-basket forecast | What escalation should be budgeted? | Historical bucket dynamics, committed overlay, approved weights |

The official scope contains COS, Inventory, Operating Supplies, Production Supplies, Production Aids, and Small Tooling.

## Effective prices

For received line $r$:

$$
p_r^{R}=\frac{\text{ExtensionReceived}_r}{\text{QuantityReceived}_r}.
$$

For an open or partially open line:

$$
p_r^{C}=\frac{\text{POValue}_r}{\text{QuantityOrdered}_r},\qquad
V_r^{C}=\max(q_r^{O}-q_r^{R},0)p_r^{C}.
$$

These fields never enter realized history interchangeably. `Cost` is a fallback only when the extension-based field is unavailable. Audit output includes

$$
\rho_r=\frac{\text{POValue}_r}{\text{Cost}_r q_r^{O}}
$$

and detects possible power-of-ten price bases. An approved UOM factor $a_i$ transforms $p_i'=p_i/a_i$ and $q_i'=a_iq_i$, preserving value.

## Matching and interval observations

Official comparison entities are exact normalized part keys or client-approved `ReplacementPartKey` groups. Heuristic families remain excluded until approved.

Consecutive observed purchases form pair $j$:

$$
y_j=\log\left(\frac{p_{j,2}}{p_{j,1}}\right),\qquad
x_j^{q}=\log\left(\frac{q_{j,2}}{q_{j,1}}\right).
$$

The pair may span many calendar months. Its exposure is distributed across them:

$$
y_j=\sum_mD_{jm}\delta_m+\gamma x_j^q+\varepsilon_j,
$$

where $D_{jm}$ is the fraction of month $m$ in the interval. This prevents assigning a multi-month change wholly to its endpoint.

## Robust identified estimation

The estimator minimizes

$$
\min_{\delta,\gamma}\sum_jw_j\mathcal{H}_{\kappa}
\left(y_j-D_j\delta-\gamma x_j^q\right)
+\lambda_s\|\Delta^2\delta\|_2^2
+\lambda_r\|\delta\|_2^2
+\lambda_\gamma\gamma^2.
$$

Base weights use capped geometric spend divided by the square root of entity pair count. Extreme short-interval ratios are downweighted, not used to remove the whole part.

Overall and bucket paths are fitted separately. Bucket $c$ shrinks toward overall:

$$
\tilde\delta_{c,m}=b_c\hat\delta_{c,m}+(1-b_c)\hat\delta_{0,m},
\qquad b_c=\frac{n_c}{n_c+k}.
$$

This avoids the rank deficiency from including an overall effect and a complete set of category effects in one design. A nonconverged overall fit blocks official output; a sparse/nonconverged bucket uses disclosed overall fallback.

## Historical index and coverage

$$
I_{c,t}=100\exp\left(\sum_{m\le t}\tilde\delta_{c,m}\right),\qquad
I_t^{D}=\sum_cs_c^*I_{c,t},\quad\sum_cs_c^*=1.
$$

Fiscal years are October 1–September 30. TTM ends at the latest complete month; a partial final month remains labeled partial in audit output and is excluded from rate forecasting.

Current matched-spend coverage is

$$
C_c=\frac{\text{current realized spend for entities with valid pairs}}
{\text{all current realized spend in bucket }c}.
$$

## Committed-cost overlay

For an open line matched to its latest earlier realized price:

$$
a_r=\frac{\log(p_r^C/p_{i,r}^{R,\text{prior}})}{\Delta d_r/365.25}.
$$

Bucket signals winsorize $a_r$, cap open-value weights, and report coverage. First-year forecast is

$$
g_{c,1}=(1-\omega_c)g_{c,1}^{I}+\omega_cg_c^{C},
$$

where

$$
\omega_c=\min\left(\omega_{\max},
\omega_{\max}\frac{C_c^C}{C_{\text{full}}^C}\right).
$$

Thus weak committed coverage cannot dominate the internal forecast.

## Forecast and fixed basket

Candidate methods are trailing-12-month mean, EWMA, damped Holt, and mean reversion. Rolling-origin backtesting selects using one 12-month fixed-basket composite error per cutoff. Training records, labels, and weights are truncated at every cutoff. Transaction WAPE and forced 0% `last_price` are not selection objectives.

Holdout price relatives are treated with the same robust policy as training evidence: log relatives are winsorized at the configured tails, bounded by the configured extreme-ratio guardrails, spend weights are capped, and flagged extreme observations retain one-quarter weight. The raw extreme-weight share is reported for every cutoff.

Later years revert toward long run:

$$
g_{c,h}=\bar g_c+\rho^{h-1}(g_{c,1}-\bar g_c),\quad h\in\{2,3\}.
$$

The fixed-basket multiplier is

$$
M(T_0,T)=\frac{\sum_iq_i^*p_i(T)}{\sum_iq_i^*p_i(T_0)}.
$$

At bucket level,

$$
M_h^D=\sum_cs_c^*M_{c,h},
$$

and year-$h$ escalation is $M_h^D/M_{h-1}^D-1$.

As a final publication guardrail, bucket annual log forecasts are bounded by configurable floors/caps (defaults: $\log(0.75)$ and $\log(1.50)$). The uncapped first-year signal and whether a bound was applied are included in the forecast audit table.

## Uncertainty and reproducibility

The cluster bootstrap samples comparison entities with replacement, preserves all pairs per sampled entity, and refits the selected estimator/forecast. Configured lower, median, and upper composite multiplier quantiles are reported with convergence rate.

Each run records input/output SHA-256 hashes, controls, seed, package/Python/platform versions, selected method, cutoff, counts, warnings, and elapsed time. Source hashes are checked again before publication.

## Data required for the next maturity level

Vendor ID, receipt date, currency, UOM/price basis, PO/line ID, facility, order status, and contract status are absent. The model therefore cannot yet attribute inflation across vendors, isolate supplier switching, prove duplicates, or perfectly time receipts. A client-approved BOM/MRP planned basket and supersession/UOM map should replace current proxies when available.
