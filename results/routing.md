
### view: full  (TRM params 72,276; gamma_arb=0.2, gamma_membrane=0.0; end-to-end on 800 prompts {'reverse-text': 200, 'medmcqa': 200, 'ifeval': 200, 'gsm8k': 200})

| router | routing acc (5 seeds) | abstain | end-to-end macro score | reverse-text | gsm8k | medmcqa | ifeval |
|---|---|---|---|---|---|---|---|
| oracle | — | — | 0.693 | 0.817 | 0.875 | 0.440 | 0.640 |
| no adapter | — | — | 0.494 | 0.115 | 0.845 | 0.375 | 0.640 |
| always reverse-text | — | — | 0.361 | 0.817 | 0.045 | 0.250 | 0.330 |
| always gsm8k | — | — | 0.505 | 0.131 | 0.875 | 0.370 | 0.645 |
| always medmcqa | — | — | 0.513 | 0.133 | 0.850 | 0.440 | 0.630 |
| always ifeval | — | — | 0.497 | 0.135 | 0.875 | 0.340 | 0.640 |
| trm | 0.998 (min 0.994) | 0.000 | 0.693 | 0.817 | 0.875 | 0.440 | 0.641 |
| ldt | 0.305 (min 0.305) | 0.630 | 0.506 | 0.153 | 0.845 | 0.390 | 0.635 |
| soft | 0.998 (min 0.994) | 0.000 | 0.693 | 0.817 | 0.875 | 0.440 | 0.641 |
| hard-filter | 0.935 (min 0.934) | 0.000 | 0.693 | 0.817 | 0.875 | 0.440 | 0.640 |
| arbitration | 0.998 (min 0.994) | 0.000 | 0.693 | 0.817 | 0.875 | 0.440 | 0.641 |
| membrane | 0.998 (min 0.994) | 0.000 | 0.693 | 0.817 | 0.875 | 0.440 | 0.641 |
| trm-abstain | 0.998 (min 0.994) | 0.000 | 0.693 | 0.817 | 0.875 | 0.440 | 0.641 |
| membrane (e2e-tuned) | 0.998 (min 0.994) | 0.000 | 0.693 | 0.817 | 0.875 | 0.440 | 0.641 |
| trm-abstain (e2e-tuned) | 0.998 (min 0.994) | 0.000 | 0.693 | 0.817 | 0.875 | 0.440 | 0.641 |
| trm + expected-utility (Bayes) | 0.643 (min 0.603) | 0.126 | 0.684 | 0.817 | 0.835 | 0.440 | 0.645 |

Routes chosen by trm (seed 0; rows = true env):

| true env | reverse-text | gsm8k | medmcqa | ifeval | abstain |
|---|---|---|---|---|---|
| reverse-text | 200 | 0 | 0 | 0 | 0 |
| gsm8k | 0 | 200 | 0 | 0 | 0 |
| medmcqa | 0 | 0 | 200 | 0 | 0 |
| ifeval | 0 | 1 | 0 | 199 | 0 |

Routes chosen by trm + expected-utility (Bayes) (seed 0; rows = true env):

| true env | reverse-text | gsm8k | medmcqa | ifeval | abstain |
|---|---|---|---|---|---|
| reverse-text | 200 | 0 | 0 | 0 | 0 |
| gsm8k | 0 | 100 | 0 | 0 | 100 |
| medmcqa | 0 | 0 | 200 | 0 | 0 |
| ifeval | 0 | 183 | 0 | 16 | 1 |

### view: content  (TRM params 72,276; gamma_arb=0.0, gamma_membrane=0.0; end-to-end on 800 prompts {'reverse-text': 200, 'medmcqa': 200, 'ifeval': 200, 'gsm8k': 200})

| router | routing acc (5 seeds) | abstain | end-to-end macro score | reverse-text | gsm8k | medmcqa | ifeval |
|---|---|---|---|---|---|---|---|
| oracle | — | — | 0.693 | 0.817 | 0.875 | 0.440 | 0.640 |
| no adapter | — | — | 0.494 | 0.115 | 0.845 | 0.375 | 0.640 |
| always reverse-text | — | — | 0.361 | 0.817 | 0.045 | 0.250 | 0.330 |
| always gsm8k | — | — | 0.505 | 0.131 | 0.875 | 0.370 | 0.645 |
| always medmcqa | — | — | 0.513 | 0.133 | 0.850 | 0.440 | 0.630 |
| always ifeval | — | — | 0.497 | 0.135 | 0.875 | 0.340 | 0.640 |
| trm | 0.865 (min 0.836) | 0.000 | 0.615 | 0.575 | 0.823 | 0.437 | 0.625 |
| ldt | 0.305 (min 0.305) | 0.426 | 0.507 | 0.156 | 0.845 | 0.390 | 0.635 |
| soft | 0.865 (min 0.836) | 0.000 | 0.615 | 0.575 | 0.823 | 0.437 | 0.625 |
| hard-filter | 0.683 (min 0.666) | 0.000 | 0.553 | 0.283 | 0.846 | 0.447 | 0.637 |
| arbitration | 0.865 (min 0.836) | 0.000 | 0.615 | 0.575 | 0.823 | 0.437 | 0.625 |
| membrane | 0.865 (min 0.836) | 0.000 | 0.615 | 0.575 | 0.823 | 0.437 | 0.625 |
| trm-abstain | 0.865 (min 0.836) | 0.002 | 0.615 | 0.575 | 0.822 | 0.437 | 0.625 |
| membrane (e2e-tuned) | 0.862 (min 0.835) | 0.006 | 0.613 | 0.566 | 0.825 | 0.437 | 0.625 |
| trm-abstain (e2e-tuned) | 0.863 (min 0.834) | 0.006 | 0.614 | 0.570 | 0.823 | 0.437 | 0.625 |
| trm + expected-utility (Bayes) | 0.570 (min 0.541) | 0.128 | 0.603 | 0.572 | 0.781 | 0.433 | 0.627 |

Routes chosen by trm (seed 0; rows = true env):

| true env | reverse-text | gsm8k | medmcqa | ifeval | abstain |
|---|---|---|---|---|---|
| reverse-text | 135 | 22 | 8 | 35 | 0 |
| gsm8k | 8 | 185 | 0 | 7 | 0 |
| medmcqa | 0 | 0 | 200 | 0 | 0 |
| ifeval | 13 | 2 | 1 | 184 | 0 |

Routes chosen by trm + expected-utility (Bayes) (seed 0; rows = true env):

| true env | reverse-text | gsm8k | medmcqa | ifeval | abstain |
|---|---|---|---|---|---|
| reverse-text | 136 | 16 | 8 | 29 | 11 |
| gsm8k | 8 | 46 | 0 | 56 | 90 |
| medmcqa | 1 | 0 | 199 | 0 | 0 |
| ifeval | 14 | 87 | 0 | 98 | 1 |
