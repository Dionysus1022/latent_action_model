# Comparison Experiment Results

本文档由 `scripts/run_comparison_experiments.py` 生成或更新。

## Raw Runs

| Task | Method | Seed | Repeat | Success Rate | Eval Time Sec | Planning Total Sec | Avg Planning Sec | Calls | Replans/Ep | Final Goal Dist | Min Goal Dist | Steps To Success | Path Length | Straight Ratio | Action L2 | Action Delta | Action Jerk | Log Path |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cube | mpc_cem | 42 | 0 | 68 | 608.4769 | 563.86147 | 281.930735 | 2 | 2 | 0.070214 | 0.057912 | 4.735294 | 0.126146 | 2.513322 | 0.901379 | 1.079532 | 1.89397 | outputs/comparison_experiments/cube/mpc_cem_seed42_repeat0.log |
| cube | mpc_cem | 42 | 1 | 68 | 616.3364 | 569.873933 | 284.936967 | 2 | 2 | 0.070215 | 0.057912 | 4.735294 | 0.126153 | 2.513357 | 0.901564 | 1.080788 | 1.89688 | outputs/comparison_experiments/cube/mpc_cem_seed42_repeat1.log |
| cube | mpc_cem | 42 | 2 | 68 | 620.7537 | 572.717292 | 286.358646 | 2 | 2 | 0.070266 | 0.057982 | 4.735294 | 0.126168 | 2.513677 | 0.900837 | 1.080259 | 1.896437 | outputs/comparison_experiments/cube/mpc_cem_seed42_repeat2.log |
| cube | mpc_cem | 43 | 0 | 74 | 619.4157 | 563.655139 | 281.82757 | 2 | 2 | 0.056191 | 0.051216 | 3.351351 | 0.110168 | 1.610252 | 0.905143 | 1.06105 | 1.844157 | outputs/comparison_experiments/cube/mpc_cem_seed43_repeat0.log |
| cube | mpc_cem | 43 | 1 | 72 | 610.6544 | 558.74643 | 279.373215 | 2 | 2 | 0.056309 | 0.051328 | 2.472222 | 0.109935 | 1.602083 | 0.907187 | 1.062005 | 1.846637 | outputs/comparison_experiments/cube/mpc_cem_seed43_repeat1.log |
| cube | mpc_cem | 43 | 2 | 72 | 605.1933 | 555.732127 | 277.866064 | 2 | 2 | 0.056309 | 0.051328 | 2.472222 | 0.109935 | 1.602083 | 0.906567 | 1.060101 | 1.84248 | outputs/comparison_experiments/cube/mpc_cem_seed43_repeat2.log |
| cube | mpc_cem | 44 | 0 | 64 | 616.2093 | 557.689343 | 278.844671 | 2 | 2 | 0.071728 | 0.064997 | 4.875 | 0.119552 | 25.262963 | 0.903797 | 1.091722 | 1.901449 | outputs/comparison_experiments/cube/mpc_cem_seed44_repeat0.log |
| cube | mpc_cem | 44 | 1 | 64 | 601.539 | 551.341128 | 275.670564 | 2 | 2 | 0.07173 | 0.064997 | 4.875 | 0.11949 | 13.597583 | 0.903919 | 1.090525 | 1.898101 | outputs/comparison_experiments/cube/mpc_cem_seed44_repeat1.log |
| cube | mpc_cem | 44 | 2 | 64 | 596.5686 | 548.503876 | 274.251938 | 2 | 2 | 0.07173 | 0.064997 | 4.875 | 0.119568 | 25.300217 | 0.903691 | 1.090915 | 1.899127 | outputs/comparison_experiments/cube/mpc_cem_seed44_repeat2.log |
| cube | gc_idm | 42 | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | outputs/comparison_experiments/cube/gc_idm_seed42_repeat0.log |
| cube | ours_full | 42 | 0 | 100 | 89.846500 | 19.039692 | 9.519846 | 2 | 2 | 0.025073 | 0.025073 | 10.600000 | 0.140414 | 1.254424 | 0.853438 | 0.109490 | 0.086116 | outputs/comparison_experiments/cube/ours_full_seed42_repeat0.log |
| cube | ours_full | 42 | 1 | 100 | 81.138000 | 20.980360 | 10.490180 | 2 | 2 | 0.024788 | 0.024788 | 10.640000 | 0.142486 | 1.266736 | 0.861248 | 0.109936 | 0.085263 | outputs/comparison_experiments/cube/ours_full_seed42_repeat1.log |
| cube | ours_full | 42 | 2 | 100 | 80.917500 | 19.168844 | 9.584422 | 2 | 2 | 0.025274 | 0.025274 | 10.600000 | 0.142604 | 1.266223 | 0.859641 | 0.112069 | 0.087307 | outputs/comparison_experiments/cube/ours_full_seed42_repeat2.log |
| cube | ours_full | 43 | 0 | 100 | 96.159400 | 20.479925 | 10.239963 | 2 | 2 | 0.021304 | 0.021304 | 8.460000 | 0.121659 | 1.364023 | 0.767144 | 0.111320 | 0.083367 | outputs/comparison_experiments/cube/ours_full_seed43_repeat0.log |
| cube | ours_full | 43 | 1 | 100 | 85.242300 | 17.457411 | 8.728706 | 2 | 2 | 0.021392 | 0.021392 | 8.480000 | 0.121215 | 1.353711 | 0.758211 | 0.109841 | 0.082543 | outputs/comparison_experiments/cube/ours_full_seed43_repeat1.log |
| cube | ours_full | 43 | 2 | 100 | 79.915700 | 19.925279 | 9.962640 | 2 | 2 | 0.021239 | 0.021239 | 8.500000 | 0.121895 | 1.359683 | 0.772264 | 0.111117 | 0.082896 | outputs/comparison_experiments/cube/ours_full_seed43_repeat2.log |
| cube | ours_full | 44 | 0 | 94 | 86.675500 | 17.757110 | 8.878555 | 2 | 2 | 0.024503 | 0.024204 | 11.531915 | 0.140636 | 6.735167 | 0.746807 | 0.116084 | 0.095628 | outputs/comparison_experiments/cube/ours_full_seed44_repeat0.log |
| cube | ours_full | 44 | 1 | 94 | 78.498300 | 18.084533 | 9.042266 | 2 | 2 | 0.024618 | 0.024370 | 11.510638 | 0.140646 | 3.180282 | 0.744787 | 0.115240 | 0.094373 | outputs/comparison_experiments/cube/ours_full_seed44_repeat1.log |
| cube | ours_full | 44 | 2 | 94 | 78.382300 | 18.627966 | 9.313983 | 2 | 2 | 0.025413 | 0.024786 | 11.489362 | 0.140742 | 1.322027 | 0.744873 | 0.116262 | 0.094732 | outputs/comparison_experiments/cube/ours_full_seed44_repeat2.log |
| pusht | mpc_cem | 42 | 0 | 90 | 938.129300 | 919.187731 | 459.593865 | 2 | 2 | 113.022844 | 58.146887 | 23.666667 | 262.871909 | 2.204120 | 0.246938 | 0.241304 | 0.407749 | outputs/comparison_experiments/pusht/mpc_cem_seed42_repeat0.log |
| pusht | mpc_cem | 42 | 1 | 90 | 940.218100 | 925.434188 | 462.717094 | 2 | 2 | 111.374046 | 58.102874 | 23.200000 | 260.473604 | 2.155537 | 0.248147 | 0.241614 | 0.408101 | outputs/comparison_experiments/pusht/mpc_cem_seed42_repeat1.log |
| pusht | mpc_cem | 42 | 2 | 90 | 921.660200 | 907.110201 | 453.555100 | 2 | 2 | 114.876599 | 59.246908 | 23.177778 | 260.215844 | 2.157661 | 0.248188 | 0.241893 | 0.408293 | outputs/comparison_experiments/pusht/mpc_cem_seed42_repeat2.log |
| pusht | mpc_cem | 43 | 0 | 82 | 957.044500 | 929.125176 | 464.562588 | 2 | 2 | 128.748149 | 64.383133 | 22.121951 | 288.822692 | 2.140404 | 0.264480 | 0.254442 | 0.431510 | outputs/comparison_experiments/pusht/mpc_cem_seed43_repeat0.log |
| pusht | mpc_cem | 43 | 1 | 82 | 958.711500 | 942.156781 | 471.078391 | 2 | 2 | 124.891532 | 64.954434 | 22.146341 | 289.115255 | 2.144260 | 0.264520 | 0.255200 | 0.433432 | outputs/comparison_experiments/pusht/mpc_cem_seed43_repeat1.log |
| pusht | mpc_cem | 43 | 2 | 84 | 959.553000 | 943.150149 | 471.575074 | 2 | 2 | 113.053305 | 65.024982 | 22.238095 | 276.355450 | 2.100429 | 0.261365 | 0.251432 | 0.427469 | outputs/comparison_experiments/pusht/mpc_cem_seed43_repeat2.log |
| pusht | mpc_cem | 44 | 0 | 90 | 950.448600 | 932.570086 | 466.285043 | 2 | 2 | 125.089418 | 65.681250 | 22.688889 | 271.509521 | 1.922082 | 0.259185 | 0.250580 | 0.427427 | outputs/comparison_experiments/pusht/mpc_cem_seed44_repeat0.log |
| pusht | mpc_cem | 44 | 1 | 90 | 936.808000 | 921.731167 | 460.865584 | 2 | 2 | 124.858023 | 65.449447 | 22.688889 | 271.505223 | 1.921936 | 0.259149 | 0.250534 | 0.427376 | outputs/comparison_experiments/pusht/mpc_cem_seed44_repeat1.log |
| pusht | mpc_cem | 44 | 2 | 90 | 813.881700 | 799.749448 | 399.874724 | 2 | 2 | 124.861717 | 65.448900 | 22.688889 | 271.489491 | 1.921970 | 0.259149 | 0.250534 | 0.427376 | outputs/comparison_experiments/pusht/mpc_cem_seed44_repeat2.log |
| pusht | ours_full | 42 | 0 | 94 | 33.741100 | 17.859045 | 8.929523 | 2 | 2 | 65.423601 | 56.253978 | 21.148936 | 225.281244 | 1.986289 | 0.232841 | 0.068621 | 0.051573 | outputs/comparison_experiments/pusht/ours_full_seed42_repeat0.log |
| pusht | ours_full | 42 | 1 | 96 | 33.298400 | 17.559114 | 8.779557 | 2 | 2 | 65.796011 | 54.690652 | 21.729167 | 225.636430 | 1.972772 | 0.232629 | 0.069548 | 0.051982 | outputs/comparison_experiments/pusht/ours_full_seed42_repeat1.log |
| pusht | ours_full | 42 | 2 | 92 | 33.156800 | 17.425483 | 8.712742 | 2 | 2 | 64.998681 | 55.300305 | 21.130435 | 229.849075 | 2.027690 | 0.232543 | 0.069312 | 0.052272 | outputs/comparison_experiments/pusht/ours_full_seed42_repeat2.log |
| pusht | ours_full | 43 | 0 | 94 | 33.375100 | 17.372868 | 8.686434 | 2 | 2 | 58.406144 | 50.160696 | 20.425532 | 214.829794 | 1.649561 | 0.232128 | 0.059122 | 0.041708 | outputs/comparison_experiments/pusht/ours_full_seed43_repeat0.log |
| pusht | ours_full | 43 | 1 | 94 | 33.531900 | 17.440395 | 8.720197 | 2 | 2 | 62.489860 | 52.781788 | 20 | 213.891747 | 1.654374 | 0.235057 | 0.060289 | 0.041709 | outputs/comparison_experiments/pusht/ours_full_seed43_repeat1.log |
| pusht | ours_full | 43 | 2 | 96 | 33.367900 | 17.198419 | 8.599209 | 2 | 2 | 56.301103 | 51.033580 | 20.395833 | 211.001897 | 1.628873 | 0.234059 | 0.059475 | 0.041924 | outputs/comparison_experiments/pusht/ours_full_seed43_repeat2.log |
| pusht | ours_full | 44 | 0 | 84 | 30.502500 | 14.575556 | 7.287778 | 2 | 2 | 72.701521 | 58.671882 | 20.428571 | 248.993398 | 1.813528 | 0.222320 | 0.062556 | 0.048733 | outputs/comparison_experiments/pusht/ours_full_seed44_repeat0.log |
| pusht | ours_full | 44 | 1 | 86 | 32.670200 | 17.117671 | 8.558836 | 2 | 2 | 65.261151 | 56.831154 | 20.232558 | 245.966031 | 1.721432 | 0.223894 | 0.061312 | 0.047672 | outputs/comparison_experiments/pusht/ours_full_seed44_repeat1.log |
| pusht | ours_full | 44 | 2 | 84 | 33.841900 | 17.222409 | 8.611205 | 2 | 2 | 73.475517 | 58.516418 | 21.142857 | 250.441616 | 1.842954 | 0.220377 | 0.062356 | 0.048893 | outputs/comparison_experiments/pusht/ours_full_seed44_repeat2.log |
| reacher | mpc_cem | 42 | 0 | 88 | 1076.424300 | 1038.447654 | 519.223827 | 2 | 2 | 0.051588 | 0.046636 | 28.159091 | 1.821521 | 3.160180 | 0.496009 | 0.637366 | 1.101708 | outputs/comparison_experiments/reacher/mpc_cem_seed42_repeat0.log |
| reacher | mpc_cem | 42 | 1 | 88 | 1078.053600 | 1040.796278 | 520.398139 | 2 | 2 | 0.051588 | 0.046636 | 28.159091 | 1.821521 | 3.160180 | 0.496009 | 0.637366 | 1.101708 | outputs/comparison_experiments/reacher/mpc_cem_seed42_repeat1.log |
| reacher | mpc_cem | 42 | 2 | 88 | 1074.407300 | 1037.559530 | 518.779765 | 2 | 2 | 0.051588 | 0.046636 | 28.159091 | 1.821521 | 3.160180 | 0.496009 | 0.637366 | 1.101708 | outputs/comparison_experiments/reacher/mpc_cem_seed42_repeat2.log |
| reacher | mpc_cem | 43 | 0 | 90 | 1080.449900 | 1041.068179 | 520.534090 | 2 | 2 | 0.052136 | 0.049911 | 27.600000 | 1.668581 | 3.204529 | 0.465897 | 0.635883 | 1.102028 | outputs/comparison_experiments/reacher/mpc_cem_seed43_repeat0.log |
| reacher | mpc_cem | 43 | 1 | 90 | 1074.244400 | 1038.168209 | 519.084105 | 2 | 2 | 0.052136 | 0.049911 | 27.600000 | 1.668581 | 3.204529 | 0.465897 | 0.635883 | 1.102028 | outputs/comparison_experiments/reacher/mpc_cem_seed43_repeat1.log |
| reacher | mpc_cem | 43 | 2 | 90 | 1055.062500 | 1016.262209 | 508.131105 | 2 | 2 | 0.052136 | 0.049911 | 27.600000 | 1.668581 | 3.204529 | 0.465897 | 0.635883 | 1.102028 | outputs/comparison_experiments/reacher/mpc_cem_seed43_repeat2.log |
| reacher | mpc_cem | 44 | 0 | 86 | 1100.367800 | 1060.885156 | 530.442578 | 2 | 2 | 0.054122 | 0.050093 | 30.581395 | 1.838186 | 3.947872 | 0.459034 | 0.621852 | 1.063078 | outputs/comparison_experiments/reacher/mpc_cem_seed44_repeat0.log |
| reacher | mpc_cem | 44 | 1 | 86 | 1090.149800 | 1053.435314 | 526.717657 | 2 | 2 | 0.054122 | 0.050093 | 30.581395 | 1.838186 | 3.947872 | 0.459034 | 0.621852 | 1.063078 | outputs/comparison_experiments/reacher/mpc_cem_seed44_repeat1.log |
| reacher | mpc_cem | 44 | 2 | 86 | 1043.254500 | 1004.113098 | 502.056549 | 2 | 2 | 0.054122 | 0.050093 | 30.581395 | 1.838186 | 3.947872 | 0.459034 | 0.621852 | 1.063078 | outputs/comparison_experiments/reacher/mpc_cem_seed44_repeat2.log |
| reacher | ours_full | 42 | 0 | 96 | 57.324000 | 18.581325 | 9.290663 | 2 | 2 | 0.053569 | 0.051002 | 28.270833 | 1.405622 | 2.422248 | 0.359387 | 0.441660 | 0.765757 | outputs/comparison_experiments/reacher/ours_full_seed42_repeat0.log |
| reacher | ours_full | 42 | 1 | 92 | 56.078300 | 17.458268 | 8.729134 | 2 | 2 | 0.052671 | 0.051323 | 30.304348 | 1.497378 | 2.521610 | 0.357284 | 0.443002 | 0.768394 | outputs/comparison_experiments/reacher/ours_full_seed42_repeat1.log |
| reacher | ours_full | 42 | 2 | 94 | 54.992100 | 17.155631 | 8.577816 | 2 | 2 | 0.053897 | 0.050967 | 30.893617 | 1.517039 | 2.528672 | 0.357478 | 0.437377 | 0.766261 | outputs/comparison_experiments/reacher/ours_full_seed42_repeat2.log |
| reacher | ours_full | 43 | 0 | 94 | 55.325100 | 18.134695 | 9.067348 | 2 | 2 | 0.050532 | 0.050070 | 25.744681 | 1.215216 | 2.514282 | 0.355987 | 0.464354 | 0.817482 | outputs/comparison_experiments/reacher/ours_full_seed43_repeat0.log |
| reacher | ours_full | 43 | 1 | 94 | 53.141000 | 17.228822 | 8.614411 | 2 | 2 | 0.051268 | 0.051268 | 29.127660 | 1.344001 | 2.969710 | 0.352504 | 0.463718 | 0.807127 | outputs/comparison_experiments/reacher/ours_full_seed43_repeat1.log |
| reacher | ours_full | 43 | 2 | 90 | 53.099200 | 17.257440 | 8.628720 | 2 | 2 | 0.053495 | 0.052959 | 26.844444 | 1.298766 | 2.720143 | 0.352177 | 0.460008 | 0.805543 | outputs/comparison_experiments/reacher/ours_full_seed43_repeat2.log |
| reacher | ours_full | 44 | 0 | 90 | 39.842700 | 13.383703 | 6.691851 | 2 | 2 | 0.054019 | 0.053068 | 27.311111 | 1.352755 | 2.885839 | 0.361297 | 0.454597 | 0.788729 | outputs/comparison_experiments/reacher/ours_full_seed44_repeat0.log |
| reacher | ours_full | 44 | 1 | 90 | 40.352000 | 14.195335 | 7.097667 | 2 | 2 | 0.053905 | 0.052887 | 29.044444 | 1.422066 | 3.202920 | 0.359258 | 0.451114 | 0.787634 | outputs/comparison_experiments/reacher/ours_full_seed44_repeat1.log |
| reacher | ours_full | 44 | 2 | 90 | 54.268600 | 17.360178 | 8.680089 | 2 | 2 | 0.050584 | 0.049136 | 28.733333 | 1.404153 | 2.938032 | 0.361286 | 0.460696 | 0.799234 | outputs/comparison_experiments/reacher/ours_full_seed44_repeat2.log |
| tworoom | mpc_cem | 42 | 0 | 84 | 1004.470700 | 981.540323 | 490.770161 | 2 | 2 | 22.151557 | 19.429327 | 15.642857 | 73.042928 | 3.798417 | 0.970228 | 1.228231 | 2.111252 | outputs/comparison_experiments/tworoom/mpc_cem_seed42_repeat0.log |
| tworoom | mpc_cem | 42 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | outputs/comparison_experiments/tworoom/mpc_cem_seed42_repeat1.log |

## Seed Summary

| Task | Method | Seed | Success Rate Mean | Success Rate Std | Eval Time Mean | Eval Time Std | Planning Time Mean | Planning Time Std | Action Jerk Mean | Action Jerk Std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | gc_idm | 42 |  |  |  |  |  |  |  |  |
| cube | mpc_cem | 42 | 68.000000 | 0.000000 | 615.189000 | 6.218308 | 568.817565 | 4.521430 | 1.895762 | 0.001568 |
| cube | mpc_cem | 43 | 72.666667 | 1.154701 | 611.754467 | 7.174732 | 559.377899 | 3.999074 | 1.844425 | 0.002091 |
| cube | mpc_cem | 44 | 64.000000 | 0.000000 | 604.772300 | 10.211755 | 552.511449 | 4.703237 | 1.899559 | 0.001715 |
| cube | ours_full | 42 | 100.000000 | 0.000000 | 83.967333 | 5.092701 | 19.729632 | 1.085085 | 0.086229 | 0.001027 |
| cube | ours_full | 43 | 100.000000 | 0.000000 | 87.105800 | 8.280635 | 19.287538 | 1.609016 | 0.082935 | 0.000413 |
| cube | ours_full | 44 | 94.000000 | 0.000000 | 81.185367 | 4.754949 | 18.156536 | 0.439870 | 0.094911 | 0.000646 |
| pusht | mpc_cem | 42 | 90.000000 | 0.000000 | 933.335867 | 10.165218 | 917.244040 | 9.315341 | 0.408048 | 0.000276 |
| pusht | mpc_cem | 43 | 82.666667 | 1.154701 | 958.436333 | 1.276687 | 938.144035 | 7.826338 | 0.430804 | 0.003044 |
| pusht | mpc_cem | 44 | 90.000000 | 0.000000 | 900.379433 | 75.219079 | 884.683567 | 73.754484 | 0.427393 | 0.000029 |
| pusht | ours_full | 42 | 94.000000 | 2.000000 | 33.398767 | 0.304806 | 17.614547 | 0.222033 | 0.051942 | 0.000351 |
| pusht | ours_full | 43 | 94.666667 | 1.154701 | 33.424967 | 0.092677 | 17.337227 | 0.124863 | 0.041780 | 0.000124 |
| pusht | ours_full | 44 | 84.666667 | 1.154701 | 32.338200 | 1.694275 | 16.305212 | 1.498841 | 0.048433 | 0.000664 |
| reacher | mpc_cem | 42 | 88.000000 | 0.000000 | 1076.295067 | 1.826582 | 1038.934487 | 1.672390 | 1.101708 | 0.000000 |
| reacher | mpc_cem | 43 | 90.000000 | 0.000000 | 1069.918933 | 13.234888 | 1031.832866 | 13.562318 | 1.102028 | 0.000000 |
| reacher | mpc_cem | 44 | 86.000000 | 0.000000 | 1077.924033 | 30.456268 | 1039.477856 | 30.852465 | 1.063078 | 0.000000 |
| reacher | ours_full | 42 | 94.000000 | 2.000000 | 56.131467 | 1.166859 | 17.731741 | 0.751160 | 0.766804 | 0.001400 |
| reacher | ours_full | 43 | 92.666667 | 2.309401 | 53.855100 | 1.273229 | 17.540319 | 0.514944 | 0.810051 | 0.006484 |
| reacher | ours_full | 44 | 90.000000 | 0.000000 | 44.821100 | 8.185737 | 14.979739 | 2.101084 | 0.791866 | 0.006405 |
| tworoom | mpc_cem | 42 | 84.000000 | 0.000000 | 1004.470700 | 0.000000 | 981.540323 | 0.000000 | 2.111252 | 0.000000 |

## Final Summary

| Task | Method | Success Rate Mean | Success Rate Std | Eval Time Mean | Eval Time Std | Planning Time Mean | Planning Time Std | Speedup vs MPC | Planning Speedup vs MPC | Final Goal Dist Mean | Min Goal Dist Mean | Steps To Success | Path Length | Straight Ratio | Action L2 | Action Delta | Action Jerk |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | gc_idm |  | 0.000000 |  | 0.000000 |  | 0.000000 |  |  |  |  |  |  |  |  |  |  |
| cube | mpc_cem | 68.222222 | 4.337605 | 610.571922 | 5.308080 | 560.235638 | 8.186827 | 1.000000 | 1.000000 | 0.066077 | 0.058074 | 4.125186 | 0.118568 | 8.501726 | 0.903787 | 1.077433 | 1.879915 |
| cube | ours_full | 98.000000 | 3.464102 | 84.086167 | 2.962005 | 19.057902 | 0.811300 | 7.261265 | 29.396501 | 0.023734 | 0.023603 | 10.201324 | 0.134700 | 2.122475 | 0.789824 | 0.112373 | 0.088025 |
| pusht | mpc_cem | 87.555556 | 4.233902 | 930.717211 | 29.116901 | 913.357214 | 26.941343 | 1.000000 | 1.000000 | 120.086181 | 62.937646 | 22.735278 | 272.484332 | 2.074267 | 0.256791 | 0.248615 | 0.422081 |
| pusht | ours_full | 91.111111 | 5.590998 | 33.053978 | 0.620020 | 17.085662 | 0.689966 | 28.157495 | 53.457525 | 64.983732 | 54.915606 | 20.737099 | 229.543470 | 1.810830 | 0.229539 | 0.063621 | 0.047385 |
| reacher | mpc_cem | 88.000000 | 2.000000 | 1074.712678 | 4.230647 | 1036.748403 | 4.265641 | 1.000000 | 1.000000 | 0.052615 | 0.048880 | 28.780162 | 1.776096 | 3.437527 | 0.473647 | 0.631700 | 1.088938 |
| reacher | ours_full | 92.222222 | 2.036700 | 51.602556 | 5.982187 | 16.750600 | 1.536594 | 20.826734 | 61.893211 | 0.052660 | 0.051409 | 28.474941 | 1.384111 | 2.744828 | 0.357406 | 0.452947 | 0.789573 |
| tworoom | mpc_cem | 84.000000 | 0.000000 | 1004.470700 | 0.000000 | 981.540323 | 0.000000 | 1.000000 | 1.000000 | 22.151557 | 19.429327 | 15.642857 | 73.042928 | 3.798417 | 0.970228 | 1.228231 | 2.111252 |

## Commands

完整命令保存在 CSV 的 `command` 字段；每次 run 的 stdout/stderr 保存在对应 `log_path`。
