# Four-route frozen middle comparison

Same local200 lower controller, friction 0.8, 10 Hz reconnection, prepared banks 1 and 3, five fixed alpha values and 10 s observation. These are model-selection development results; the relaxed task thresholds were set after inspecting U170's earlier task panel.

| model | route | alpha | original goal | H10 survival | narrow | wide | mean turn speed m/s | mean max original cross-track m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| U170 | tasktest_high_S_right_64_R2p9_v3p1 | 0 | 2/2 | 2/2 | 2/2 | 1/2 | 2.635 | 0.357 |
| U170 | tasktest_high_S_right_64_R2p9_v3p1 | 0.25 | 2/2 | 2/2 | 2/2 | 2/2 | 2.671 | 0.379 |
| U170 | tasktest_high_S_right_64_R2p9_v3p1 | 0.5 | 2/2 | 2/2 | 2/2 | 2/2 | 2.703 | 0.340 |
| U170 | tasktest_high_S_right_64_R2p9_v3p1 | 0.75 | 2/2 | 2/2 | 0/2 | 2/2 | 2.730 | 0.495 |
| U170 | tasktest_high_S_right_64_R2p9_v3p1 | 1 | 2/2 | 2/2 | 0/2 | 2/2 | 2.786 | 0.624 |
| U170 | tasktest_high_single_left_92_R3p05_v3p0 | 0 | 2/2 | 2/2 | 0/2 | 2/2 | 2.704 | 0.601 |
| U170 | tasktest_high_single_left_92_R3p05_v3p0 | 0.25 | 2/2 | 2/2 | 0/2 | 2/2 | 2.768 | 0.673 |
| U170 | tasktest_high_single_left_92_R3p05_v3p0 | 0.5 | 2/2 | 2/2 | 0/2 | 2/2 | 2.825 | 0.739 |
| U170 | tasktest_high_single_left_92_R3p05_v3p0 | 0.75 | 2/2 | 2/2 | 0/2 | 2/2 | 2.873 | 0.797 |
| U170 | tasktest_high_single_left_92_R3p05_v3p0 | 1 | 2/2 | 2/2 | 0/2 | 2/2 | 2.912 | 0.853 |
| U170 | tasktest_low_S_left_68_R1p35_v2p05 | 0 | 2/2 | 2/2 | 2/2 | 0/2 | 1.621 | 0.300 |
| U170 | tasktest_low_S_left_68_R1p35_v2p05 | 0.25 | 2/2 | 2/2 | 2/2 | 0/2 | 1.655 | 0.296 |
| U170 | tasktest_low_S_left_68_R1p35_v2p05 | 0.5 | 2/2 | 2/2 | 2/2 | 0/2 | 1.691 | 0.297 |
| U170 | tasktest_low_S_left_68_R1p35_v2p05 | 0.75 | 2/2 | 2/2 | 2/2 | 2/2 | 1.758 | 0.305 |
| U170 | tasktest_low_S_left_68_R1p35_v2p05 | 1 | 2/2 | 2/2 | 2/2 | 2/2 | 1.823 | 0.328 |
| U170 | tasktest_low_single_right_88_R1p2_v1p95 | 0 | 2/2 | 2/2 | 2/2 | 0/2 | 1.504 | 0.217 |
| U170 | tasktest_low_single_right_88_R1p2_v1p95 | 0.25 | 2/2 | 2/2 | 2/2 | 0/2 | 1.522 | 0.204 |
| U170 | tasktest_low_single_right_88_R1p2_v1p95 | 0.5 | 2/2 | 2/2 | 2/2 | 0/2 | 1.560 | 0.209 |
| U170 | tasktest_low_single_right_88_R1p2_v1p95 | 0.75 | 2/2 | 2/2 | 2/2 | 0/2 | 1.584 | 0.244 |
| U170 | tasktest_low_single_right_88_R1p2_v1p95 | 1 | 2/2 | 2/2 | 2/2 | 2/2 | 1.669 | 0.352 |
| H75 | tasktest_high_S_right_64_R2p9_v3p1 | 0 | 2/2 | 2/2 | 2/2 | 2/2 | 2.739 | 0.178 |
| H75 | tasktest_high_S_right_64_R2p9_v3p1 | 0.25 | 2/2 | 2/2 | 2/2 | 2/2 | 2.775 | 0.336 |
| H75 | tasktest_high_S_right_64_R2p9_v3p1 | 0.5 | 2/2 | 2/2 | 0/2 | 2/2 | 2.813 | 0.494 |
| H75 | tasktest_high_S_right_64_R2p9_v3p1 | 0.75 | 2/2 | 2/2 | 0/2 | 2/2 | 2.876 | 0.632 |
| H75 | tasktest_high_S_right_64_R2p9_v3p1 | 1 | 2/2 | 2/2 | 0/2 | 2/2 | 2.935 | 0.745 |
| H75 | tasktest_high_single_left_92_R3p05_v3p0 | 0 | 2/2 | 2/2 | 2/2 | 2/2 | 2.726 | 0.113 |
| H75 | tasktest_high_single_left_92_R3p05_v3p0 | 0.25 | 2/2 | 2/2 | 2/2 | 2/2 | 2.774 | 0.150 |
| H75 | tasktest_high_single_left_92_R3p05_v3p0 | 0.5 | 2/2 | 2/2 | 2/2 | 2/2 | 2.831 | 0.273 |
| H75 | tasktest_high_single_left_92_R3p05_v3p0 | 0.75 | 2/2 | 2/2 | 1/2 | 2/2 | 2.887 | 0.430 |
| H75 | tasktest_high_single_left_92_R3p05_v3p0 | 1 | 2/2 | 2/2 | 0/2 | 2/2 | 2.929 | 0.588 |
| H75 | tasktest_low_S_left_68_R1p35_v2p05 | 0 | 2/2 | 2/2 | 2/2 | 0/2 | 1.726 | 0.179 |
| H75 | tasktest_low_S_left_68_R1p35_v2p05 | 0.25 | 2/2 | 2/2 | 2/2 | 2/2 | 1.795 | 0.171 |
| H75 | tasktest_low_S_left_68_R1p35_v2p05 | 0.5 | 2/2 | 2/2 | 2/2 | 2/2 | 1.878 | 0.224 |
| H75 | tasktest_low_S_left_68_R1p35_v2p05 | 0.75 | 2/2 | 2/2 | 2/2 | 2/2 | 1.949 | 0.391 |
| H75 | tasktest_low_S_left_68_R1p35_v2p05 | 1 | 2/2 | 2/2 | 0/2 | 2/2 | 1.997 | 0.544 |
| H75 | tasktest_low_single_right_88_R1p2_v1p95 | 0 | 2/2 | 2/2 | 2/2 | 0/2 | 1.618 | 0.127 |
| H75 | tasktest_low_single_right_88_R1p2_v1p95 | 0.25 | 2/2 | 2/2 | 2/2 | 2/2 | 1.723 | 0.239 |
| H75 | tasktest_low_single_right_88_R1p2_v1p95 | 0.5 | 2/2 | 2/2 | 0/2 | 2/2 | 1.835 | 0.493 |
| H75 | tasktest_low_single_right_88_R1p2_v1p95 | 0.75 | 2/2 | 2/2 | 0/2 | 2/2 | 1.912 | 0.734 |
| H75 | tasktest_low_single_right_88_R1p2_v1p95 | 1 | 2/2 | 2/2 | 0/2 | 0/2 | 1.946 | 0.944 |
| W400 | tasktest_high_S_right_64_R2p9_v3p1 | 0 | 2/2 | 2/2 | 2/2 | 2/2 | 2.800 | 0.208 |
| W400 | tasktest_high_S_right_64_R2p9_v3p1 | 0.25 | 2/2 | 2/2 | 2/2 | 2/2 | 2.849 | 0.364 |
| W400 | tasktest_high_S_right_64_R2p9_v3p1 | 0.5 | 2/2 | 2/2 | 0/2 | 2/2 | 2.884 | 0.513 |
| W400 | tasktest_high_S_right_64_R2p9_v3p1 | 0.75 | 2/2 | 2/2 | 0/2 | 2/2 | 2.924 | 0.644 |
| W400 | tasktest_high_S_right_64_R2p9_v3p1 | 1 | 2/2 | 2/2 | 0/2 | 2/2 | 2.954 | 0.748 |
| W400 | tasktest_high_single_left_92_R3p05_v3p0 | 0 | 2/2 | 2/2 | 2/2 | 2/2 | 2.735 | 0.107 |
| W400 | tasktest_high_single_left_92_R3p05_v3p0 | 0.25 | 2/2 | 2/2 | 2/2 | 2/2 | 2.780 | 0.189 |
| W400 | tasktest_high_single_left_92_R3p05_v3p0 | 0.5 | 2/2 | 2/2 | 2/2 | 2/2 | 2.835 | 0.328 |
| W400 | tasktest_high_single_left_92_R3p05_v3p0 | 0.75 | 2/2 | 2/2 | 0/2 | 2/2 | 2.887 | 0.485 |
| W400 | tasktest_high_single_left_92_R3p05_v3p0 | 1 | 2/2 | 2/2 | 0/2 | 2/2 | 2.926 | 0.624 |
| W400 | tasktest_low_S_left_68_R1p35_v2p05 | 0 | 2/2 | 2/2 | 2/2 | 2/2 | 1.827 | 0.237 |
| W400 | tasktest_low_S_left_68_R1p35_v2p05 | 0.25 | 2/2 | 2/2 | 2/2 | 2/2 | 1.910 | 0.154 |
| W400 | tasktest_low_S_left_68_R1p35_v2p05 | 0.5 | 2/2 | 2/2 | 2/2 | 2/2 | 1.967 | 0.270 |
| W400 | tasktest_low_S_left_68_R1p35_v2p05 | 0.75 | 2/2 | 2/2 | 2/2 | 2/2 | 1.998 | 0.390 |
| W400 | tasktest_low_S_left_68_R1p35_v2p05 | 1 | 2/2 | 2/2 | 0/2 | 2/2 | 2.020 | 0.530 |
| W400 | tasktest_low_single_right_88_R1p2_v1p95 | 0 | 2/2 | 2/2 | 0/2 | 0/2 | 1.755 | 0.196 |
| W400 | tasktest_low_single_right_88_R1p2_v1p95 | 0.25 | 2/2 | 2/2 | 0/2 | 2/2 | 1.861 | 0.417 |
| W400 | tasktest_low_single_right_88_R1p2_v1p95 | 0.5 | 2/2 | 2/2 | 0/2 | 2/2 | 1.916 | 0.606 |
| W400 | tasktest_low_single_right_88_R1p2_v1p95 | 0.75 | 2/2 | 2/2 | 0/2 | 2/2 | 1.938 | 0.779 |
| W400 | tasktest_low_single_right_88_R1p2_v1p95 | 1 | 2/2 | 2/2 | 0/2 | 0/2 | 1.951 | 0.958 |

The machine-readable file contains every route, bank and alpha outcome. Actual/reference XY panels are retained in the source runs and the portable result package.
