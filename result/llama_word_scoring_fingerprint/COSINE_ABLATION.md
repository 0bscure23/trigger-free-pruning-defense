# Cosine Ablation

Hybrid score formula: `protect - abs(proxy * cosine)` with old/new components crossed. Selection uses `layer>=2`, `score<=0`, cap 320.

| mode | selected | golden overlap | current61 overlap | layer hist |
|---|---:|---:|---:|---|
| `old_all` | 21 | 21 | 4 | `{29: 2, 30: 9, 31: 10}` |
| `new_all` | 61 | 4 | 61 | `{2: 1, 3: 1, 15: 1, 19: 1, 25: 2, 27: 1, 28: 4, 29: 13, 30: 16, 31: 21}` |
| `old_protect_old_proxy_new_cosine` | 11 | 5 | 3 | `{29: 2, 30: 1, 31: 8}` |
| `old_protect_new_proxy_old_cosine` | 45 | 12 | 9 | `{7: 1, 29: 5, 30: 10, 31: 29}` |
| `new_protect_old_proxy_old_cosine` | 50 | 16 | 10 | `{28: 1, 29: 8, 30: 10, 31: 31}` |
| `new_protect_new_proxy_old_cosine` | 116 | 10 | 26 | `{7: 1, 25: 1, 26: 2, 28: 1, 29: 19, 30: 28, 31: 64}` |
| `old_protect_new_proxy_new_cosine` | 19 | 5 | 12 | `{3: 1, 15: 1, 29: 2, 30: 2, 31: 13}` |
| `new_protect_old_proxy_new_cosine` | 23 | 5 | 11 | `{3: 1, 25: 1, 28: 3, 29: 2, 30: 4, 31: 12}` |

## Top 10 Per Mode
### `old_all`
- `channel:31:7000` score=-0.00130718
- `channel:29:2823` score=-0.00105258
- `channel:30:9255` score=-0.00097894
- `channel:30:6369` score=-0.00079696
- `channel:30:8030` score=-0.00067065
- `channel:29:13691` score=-0.00063236
- `channel:31:7124` score=-0.00058269
- `channel:31:6059` score=-0.00028346
- `channel:30:8429` score=-0.00022551
- `channel:31:4751` score=-0.00021977
### `new_all`
- `channel:29:2823` score=-0.00352978
- `channel:3:13690` score=-0.00277482
- `channel:31:7873` score=-0.00148343
- `channel:31:7124` score=-0.00132797
- `channel:31:5311` score=-0.00110402
- `channel:30:1623` score=-0.00087931
- `channel:30:8429` score=-0.00086581
- `channel:30:13628` score=-0.00081262
- `channel:30:7674` score=-0.00076678
- `channel:30:10006` score=-0.00064146
### `old_protect_old_proxy_new_cosine`
- `channel:31:7000` score=-0.00372049
- `channel:31:9241` score=-0.00121324
- `channel:29:2823` score=-0.0008634
- `channel:31:1526` score=-0.00070464
- `channel:31:3492` score=-0.00045933
- `channel:31:6059` score=-0.00038639
- `channel:31:683` score=-0.00031326
- `channel:31:5463` score=-0.00026865
- `channel:31:4933` score=-0.00022998
- `channel:29:4856` score=-1.998e-05
### `old_protect_new_proxy_old_cosine`
- `channel:30:6369` score=-0.00410024
- `channel:31:7124` score=-0.0039274
- `channel:29:2823` score=-0.00333133
- `channel:31:13995` score=-0.00193013
- `channel:30:6959` score=-0.00192566
- `channel:30:3843` score=-0.00175542
- `channel:30:8429` score=-0.00149103
- `channel:31:122` score=-0.00131831
- `channel:31:10310` score=-0.00124107
- `channel:31:9133` score=-0.00115199
### `new_protect_old_proxy_old_cosine`
- `channel:31:12185` score=-0.00262849
- `channel:30:9255` score=-0.00191647
- `channel:29:664` score=-0.00184129
- `channel:29:2823` score=-0.00161382
- `channel:30:6369` score=-0.00153868
- `channel:30:8030` score=-0.00149814
- `channel:31:7000` score=-0.00118464
- `channel:29:13691` score=-0.00116022
- `channel:31:3492` score=-0.00110036
- `channel:31:6059` score=-0.00074265
### `new_protect_new_proxy_old_cosine`
- `channel:30:6369` score=-0.00484196
- `channel:29:2823` score=-0.00389257
- `channel:31:7124` score=-0.00379484
- `channel:31:122` score=-0.00298116
- `channel:31:1674` score=-0.00242947
- `channel:31:422` score=-0.00235267
- `channel:31:9521` score=-0.00211193
- `channel:31:13995` score=-0.00195397
- `channel:31:8240` score=-0.00157856
- `channel:30:8429` score=-0.0015765
### `old_protect_new_proxy_new_cosine`
- `channel:29:2823` score=-0.00296854
- `channel:31:7124` score=-0.00146053
- `channel:31:5311` score=-0.00091758
- `channel:30:8429` score=-0.00078034
- `channel:3:13690` score=-0.00073682
- `channel:31:4933` score=-0.00065384
- `channel:31:13823` score=-0.00041843
- `channel:29:4856` score=-0.00030952
- `channel:31:2944` score=-0.00025336
- `channel:31:7280` score=-0.00022273
### `new_protect_old_proxy_new_cosine`
- `channel:31:7000` score=-0.00359794
- `channel:31:9241` score=-0.00172679
- `channel:29:2823` score=-0.00142464
- `channel:31:3492` score=-0.00139202
- `channel:3:13690` score=-0.00089659
- `channel:31:6059` score=-0.00084559
- `channel:31:1526` score=-0.00068604
- `channel:28:12074` score=-0.0003281
- `channel:31:12387` score=-0.00030487
- `channel:31:3700` score=-0.00022704
