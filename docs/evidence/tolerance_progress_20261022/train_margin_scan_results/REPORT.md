# TRAIN同状态方向容扰度扫描

Frozen SPEC, no refit. Complete parent TRAIN prefix and fixed uniform suffix; conditional recoverability only. Grouped calibration/validation, descriptive correlated rows; no independent final TEST.

32状态×13力档；0.1秒脉冲，保留原20秒截止，无训练。每第4个状态预留validation，所有公式参数原样，未拟合。

物理失败0，路径域失败6，提前域退出导致的剩余物理存活UNKNOWN=6。

```json
{
  "calibration": {
    "all_cases": {
      "task_recovered": {
        "n": 312,
        "successes": 306,
        "predicted_within": 212,
        "false_positive": 0,
        "conservative": 94,
        "risk_rank_auc": 0.906318082788671
      },
      "performance_success": {
        "n": 312,
        "successes": 298,
        "predicted_within": 212,
        "false_positive": 5,
        "conservative": 91,
        "risk_rank_auc": 0.714046021093001
      },
      "physical_survived_horizon": {
        "n": 308,
        "successes": 308,
        "predicted_within": 212,
        "false_positive": 0,
        "conservative": 96,
        "risk_rank_auc": null
      }
    },
    "valid_nominal_nonzero": {
      "task_recovered": {
        "n": 288,
        "successes": 282,
        "predicted_within": 188,
        "false_positive": 0,
        "conservative": 94,
        "risk_rank_auc": 0.8983451536643026
      },
      "performance_success": {
        "n": 288,
        "successes": 274,
        "predicted_within": 188,
        "false_positive": 5,
        "conservative": 91,
        "risk_rank_auc": 0.7012513034410844
      },
      "physical_survived_horizon": {
        "n": 284,
        "successes": 284,
        "predicted_within": 188,
        "false_positive": 0,
        "conservative": 96,
        "risk_rank_auc": null
      }
    },
    "states": 24,
    "nominal_task_failures": 0
  },
  "validation": {
    "all_cases": {
      "task_recovered": {
        "n": 104,
        "successes": 102,
        "predicted_within": 69,
        "false_positive": 0,
        "conservative": 33,
        "risk_rank_auc": 0.9705882352941176
      },
      "performance_success": {
        "n": 104,
        "successes": 96,
        "predicted_within": 69,
        "false_positive": 6,
        "conservative": 33,
        "risk_rank_auc": 0.3802083333333333
      },
      "physical_survived_horizon": {
        "n": 102,
        "successes": 102,
        "predicted_within": 69,
        "false_positive": 0,
        "conservative": 33,
        "risk_rank_auc": null
      }
    },
    "valid_nominal_nonzero": {
      "task_recovered": {
        "n": 96,
        "successes": 94,
        "predicted_within": 61,
        "false_positive": 0,
        "conservative": 33,
        "risk_rank_auc": 0.9680851063829787
      },
      "performance_success": {
        "n": 96,
        "successes": 88,
        "predicted_within": 61,
        "false_positive": 6,
        "conservative": 33,
        "risk_rank_auc": 0.37642045454545453
      },
      "physical_survived_horizon": {
        "n": 94,
        "successes": 94,
        "predicted_within": 61,
        "false_positive": 0,
        "conservative": 33,
        "risk_rank_auc": null
      }
    },
    "states": 8,
    "nominal_task_failures": 0
  }
}
```

## 实际XY与误差

- [bank000](bank000.png)
- [bank016](bank016.png)
- [bank032](bank032.png)
- [bank049](bank049.png)
- [bank065](bank065.png)
- [bank082](bank082.png)
- [bank098](bank098.png)
- [bank115](bank115.png)
- [bank131](bank131.png)
- [bank148](bank148.png)
- [bank164](bank164.png)
- [bank181](bank181.png)
- [bank197](bank197.png)
- [bank214](bank214.png)
- [bank230](bank230.png)
- [bank247](bank247.png)
- [bank263](bank263.png)
- [bank280](bank280.png)
- [bank296](bank296.png)
- [bank313](bank313.png)
- [bank329](bank329.png)
- [bank346](bank346.png)
- [bank362](bank362.png)
- [bank379](bank379.png)
- [bank395](bank395.png)
- [bank412](bank412.png)
- [bank428](bank428.png)
- [bank445](bank445.png)
- [bank461](bank461.png)
- [bank478](bank478.png)
- [bank494](bank494.png)
- [bank511](bank511.png)

方向离散边界见summary.json：保留非单调、UNKNOWN、名义失败和右删失。最大通过力档不是连续安全边界；不把父前缀+新后缀宣称为同一Actor全程执行。
