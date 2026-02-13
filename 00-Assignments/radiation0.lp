\ Model Radiation Therapy
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  3 x[1] + 2 x[2] + 2.5 x[3] + x[4] + 2 x[5] + 2 x[6]
Subject To
 MinimalDoseCancer1: x[2] + 2 x[4] >= 7
 MinimalDoseCancer2: 1.5 x[3] + x[4] >= 7
 MinimalDoseCancer3: 2 x[1] + x[5] >= 7
 MinimalDoseCancer4: 1.5 x[3] + x[5] >= 7
 MinimalDoseCancer5: 2.5 x[2] + 2 x[6] >= 7
 MaximalDoseSpinal: 2 x[2] + 2 x[5] <= 5
Bounds
End
