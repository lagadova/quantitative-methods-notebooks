\ Model Radiation Therapy
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  3 beam_left[1] + 2 beam_left[2] + 2.5 beam_left[3] + beam_upper[4]
   + 2 beam_upper[5] + 2 beam_upper[6]
Subject To
 MinimalDoseCancer[1,5]: 2 beam_left[1] + beam_upper[5] >= 7
 MinimalDoseCancer[2,4]: beam_left[2] + 2 beam_upper[4] >= 7
 MinimalDoseCancer[2,6]: 2.5 beam_left[2] + 2 beam_upper[6] >= 7
 MinimalDoseCancer[3,4]: 1.5 beam_left[3] + beam_upper[4] >= 7
 MinimalDoseCancer[3,5]: 1.5 beam_left[3] + beam_upper[5] >= 7
 MaximalDoseSpinal[2,5]: 2 beam_left[2] + 2 beam_upper[5] <= 5
Bounds
End
