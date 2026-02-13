\ Model Bus Schedule
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  Shifts[00_01] + Shifts[04_01] + Shifts[08_01] + Shifts[12_01]
   + Shifts[16_01] + Shifts[20_01]
Subject To
 Demand[0,00_01]: Shifts[00_01] + Shifts[20_01] >= 4
 Demand[1,04_01]: Shifts[00_01] + Shifts[04_01] >= 8
 Demand[2,08_01]: Shifts[04_01] + Shifts[08_01] >= 10
 Demand[3,12_01]: Shifts[08_01] + Shifts[12_01] >= 7
 Demand[4,16_01]: Shifts[12_01] + Shifts[16_01] >= 12
 Demand[5,20_01]: Shifts[16_01] + Shifts[20_01] >= 4
Bounds
Generals
 Shifts[00_01] Shifts[04_01] Shifts[08_01] Shifts[12_01] Shifts[16_01]
 Shifts[20_01]
End
