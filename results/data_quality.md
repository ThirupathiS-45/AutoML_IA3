# Data quality report

- Crops: **Onion, Potato, Tomato, Cabbage**
- Entity level: **state** (state-wide aggregate of mandi modal prices, not a single mandi)
- Source: CEDA Agmarknet portal, Ashoka University (underlying data: Agmarknet, Ministry of Agriculture and Farmers Welfare, GoI)
- Requested range: 2018-01-01 to 2025-10-30
- Weekly frequency: `W-MON`
- Price unit: rupees per quintal

## Coverage by candidate series

| crop    | market         |   weeks_total |   weeks_present |   weeks_missing | missing_fraction   | first_week   | last_week   |   mean_price |
|:--------|:---------------|--------------:|----------------:|----------------:|:-------------------|:-------------|:------------|-------------:|
| Cabbage | Maharashtra    |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1013 |
| Cabbage | Karnataka      |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1004 |
| Cabbage | Gujarat        |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1213 |
| Cabbage | Rajasthan      |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1327 |
| Cabbage | Uttar Pradesh  |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1219 |
| Cabbage | West Bengal    |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1866 |
| Cabbage | Madhya Pradesh |           409 |             368 |              41 | 10.02%             | 2018-01-01   | 2025-10-27  |          960 |
| Cabbage | Bihar          |           409 |              82 |             327 | 79.95%             | 2018-01-01   | 2025-10-27  |         2232 |
| Onion   | Maharashtra    |           409 |             402 |               7 | 1.71%              | 2018-01-01   | 2025-10-27  |         1639 |
| Onion   | Karnataka      |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1869 |
| Onion   | Gujarat        |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1684 |
| Onion   | Madhya Pradesh |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1255 |
| Onion   | Rajasthan      |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1728 |
| Onion   | Uttar Pradesh  |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1862 |
| Onion   | West Bengal    |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         2521 |
| Onion   | Bihar          |           409 |             218 |             191 | 46.70%             | 2018-01-01   | 2025-10-27  |         2639 |
| Potato  | Maharashtra    |           409 |             407 |               2 | 0.49%              | 2018-01-01   | 2025-10-27  |         1575 |
| Potato  | Karnataka      |           409 |             407 |               2 | 0.49%              | 2018-01-01   | 2025-10-27  |         1851 |
| Potato  | Gujarat        |           409 |             408 |               1 | 0.24%              | 2018-01-01   | 2025-10-27  |         1416 |
| Potato  | Madhya Pradesh |           409 |             408 |               1 | 0.24%              | 2018-01-01   | 2025-10-27  |         1139 |
| Potato  | Rajasthan      |           409 |             408 |               1 | 0.24%              | 2018-01-01   | 2025-10-27  |         1207 |
| Potato  | Uttar Pradesh  |           409 |             408 |               1 | 0.24%              | 2018-01-01   | 2025-10-27  |         1165 |
| Potato  | West Bengal    |           409 |             408 |               1 | 0.24%              | 2018-01-01   | 2025-10-27  |         1464 |
| Potato  | Bihar          |           409 |             224 |             185 | 45.23%             | 2018-01-01   | 2025-10-27  |         1790 |
| Tomato  | Maharashtra    |           409 |             408 |               1 | 0.24%              | 2018-01-01   | 2025-10-27  |         1537 |
| Tomato  | Karnataka      |           409 |             408 |               1 | 0.24%              | 2018-01-01   | 2025-10-27  |         1486 |
| Tomato  | Gujarat        |           409 |             408 |               1 | 0.24%              | 2018-01-01   | 2025-10-27  |         1981 |
| Tomato  | Rajasthan      |           409 |             408 |               1 | 0.24%              | 2018-01-01   | 2025-10-27  |         1781 |
| Tomato  | Uttar Pradesh  |           409 |             408 |               1 | 0.24%              | 2018-01-01   | 2025-10-27  |         1758 |
| Tomato  | West Bengal    |           409 |             408 |               1 | 0.24%              | 2018-01-01   | 2025-10-27  |         2879 |
| Tomato  | Madhya Pradesh |           409 |             398 |              11 | 2.69%              | 2018-01-01   | 2025-10-27  |         1339 |
| Tomato  | Bihar          |           409 |              77 |             332 | 81.17%             | 2018-01-01   | 2025-10-27  |         2372 |

## Stop rule

**PASSED** - every selected series is within the 10% missing limit.

## Saved series

| crop    | market         |
|:--------|:---------------|
| Cabbage | Gujarat        |
| Cabbage | Karnataka      |
| Cabbage | Maharashtra    |
| Cabbage | Rajasthan      |
| Cabbage | Uttar Pradesh  |
| Cabbage | West Bengal    |
| Onion   | Gujarat        |
| Onion   | Karnataka      |
| Onion   | Madhya Pradesh |
| Onion   | Maharashtra    |
| Onion   | Rajasthan      |
| Onion   | Uttar Pradesh  |
| Potato  | Gujarat        |
| Potato  | Karnataka      |
| Potato  | Madhya Pradesh |
| Potato  | Maharashtra    |
| Potato  | Rajasthan      |
| Potato  | Uttar Pradesh  |
| Tomato  | Gujarat        |
| Tomato  | Karnataka      |
| Tomato  | Maharashtra    |
| Tomato  | Rajasthan      |
| Tomato  | Uttar Pradesh  |
| Tomato  | West Bengal    |

- Series kept: 24
- Rows: 9816
- Weeks per series: 409
