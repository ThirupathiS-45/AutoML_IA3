# Data quality report

- Commodity: **onion**
- Entity level: **state** (state-wide aggregate of mandi modal prices, not a single mandi)
- Source: CEDA Agmarknet portal, Ashoka University (underlying data: Agmarknet, GoI)
- Requested range: 2018-01-01 to 2025-10-30
- Weekly frequency: `W-MON`

## Coverage by candidate market

| market         |   weeks_total |   weeks_present |   weeks_missing | missing_fraction   | first_week   | last_week   |   mean_price |
|:---------------|--------------:|----------------:|----------------:|:-------------------|:-------------|:------------|-------------:|
| Maharashtra    |           409 |             402 |               7 | 1.71%              | 2018-01-01   | 2025-10-27  |         1639 |
| Karnataka      |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1869 |
| Gujarat        |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1684 |
| Madhya Pradesh |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1255 |
| Rajasthan      |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1728 |
| Uttar Pradesh  |           409 |             404 |               5 | 1.22%              | 2018-01-01   | 2025-10-27  |         1862 |

## Stop rule

**PASSED** - every selected market is within the 10% missing limit.

## Saved series

- Markets kept: Gujarat, Karnataka, Maharashtra
- Rows: 1227
- Weeks per market: 409
