# Keepa Product Finder API Reference

**Token Costs**: 10 + 1 per 100 ASINs
**Endpoint**: `POST /query?domain=<domainId>&key=<yourAccessKey>`

## Query Object Structure

```json
{
  "rootCategory": Long array,
  "categories_include": Long array,
  "categories_exclude": Long array,
  "title": String,
  "brand": String array,
  "manufacturer": String array,
  "asins": String array,
  "perPage": Integer,
  "page": Integer,
  "sort": [["fieldName", "asc|desc"]]
}
```

## Critical Filters (High Value)

### Sales & Volume (Primary)
- `current_SALES_gte` / `_lte`: Sales Rank (BSR) - **The Gold Standard**. 100% coverage. Primary driver for estimates.
- `deltaPercent90_monthlySold_gte`: Growth trend

### Sales & Volume (Secondary / Validation)
- `monthlySold_gte`: "1K+ bought in past month" badge.
  - **WARNING**: Sparse data. Most ASINs do not have this.
  - Best used for **calibration** of BSR estimates, not primary filtering.

### Supply & Availability
- `outOfStockCountAmazon30_gte`: Amazon instability count
- `outOfStockPercentage90_gte`: Long-term availability issues
- `availabilityAmazon`: -1=No Offer, 0=In Stock, 1=Preorder
- `buyBoxIsBackorder`: Boolean

### Buy Box Intelligence
- `buyBoxStatsAmazon30_gte`: Amazon BB share %
- `buyBoxStatsTopSeller30_gte`: Dominant seller share %
- `buyBoxStatsSellerCount30_gte`: Number of rotating sellers
- `buyBoxIsAmazon`: Boolean
- `buyBoxIsFBA`: Boolean

### Product Attributes
- `variationCount_gte`: Size of product family
- `hasReviews`: Boolean
- `isSNS`: Subscribe & Save eligible
- `isHazMat`: Hazardous material
- `trackingSince_gte`: Launch date filtering

## Pricing Filters (Integer Cents)
Format: `current_[TYPE]_gte` / `_lte`

Types:
- `AMAZON`: Amazon 1P price
- `NEW`: Lowest New offer
- `USED`: Lowest Used offer
- `BUY_BOX_SHIPPING`: Buy Box price
- `LIGHTNING_DEAL`: Active deal price

## Response Format

```json
{
    "asinList": ["B0...", "B0..."],
    "totalResults": 150,
    "searchInsights": {...} // Only if &stats=1
}
```

## Domain IDs
1: com | 2: co.uk | 3: de | 4: fr | 5: co.jp | 6: ca | 8: it | 9: es | 10: in
